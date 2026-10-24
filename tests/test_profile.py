import json

from fortran_nlp.profile import (
    get_default_profile_path,
    get_machine_fingerprint,
    get_package_version,
    get_profile_path,
    load_profile,
    validate_profile,
)


def make_valid_profile() -> dict:
    """Create a profile compatible with the current test machine."""

    return {
        "schema_version": 1,
        "profile_version": "cpu-profile-v1",
        "package_version": get_package_version(),
        "machine_fingerprint": get_machine_fingerprint(),
        "decisions": {
            "lexical": {},
            "vector": {},
        },
    }


def write_profile(
    path,
    profile: dict,
) -> None:
    """Write a JSON profile for one test."""

    path.write_text(
        json.dumps(profile),
        encoding="utf-8",
    )


def test_default_profile_path_uses_xdg_config_home(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "XDG_CONFIG_HOME",
        str(tmp_path),
    )

    expected = (
        tmp_path
        / "fortran-nlp-kernels"
        / "cpu_profile.json"
    )

    assert get_default_profile_path() == expected


def test_default_profile_path_uses_home_config(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv(
        "XDG_CONFIG_HOME",
        raising=False,
    )

    monkeypatch.setattr(
        "pathlib.Path.home",
        lambda: tmp_path,
    )

    expected = (
        tmp_path
        / ".config"
        / "fortran-nlp-kernels"
        / "cpu_profile.json"
    )

    assert get_default_profile_path() == expected


def test_explicit_profile_path_has_highest_precedence(
    monkeypatch,
    tmp_path,
):
    environment_path = tmp_path / "environment.json"
    explicit_path = tmp_path / "explicit.json"

    monkeypatch.setenv(
        "FORTRAN_NLP_PROFILE",
        str(environment_path),
    )

    assert get_profile_path(explicit_path) == explicit_path


def test_environment_profile_path_precedes_default(
    monkeypatch,
    tmp_path,
):
    environment_path = tmp_path / "environment.json"

    monkeypatch.setenv(
        "FORTRAN_NLP_PROFILE",
        str(environment_path),
    )

    assert get_profile_path() == environment_path


def test_missing_profile_returns_safe_result(
    tmp_path,
):
    profile_path = tmp_path / "missing.json"

    result = load_profile(profile_path)

    assert result.profile is None
    assert result.path == profile_path
    assert result.status == "missing"
    assert result.compatible is False


def test_compatible_profile_loads_successfully(
    tmp_path,
):
    profile_path = tmp_path / "cpu_profile.json"

    write_profile(
        profile_path,
        make_valid_profile(),
    )

    result = load_profile(profile_path)

    assert result.profile is not None
    assert result.path == profile_path
    assert result.status == "compatible"
    assert result.compatible is True
    assert result.profile["schema_version"] == 1
    assert (
        result.profile["profile_version"]
        == "cpu-profile-v1"
    )


def test_malformed_json_returns_safe_result(
    tmp_path,
):
    profile_path = tmp_path / "cpu_profile.json"

    profile_path.write_text(
        "{not valid json",
        encoding="utf-8",
    )

    result = load_profile(profile_path)

    assert result.profile is None
    assert result.status == "malformed"
    assert result.compatible is False
    assert "invalid JSON" in result.reason


def test_invalid_schema_returns_safe_result(
    tmp_path,
):
    profile = make_valid_profile()
    profile["schema_version"] = 999

    profile_path = tmp_path / "cpu_profile.json"

    write_profile(
        profile_path,
        profile,
    )

    result = load_profile(profile_path)

    assert result.profile is None
    assert result.status == "invalid"
    assert result.compatible is False
    assert "schema version" in result.reason


def test_package_version_mismatch_is_incompatible(
    tmp_path,
):
    profile = make_valid_profile()
    profile["package_version"] = "0.0.0"

    profile_path = tmp_path / "cpu_profile.json"

    write_profile(
        profile_path,
        profile,
    )

    result = load_profile(profile_path)

    assert result.profile is None
    assert result.status == "incompatible"
    assert result.compatible is False
    assert "package version" in result.reason


def test_machine_fingerprint_mismatch_is_incompatible(
    tmp_path,
):
    profile = make_valid_profile()

    profile["machine_fingerprint"][
        "logical_cpu_count"
    ] += 1

    profile_path = tmp_path / "cpu_profile.json"

    write_profile(
        profile_path,
        profile,
    )

    result = load_profile(profile_path)

    assert result.profile is None
    assert result.status == "incompatible"
    assert result.compatible is False
    assert "logical_cpu_count" in result.reason


def test_missing_decisions_section_is_invalid(
    tmp_path,
):
    profile = make_valid_profile()
    del profile["decisions"]

    profile_path = tmp_path / "cpu_profile.json"

    write_profile(
        profile_path,
        profile,
    )

    result = load_profile(profile_path)

    assert result.profile is None
    assert result.status == "invalid"
    assert result.compatible is False
    assert "decisions" in result.reason


def test_validate_profile_returns_validated_mapping():
    profile = make_valid_profile()

    validated = validate_profile(profile)

    assert validated is profile
    assert validated["schema_version"] == 1
    assert "lexical" in validated["decisions"]
    assert "vector" in validated["decisions"]