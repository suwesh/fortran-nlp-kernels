import json

import pytest

from fortran_nlp.calibration import (
    build_profile,
    main,
    reset_profile,
    show_profile,
    write_profile_atomic,
)
from fortran_nlp.profile import (
    get_machine_fingerprint,
    get_package_version,
    load_profile,
)


def make_decisions() -> dict:
    """Create minimal decisions accepted by the profile schema."""

    return {
        "lexical": {
            "single": {
                "serial_max_document_count": 93,
                "serial_max_corpus_shingles": 16_359,
            },
            "batch": {
                "native_batch_min_candidates": 32,
            },
        },
        "vector": {
            "ip": {
                "single": {
                    "backend": "fortran_blas",
                    "strategy": "blas_single",
                    "thread_policy": "single_thread",
                },
            },
            "l2": {
                "single": {
                    "backend": "fortran_openmp",
                    "strategy": "openmp",
                    "thread_policy": "physical_cores",
                },
            },
        },
    }


def make_profile() -> dict:
    """Build a valid profile for the current test machine."""

    return build_profile(
        make_decisions()
    )


def test_build_profile_creates_valid_metadata():
    profile = make_profile()

    assert profile["schema_version"] == 1
    assert profile["profile_version"] == "cpu-profile-v1"
    assert profile["package_version"] == get_package_version()

    assert (
        profile["machine_fingerprint"]
        == get_machine_fingerprint()
    )

    assert profile["decisions"] == make_decisions()

    assert isinstance(
        profile["created_at_utc"],
        str,
    )

    assert profile["created_at_utc"]


def test_build_profile_requires_dictionary():
    with pytest.raises(
        TypeError,
        match="decisions must be a dictionary",
    ):
        build_profile(None)


@pytest.mark.parametrize(
    "missing_section",
    [
        "lexical",
        "vector",
    ],
)
def test_build_profile_requires_decision_sections(
    missing_section,
):
    decisions = make_decisions()

    del decisions[missing_section]

    with pytest.raises(
        ValueError,
        match=(
            rf"decisions\.{missing_section} "
            "must be a dictionary"
        ),
    ):
        build_profile(decisions)


def test_write_profile_atomic_creates_valid_profile(
    tmp_path,
):
    profile_path = (
        tmp_path
        / "profiles"
        / "cpu_profile.json"
    )

    written_path = write_profile_atomic(
        make_profile(),
        profile_path,
    )

    assert written_path == profile_path
    assert profile_path.is_file()

    result = load_profile(profile_path)

    assert result.compatible is True
    assert result.status == "compatible"
    assert result.profile is not None
    assert result.profile["decisions"] == make_decisions()


def test_write_profile_atomic_replaces_existing_profile(
    tmp_path,
):
    profile_path = tmp_path / "cpu_profile.json"

    first_profile = make_profile()

    write_profile_atomic(
        first_profile,
        profile_path,
    )

    second_decisions = make_decisions()
    second_decisions["lexical"]["single"][
        "serial_max_document_count"
    ] = 128

    second_profile = build_profile(
        second_decisions
    )

    write_profile_atomic(
        second_profile,
        profile_path,
    )

    stored_profile = json.loads(
        profile_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        stored_profile["decisions"]["lexical"]["single"][
            "serial_max_document_count"
        ]
        == 128
    )


def test_write_profile_atomic_leaves_no_temporary_files(
    tmp_path,
):
    profile_path = tmp_path / "cpu_profile.json"

    write_profile_atomic(
        make_profile(),
        profile_path,
    )

    temporary_files = list(
        tmp_path.glob(
            f".{profile_path.name}.*.tmp"
        )
    )

    assert temporary_files == []


def test_write_profile_atomic_rejects_invalid_profile(
    tmp_path,
):
    profile_path = tmp_path / "cpu_profile.json"

    invalid_profile = make_profile()
    invalid_profile["schema_version"] = 999

    with pytest.raises(
        RuntimeError,
        match="schema version",
    ):
        write_profile_atomic(
            invalid_profile,
            profile_path,
        )

    assert not profile_path.exists()


def test_show_profile_returns_compatible_profile(
    tmp_path,
):
    profile_path = tmp_path / "cpu_profile.json"

    profile = make_profile()

    write_profile_atomic(
        profile,
        profile_path,
    )

    shown_profile = show_profile(
        profile_path
    )

    assert shown_profile is not None
    assert shown_profile["decisions"] == profile["decisions"]


def test_show_profile_returns_none_when_missing(
    tmp_path,
):
    profile_path = tmp_path / "missing.json"

    assert show_profile(profile_path) is None


def test_show_profile_returns_none_when_incompatible(
    tmp_path,
):
    profile_path = tmp_path / "cpu_profile.json"

    profile = make_profile()
    profile["package_version"] = "0.0.0"

    profile_path.write_text(
        json.dumps(profile),
        encoding="utf-8",
    )

    assert show_profile(profile_path) is None


def test_reset_profile_removes_existing_profile(
    tmp_path,
):
    profile_path = tmp_path / "cpu_profile.json"

    write_profile_atomic(
        make_profile(),
        profile_path,
    )

    removed = reset_profile(
        profile_path
    )

    assert removed is True
    assert not profile_path.exists()


def test_reset_profile_returns_false_when_missing(
    tmp_path,
):
    profile_path = tmp_path / "missing.json"

    removed = reset_profile(
        profile_path
    )

    assert removed is False


def test_reset_profile_does_not_remove_other_files(
    tmp_path,
):
    profile_path = tmp_path / "cpu_profile.json"
    unrelated_path = tmp_path / "keep.json"

    write_profile_atomic(
        make_profile(),
        profile_path,
    )

    unrelated_path.write_text(
        "{}",
        encoding="utf-8",
    )

    reset_profile(profile_path)

    assert not profile_path.exists()
    assert unrelated_path.exists()


def test_cli_show_prints_compatible_profile(
    tmp_path,
    capsys,
):
    profile_path = tmp_path / "cpu_profile.json"

    write_profile_atomic(
        make_profile(),
        profile_path,
    )

    exit_code = main(
        [
            "--show",
            "--output",
            str(profile_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"profile_version": "cpu-profile-v1"' in captured.out
    assert '"machine_fingerprint"' in captured.out
    assert captured.err == ""


def test_cli_show_missing_profile_returns_one(
    tmp_path,
    capsys,
):
    profile_path = tmp_path / "missing.json"

    exit_code = main(
        [
            "--show",
            "--output",
            str(profile_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "No active compatible profile" in captured.out
    assert str(profile_path) in captured.out


def test_cli_reset_removes_profile(
    tmp_path,
    capsys,
):
    profile_path = tmp_path / "cpu_profile.json"

    write_profile_atomic(
        make_profile(),
        profile_path,
    )

    exit_code = main(
        [
            "--reset",
            "--output",
            str(profile_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert not profile_path.exists()
    assert "Removed CPU calibration profile" in captured.out


def test_cli_reset_missing_profile_succeeds(
    tmp_path,
    capsys,
):
    profile_path = tmp_path / "missing.json"

    exit_code = main(
        [
            "--reset",
            "--output",
            str(profile_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No CPU calibration profile existed" in captured.out


"""def test_cli_calibration_reports_not_implemented(
    tmp_path,
    capsys,
):
    profile_path = tmp_path / "cpu_profile.json"

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--output",
                str(profile_path),
            ]
        )

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert "not implemented yet" in captured.err
    assert not profile_path.exists()"""