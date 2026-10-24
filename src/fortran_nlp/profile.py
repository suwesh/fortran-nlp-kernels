"""Machine-specific CPU calibration profile loading."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
from typing import Any


_PROFILE_ENVIRONMENT_VARIABLE = "FORTRAN_NLP_PROFILE"
_PROFILE_DIRECTORY = "fortran-nlp-kernels"
_PROFILE_FILENAME = "cpu_profile.json"

_SUPPORTED_PROFILE_SCHEMA_VERSION = 1
_PACKAGE_NAME = "fortran-nlp-kernels"


class ProfileError(RuntimeError):
    """Raised when an explicitly requested profile is unusable."""


@dataclass(frozen=True)
class ProfileLoadResult:
    """Result of attempting to load a machine calibration profile."""

    profile: dict[str, Any] | None
    path: Path
    status: str
    reason: str

    @property
    def compatible(self) -> bool:
        return self.profile is not None


def get_package_version() -> str:
    """Return the installed package version."""

    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return "unknown"


def get_available_logical_cpu_ids() -> tuple[int, ...]:
    """Return logical CPUs currently available to this process."""

    if hasattr(os, "sched_getaffinity"):
        try:
            return tuple(sorted(os.sched_getaffinity(0)))
        except OSError:
            pass

    logical_cpu_count = os.cpu_count()

    if logical_cpu_count is None:
        return ()

    return tuple(range(logical_cpu_count))


def get_linux_physical_core_count(
    logical_cpu_ids: tuple[int, ...],
) -> int | None:
    """
    Count physical cores available to this process using Linux sysfs.

    Each unique package/core pair represents one physical core.
    """

    if platform.system() != "Linux":
        return None

    physical_cores: set[tuple[int, int]] = set()

    for logical_cpu_id in logical_cpu_ids:
        topology_path = Path(
            f"/sys/devices/system/cpu/cpu{logical_cpu_id}/topology"
        )

        try:
            package_id = int(
                (
                    topology_path
                    / "physical_package_id"
                ).read_text(encoding="utf-8").strip()
            )

            core_id = int(
                (
                    topology_path
                    / "core_id"
                ).read_text(encoding="utf-8").strip()
            )
        except (OSError, ValueError):
            return None

        physical_cores.add(
            (
                package_id,
                core_id,
            )
        )

    if not physical_cores:
        return None

    return len(physical_cores)


def get_machine_fingerprint() -> dict[str, Any]:
    """Return machine attributes used for profile compatibility."""

    logical_cpu_ids = get_available_logical_cpu_ids()

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": len(logical_cpu_ids),
        "physical_core_count": (
            get_linux_physical_core_count(logical_cpu_ids)
        ),
    }


def get_default_profile_path() -> Path:
    """Return the default user calibration-profile path."""

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")

    if xdg_config_home:
        config_root = Path(xdg_config_home).expanduser()
    else:
        config_root = Path.home() / ".config"

    return (
        config_root
        / _PROFILE_DIRECTORY
        / _PROFILE_FILENAME
    )


def get_profile_path(
    explicit_path: str | os.PathLike[str] | None = None,
) -> Path:
    """
    Resolve the calibration-profile path.

    Precedence:
        1. Explicit function argument
        2. FORTRAN_NLP_PROFILE
        3. Default user configuration path
    """

    if explicit_path is not None:
        return Path(explicit_path).expanduser()

    environment_path = os.environ.get(
        _PROFILE_ENVIRONMENT_VARIABLE
    )

    if environment_path:
        return Path(environment_path).expanduser()

    return get_default_profile_path()


def _require_mapping(
    value: Any,
    *,
    field: str,
) -> dict[str, Any]:
    """Return a JSON object or raise a profile validation error."""

    if not isinstance(value, dict):
        raise ProfileError(
            f"CPU profile field '{field}' must be a JSON object"
        )

    return value


def validate_profile(
    profile: Any,
) -> dict[str, Any]:
    """Validate the minimum machine-profile structure."""

    root = _require_mapping(
        profile,
        field="<root>",
    )

    schema_version = root.get("schema_version")

    if schema_version != _SUPPORTED_PROFILE_SCHEMA_VERSION:
        raise ProfileError(
            "Unsupported CPU profile schema version: "
            f"{schema_version!r}. Expected "
            f"{_SUPPORTED_PROFILE_SCHEMA_VERSION}."
        )

    profile_version = root.get("profile_version")

    if not isinstance(profile_version, str) or not profile_version:
        raise ProfileError(
            "CPU profile field 'profile_version' "
            "must be a non-empty string"
        )

    package_version = root.get("package_version")

    if not isinstance(package_version, str) or not package_version:
        raise ProfileError(
            "CPU profile field 'package_version' "
            "must be a non-empty string"
        )

    fingerprint = _require_mapping(
        root.get("machine_fingerprint"),
        field="machine_fingerprint",
    )

    required_fingerprint_fields = {
        "system": str,
        "machine": str,
        "processor": str,
        "logical_cpu_count": int,
    }

    for field, expected_type in required_fingerprint_fields.items():
        value = fingerprint.get(field)

        if not isinstance(value, expected_type):
            raise ProfileError(
                "CPU profile field "
                f"'machine_fingerprint.{field}' must be of type "
                f"{expected_type.__name__}"
            )
    logical_cpu_count = fingerprint["logical_cpu_count"]
    if logical_cpu_count < 1:
        raise ProfileError(
            "CPU profile field 'machine_fingerprint.logical_cpu_count' must be at least 1"
        )

    physical_core_count = fingerprint.get("physical_core_count")

    if (
        physical_core_count is not None
        and not isinstance(physical_core_count, int)
    ):
        raise ProfileError(
            "CPU profile field 'machine_fingerprint.physical_core_count' must be an integer or null"
        )

    if (
        physical_core_count is not None
        and (
            physical_core_count < 1
            or physical_core_count > logical_cpu_count
        )
    ):
        raise ProfileError(
            "CPU profile field 'machine_fingerprint.physical_core_count' must be between 1 and logical_cpu_count"
        )

    decisions = _require_mapping(
        root.get("decisions"),
        field="decisions",
    )

    _require_mapping(
        decisions.get("lexical"),
        field="decisions.lexical",
    )

    _require_mapping(
        decisions.get("vector"),
        field="decisions.vector",
    )

    return root


def is_profile_compatible(
    profile: dict[str, Any],
) -> tuple[bool, str]:
    """Check whether a validated profile matches this installation."""

    profile_package_version = profile["package_version"]
    current_package_version = get_package_version()

    if profile_package_version != current_package_version:
        return (
            False,
            "Profile package version "
            f"{profile_package_version!r} does not match installed "
            f"package version {current_package_version!r}",
        )

    expected_fingerprint = get_machine_fingerprint()
    actual_fingerprint = profile["machine_fingerprint"]

    for field, expected_value in expected_fingerprint.items():
        actual_value = actual_fingerprint.get(field)

        if actual_value != expected_value:
            return (
                False,
                "Machine fingerprint mismatch for "
                f"{field!r}: profile={actual_value!r}, "
                f"current={expected_value!r}",
            )

    return True, "Profile is compatible"


def load_profile(
    path: str | os.PathLike[str] | None = None,
) -> ProfileLoadResult:
    """
    Load a compatible calibration profile.

    Missing, malformed, stale, and incompatible profiles return a
    result with profile=None. Runtime dispatch can then safely use the
    shipped fallback policy.
    """

    profile_path = get_profile_path(path)

    if not profile_path.is_file():
        return ProfileLoadResult(
            profile=None,
            path=profile_path,
            status="missing",
            reason="No CPU calibration profile exists",
        )

    try:
        with profile_path.open(
            "r",
            encoding="utf-8",
        ) as profile_file:
            raw_profile = json.load(profile_file)

        profile = validate_profile(raw_profile)

    except json.JSONDecodeError as error:
        return ProfileLoadResult(
            profile=None,
            path=profile_path,
            status="malformed",
            reason=f"CPU profile contains invalid JSON: {error}",
        )

    except (OSError, ProfileError) as error:
        return ProfileLoadResult(
            profile=None,
            path=profile_path,
            status="invalid",
            reason=str(error),
        )

    compatible, reason = is_profile_compatible(profile)

    if not compatible:
        return ProfileLoadResult(
            profile=None,
            path=profile_path,
            status="incompatible",
            reason=reason,
        )

    return ProfileLoadResult(
        profile=profile,
        path=profile_path,
        status="compatible",
        reason=reason,
    )