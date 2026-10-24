"""CPU backend policy loading and runtime dispatch decisions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
from typing import Any, Literal


_POLICY_PACKAGE = "fortran_nlp.policies"
_POLICY_FILENAME = "cpu_fallback_v1.json"

_SUPPORTED_SCHEMA_VERSION = 1
_SUPPORTED_POLICY_VERSION = "cpu-fallback-v1"


DecisionSource = Literal[
    "explicit",
    "profile",
    "fallback",
]


class PolicyError(RuntimeError):
    """Raised when a packaged CPU fallback policy is unusable."""


@dataclass(frozen=True)
class BackendDecision:
    """
    Final backend decision for one search operation.

    The backend identifies the public/native backend family.
    The strategy identifies the concrete execution path within
    that backend.
    """

    backend: str
    strategy: str
    source: DecisionSource
    reason: str
    thread_policy: str | None = None


def _require_mapping(
    value: Any,
    *,
    field: str,
) -> dict[str, Any]:
    """Return a JSON object or raise a clear policy error."""

    if not isinstance(value, dict):
        raise PolicyError(
            f"CPU fallback policy field '{field}' "
            "must be a JSON object"
        )

    return value


def _validate_shipped_policy(
    policy: Any,
) -> dict[str, Any]:
    """Validate the minimum structure required by runtime dispatch."""

    root = _require_mapping(
        policy,
        field="<root>",
    )

    schema_version = root.get("schema_version")

    if schema_version != _SUPPORTED_SCHEMA_VERSION:
        raise PolicyError(
            "Unsupported CPU fallback policy schema version: "
            f"{schema_version!r}. Expected "
            f"{_SUPPORTED_SCHEMA_VERSION}."
        )

    policy_version = root.get("policy_version")

    if policy_version != _SUPPORTED_POLICY_VERSION:
        raise PolicyError(
            "Unsupported CPU fallback policy version: "
            f"{policy_version!r}. Expected "
            f"'{_SUPPORTED_POLICY_VERSION}'."
        )

    compatibility = _require_mapping(
        root.get("package_compatibility"),
        field="package_compatibility",
    )

    for field in (
        "minimum_version",
        "maximum_version_exclusive",
    ):
        value = compatibility.get(field)

        if not isinstance(value, str) or not value:
            raise PolicyError(
                "CPU fallback policy field "
                f"'package_compatibility.{field}' "
                "must be a non-empty string"
            )

    defaults = _require_mapping(
        root.get("defaults"),
        field="defaults",
    )

    required_defaults = {
        "openmp_thread_policy": str,
        "openblas_thread_policy": str,
        "full_smt_automatic": bool,
    }

    for field, expected_type in required_defaults.items():
        value = defaults.get(field)

        if not isinstance(value, expected_type):
            raise PolicyError(
                f"CPU fallback policy field 'defaults.{field}' "
                f"must be of type {expected_type.__name__}"
            )

    lexical = _require_mapping(
        root.get("lexical"),
        field="lexical",
    )

    for section in (
        "single",
        "batch",
        "memory",
    ):
        _require_mapping(
            lexical.get(section),
            field=f"lexical.{section}",
        )

    vector = _require_mapping(
        root.get("vector"),
        field="vector",
    )

    for metric in (
        "ip",
        "cosine",
        "l2",
    ):
        _require_mapping(
            vector.get(metric),
            field=f"vector.{metric}",
        )

    return root


@lru_cache(maxsize=1)
def _load_shipped_policy() -> dict[str, Any]:
    """
    Load and validate the immutable fallback policy from the package.

    The result is cached because the packaged resource cannot change
    during the lifetime of an installed package.
    """

    try:
        policy_resource = files(_POLICY_PACKAGE).joinpath(
            _POLICY_FILENAME
        )

        with policy_resource.open(
            "r",
            encoding="utf-8",
        ) as policy_file:
            raw_policy = json.load(policy_file)

    except FileNotFoundError as error:
        raise PolicyError(
            "Packaged CPU fallback policy is missing: "
            f"{_POLICY_PACKAGE}/{_POLICY_FILENAME}. "
            "The installation may be incomplete."
        ) from error

    except json.JSONDecodeError as error:
        raise PolicyError(
            "Packaged CPU fallback policy contains invalid JSON: "
            f"{error}"
        ) from error

    except (ModuleNotFoundError, OSError) as error:
        raise PolicyError(
            "Packaged CPU fallback policy could not be read: "
            f"{error}"
        ) from error

    return _validate_shipped_policy(raw_policy)


def get_shipped_policy() -> dict[str, Any]:
    """
    Return a copy of the validated shipped fallback policy.

    Returning a copy prevents callers from modifying the cached
    process-wide policy.
    """

    return deepcopy(_load_shipped_policy())