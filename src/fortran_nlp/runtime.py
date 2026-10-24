"""Runtime resource discovery and thread-control utilities."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Iterator

from fortran_nlp import _cpu

from .profile import get_machine_fingerprint


_LINUX_MEMINFO_PATH = Path("/proc/meminfo")

_OPENMP_RUNTIME_LOCK = RLock()
_OPENBLAS_RUNTIME_LOCK = RLock()


def get_available_memory_bytes() -> int | None:
    """
    Return memory currently available for new allocations.

    Linux MemAvailable estimates how much memory can be allocated
    without swapping. None is returned when the value cannot be
    determined safely.
    """

    try:
        contents = _LINUX_MEMINFO_PATH.read_text(
            encoding="utf-8"
        )
    except OSError:
        return None

    for line in contents.splitlines():
        if not line.startswith("MemAvailable:"):
            continue

        fields = line.split()

        if len(fields) < 2:
            return None

        try:
            available_kibibytes = int(fields[1])
        except ValueError:
            return None

        if available_kibibytes < 0:
            return None

        return available_kibibytes * 1024

    return None


def get_openmp_max_threads() -> int:
    """Return the current OpenMP maximum thread count."""

    thread_count = int(
        _cpu.get_openmp_max_threads()
    )

    if thread_count < 1:
        raise RuntimeError(
            "OpenMP runtime returned an invalid thread count: "
            f"{thread_count}"
        )

    return thread_count


def set_openmp_num_threads(
    thread_count: int,
) -> None:
    """Set the OpenMP thread count for subsequent parallel regions."""

    if (
        not isinstance(thread_count, int)
        or isinstance(thread_count, bool)
    ):
        raise TypeError(
            "thread_count must be an integer"
        )

    if thread_count < 1:
        raise ValueError(
            "thread_count must be at least 1"
        )

    _cpu.set_openmp_num_threads(
        thread_count
    )


def resolve_openmp_thread_policy(
    thread_policy: str | int | None,
) -> int | None:
    """
    Resolve an OpenMP thread policy to a concrete thread count.

    None:
        Preserve the existing OpenMP runtime configuration.

    "physical_cores":
        Use the available physical-core count.

    "single_thread":
        Use one OpenMP thread.

    Positive integer:
        Use the explicitly specified thread count.
    """

    if thread_policy is None:
        return None

    if (
        isinstance(thread_policy, int)
        and not isinstance(thread_policy, bool)
    ):
        if thread_policy < 1:
            raise ValueError(
                "OpenMP thread count must be at least 1"
            )

        return thread_policy

    if not isinstance(thread_policy, str):
        raise TypeError(
            "thread_policy must be a string, integer, or None"
        )

    if thread_policy == "single_thread":
        return 1

    if thread_policy == "physical_cores":
        fingerprint = get_machine_fingerprint()

        physical_core_count = fingerprint.get(
            "physical_core_count"
        )

        if (
            not isinstance(physical_core_count, int)
            or isinstance(physical_core_count, bool)
            or physical_core_count < 1
        ):
            raise RuntimeError(
                "Physical-core thread policy could not be "
                "resolved on this machine"
            )

        return physical_core_count

    raise ValueError(
        f"Unsupported OpenMP thread policy: "
        f"{thread_policy!r}"
    )


@contextmanager
def apply_openmp_thread_policy(
    thread_policy: str | int | None,
) -> Iterator[int | None]:
    """
    Temporarily apply an OpenMP thread policy.

    The previous OpenMP thread count is restored even if the native
    operation raises. A process-local reentrant lock prevents package
    operations from changing and restoring the OpenMP configuration
    over one another.

    When thread_policy is None, no runtime state is changed.
    """

    selected_thread_count = resolve_openmp_thread_policy(
        thread_policy
    )

    if selected_thread_count is None:
        yield None
        return

    with _OPENMP_RUNTIME_LOCK:
        previous_thread_count = get_openmp_max_threads()

        set_openmp_num_threads(
            selected_thread_count
        )

        try:
            yield selected_thread_count
        finally:
            set_openmp_num_threads(
                previous_thread_count
            )

def get_openblas_num_threads() -> int:
    """Return the current OpenBLAS thread count."""

    thread_count = int(
        _cpu.get_openblas_num_threads()
    )

    if thread_count < 1:
        raise RuntimeError(
            "OpenBLAS returned an invalid thread count: "
            f"{thread_count}"
        )

    return thread_count


def set_openblas_num_threads(
    thread_count: int,
) -> None:
    """Set the thread count for subsequent OpenBLAS operations."""

    if (
        not isinstance(thread_count, int)
        or isinstance(thread_count, bool)
    ):
        raise TypeError(
            "thread_count must be an integer"
        )

    if thread_count < 1:
        raise ValueError(
            "thread_count must be at least 1"
        )

    _cpu.set_openblas_num_threads(
        thread_count
    )


def resolve_openblas_thread_policy(
    thread_policy: str | int | None,
) -> int | None:
    """Resolve an OpenBLAS policy to a concrete thread count."""

    if thread_policy is None:
        return None

    if (
        isinstance(thread_policy, int)
        and not isinstance(thread_policy, bool)
    ):
        if thread_policy < 1:
            raise ValueError(
                "OpenBLAS thread count must be at least 1"
            )

        return thread_policy

    if not isinstance(thread_policy, str):
        raise TypeError(
            "thread_policy must be a string, integer, or None"
        )

    if thread_policy == "single_thread":
        return 1

    if thread_policy == "physical_cores":
        fingerprint = get_machine_fingerprint()

        physical_core_count = fingerprint.get(
            "physical_core_count"
        )

        if (
            not isinstance(physical_core_count, int)
            or isinstance(physical_core_count, bool)
            or physical_core_count < 1
        ):
            raise RuntimeError(
                "Physical-core OpenBLAS policy could not be "
                "resolved on this machine"
            )

        return physical_core_count

    raise ValueError(
        "Unsupported OpenBLAS thread policy: "
        f"{thread_policy!r}"
    )


@contextmanager
def apply_openblas_thread_policy(
    thread_policy: str | int | None,
) -> Iterator[int | None]:
    """
    Temporarily apply an OpenBLAS thread policy.

    The previous thread count is restored after the operation,
    including when the native operation raises.
    """

    selected_thread_count = resolve_openblas_thread_policy(
        thread_policy
    )

    if selected_thread_count is None:
        yield None
        return

    with _OPENBLAS_RUNTIME_LOCK:
        previous_thread_count = get_openblas_num_threads()

        set_openblas_num_threads(
            selected_thread_count
        )

        try:
            yield selected_thread_count
        finally:
            set_openblas_num_threads(
                previous_thread_count
            )