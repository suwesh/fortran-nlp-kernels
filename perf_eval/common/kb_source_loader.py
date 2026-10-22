from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import yaml

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "lexical_cpu_matrix.local.yaml"
)

def load_kb_config(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """
    Load and validate the local KB benchmark configuration.
    """
    resolved_config_path = (
        Path(config_path)
        .expanduser()
        .resolve()
    )
    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            "KB configuration file was not found: "
            f"{resolved_config_path}"
        )
    with resolved_config_path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(
            "KB configuration must contain a YAML mapping"
        )
    if "kb_root" not in config:
        raise ValueError(
            "KB configuration is missing 'kb_root'"
        )
    if "sources" not in config:
        raise ValueError(
            "KB configuration is missing 'sources'"
        )
    if not isinstance(config["sources"], dict):
        raise ValueError(
            "'sources' must contain a YAML mapping"
        )
    return config

def resolve_kb_root(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> Path:
    """
    Resolve the configured KB artifact directory.
    """
    config = load_kb_config(config_path)
    kb_root = (
        Path(config["kb_root"])
        .expanduser()
        .resolve()
    )
    if not kb_root.is_dir():
        raise FileNotFoundError(
            "Configured KB root directory was not found: "
            f"{kb_root}"
        )
    return kb_root

def get_source_config(
    source: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """
    Return and validate one source configuration.
    """
    config = load_kb_config(config_path)
    sources = config["sources"]
    if source not in sources:
        available_sources = ", ".join(
            sorted(sources)
        )
        raise ValueError(
            f"Unknown KB source '{source}'. "
            f"Available sources: {available_sources}"
        )
    source_config = sources[source]
    if not isinstance(source_config, dict):
        raise ValueError(
            f"Configuration for source '{source}' "
            "must contain a YAML mapping"
        )
    if "metadata" not in source_config:
        raise ValueError(
            f"Source '{source}' is missing "
            "the 'metadata' artifact"
        )
    return source_config

def resolve_metadata_path(
    source: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> Path:
    """
    Resolve and validate one metadata JSON path.
    """
    kb_root = resolve_kb_root(config_path)
    source_config = get_source_config(
        source,
        config_path,
    )
    metadata_path = (
        kb_root
        / source_config["metadata"]
    ).resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file for source '{source}' "
            f"was not found: {metadata_path}"
        )
    return metadata_path

def resolve_embedding_path(
    source: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> Path:
    """
    Resolve and validate one embedding-array path.
    Embeddings are not required for the current lexical experiment,
    but this function is retained for future vector-search evaluation.
    """
    kb_root = resolve_kb_root(config_path)
    source_config = get_source_config(
        source,
        config_path,
    )
    if "embeddings" not in source_config:
        raise ValueError(
            f"Source '{source}' does not define "
            "an 'embeddings' artifact"
        )
    embedding_path = (
        kb_root
        / source_config["embeddings"]
    ).resolve()
    if not embedding_path.is_file():
        raise FileNotFoundError(
            f"Embedding file for source '{source}' "
            f"was not found: {embedding_path}"
        )
    return embedding_path

def resolve_index_path(
    source: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> Path:
    """
    Resolve and validate one FAISS index path.
    The index is not loaded by this module because vector benchmarks
    may need to control FAISS construction and loading separately.
    """
    kb_root = resolve_kb_root(config_path)
    source_config = get_source_config(
        source,
        config_path,
    )
    if "index" not in source_config:
        raise ValueError(
            f"Source '{source}' does not define "
            "an 'index' artifact"
        )
    index_path = (
        kb_root
        / source_config["index"]
    ).resolve()
    if not index_path.is_file():
        raise FileNotFoundError(
            f"FAISS index for source '{source}' "
            f"was not found: {index_path}"
        )
    return index_path

def resolve_kb_paths(
    source: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> dict[str, Path]:
    """
    Resolve all artifacts configured for one source.
    Only configured artifact keys are returned.
    """
    source_config = get_source_config(
        source,
        config_path,
    )
    kb_root = resolve_kb_root(config_path)
    resolved_paths: dict[str, Path] = {}
    for artifact in (
        "metadata",
        "embeddings",
        "index",
    ):
        if artifact not in source_config:
            continue
        artifact_path = (
            kb_root
            / source_config[artifact]
        ).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Configured {artifact} artifact for "
                f"source '{source}' was not found: "
                f"{artifact_path}"
            )
        resolved_paths[artifact] = artifact_path
    return resolved_paths

def load_kb_metadata(
    source: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> list[dict[str, Any]]:
    """
    Load metadata without loading embeddings or a FAISS index.
    This is the preferred loader for lexical CPU experiments.
    """
    metadata_path = resolve_metadata_path(
        source,
        config_path,
    )
    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, list):
        raise ValueError(
            f"Metadata for source '{source}' "
            "must contain a JSON list"
        )
    for row_index, item in enumerate(metadata):
        if not isinstance(item, dict):
            raise ValueError(
                f"Metadata row {row_index} for source "
                f"'{source}' must contain a JSON object"
            )
    return metadata

def load_kb_embeddings(
    source: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    mmap_mode: str | None = "r",
) -> np.ndarray:
    """
    Load one rank-two embedding array.
    Memory mapping is enabled by default so future benchmarks can
    inspect large embedding files without immediately copying them.
    """
    embedding_path = resolve_embedding_path(
        source,
        config_path,
    )
    embeddings = np.load(
        embedding_path,
        mmap_mode=mmap_mode,
        allow_pickle=False,
    )
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings for source '{source}' must "
            "be a rank-two array; received shape "
            f"{embeddings.shape}"
        )
    return embeddings

def validate_metadata_embedding_alignment(
    source: str,
    metadata: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> None:
    """
    Verify positional alignment between metadata and embeddings.
    """
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            "Metadata/embedding alignment failure for "
            f"source '{source}': {len(metadata)} "
            "metadata rows versus "
            f"{embeddings.shape[0]} embedding rows"
        )

def load_kb_source(
    source: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    mmap_mode: str | None = "r",
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
]:
    """
    Load aligned metadata and embeddings for a future vector benchmark.
    """
    metadata = load_kb_metadata(
        source,
        config_path,
    )
    embeddings = load_kb_embeddings(
        source,
        config_path,
        mmap_mode=mmap_mode,
    )
    validate_metadata_embedding_alignment(
        source,
        metadata,
        embeddings,
    )
    return metadata, embeddings

def configured_sources(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> tuple[str, ...]:
    """
    Return configured source names in YAML insertion order.
    """
    config = load_kb_config(config_path)
    return tuple(config["sources"])

def lexical_builder_name(
    source: str,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> str:
    """
    Return the lexical-document builder assigned to one source.
    """
    source_config = get_source_config(
        source,
        config_path,
    )
    builder_name = source_config.get(
        "lexical_builder",
        "direct",
    )
    if not isinstance(builder_name, str):
        raise ValueError(
            f"'lexical_builder' for source '{source}' "
            "must be a string"
        )
    return builder_name