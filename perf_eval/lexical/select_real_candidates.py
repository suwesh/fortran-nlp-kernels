from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from fortran_nlp.lexical.preprocessing import (
    encode_text_shingles,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "perf_eval"
    / "config"
    / "lexical_cpu_matrix.local.yaml"
)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select short and long real candidate documents "
            "from each lexical corpus source."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()

def resolve_repository_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()

def load_config(config_path: Path) -> dict:
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping"
        )
    return config

def load_manifest(
    manifest_path: Path,
) -> list[dict]:
    documents = []
    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as manifest_file:
        for line_number, line in enumerate(
            manifest_file,
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            required_fields = (
                "positional_id",
                "source",
                "document_id",
                "text",
            )
            for field in required_fields:
                if field not in row:
                    raise ValueError(
                        f"Manifest line {line_number} "
                        f"is missing '{field}'"
                    )
            documents.append(row)
    if not documents:
        raise ValueError(
            "Document manifest contains no documents"
        )
    return documents

def calculate_shingle_counts(
    documents: list[dict],
    *,
    shingle_size: int,
) -> None:
    for document in documents:
        shingles = encode_text_shingles(
            document["text"],
            shingle_size=shingle_size,
        )
        document["shingle_count"] = int(
            shingles.size
        )

def select_candidates(
    documents: list[dict],
) -> list[dict]:
    documents_by_source: dict[
        str,
        list[dict],
    ] = {}
    for document in documents:
        documents_by_source.setdefault(
            document["source"],
            [],
        ).append(document)
    selected_candidates = []
    for source, source_documents in (
        documents_by_source.items()
    ):
        ordered_documents = sorted(
            source_documents,
            key=lambda document: (
                document["shingle_count"],
                document["positional_id"],
            ),
        )
        shortest = ordered_documents[0]
        longest = ordered_documents[-1]
        selections = (
            ("short", shortest),
            ("long", longest),
        )
        for length_category, document in selections:
            selected_candidates.append(
                {
                    "candidate_id": (
                        f"{source}-{length_category}"
                    ),
                    "category": length_category,
                    "source": source,
                    "origin_positional_id": (
                        document["positional_id"]
                    ),
                    "origin_document_id": (
                        document["document_id"]
                    ),
         "shingle_count": (
                        document["shingle_count"]
                    ),
                    "text": document["text"],
                }
            )
    return selected_candidates

def write_candidates(
    output_path: Path,
    candidates: list[dict],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for candidate in candidates:
            output_file.write(
                json.dumps(
                    candidate,
                    ensure_ascii=False,
                )
            )
            output_file.write("\n")

def main() -> None:
    arguments = parse_arguments()
    config_path = (
        arguments.config
        .expanduser()
        .resolve()
    )
    config = load_config(config_path)
    manifest_path = resolve_repository_path(
        config["packed_corpus"][
            "document_manifest_path"
        ]
    )
    output_path = resolve_repository_path(
        config["candidates"]["input_path"]
    )
    shingle_size = int(
        config["packed_corpus"]["shingle_size"]
    )
    if output_path.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Candidate output already exists: {output_path}. "
            "Pass --overwrite to replace it."
        )
    documents = load_manifest(manifest_path)
    calculate_shingle_counts(
        documents,
        shingle_size=shingle_size,
    )
    candidates = select_candidates(documents)
    write_candidates(
        output_path,
        candidates,
    )
    print("Real lexical candidate selection")
    print("=" * 64)
    for candidate in candidates:
        print(
            f"{candidate['candidate_id']}: "
            f"source={candidate['source']}, "
            f"shingles={candidate['shingle_count']}, "
            f"origin={candidate['origin_positional_id']}"
        )
    print("-" * 64)
    print(f"Candidates selected: {len(candidates)}")
    print(f"Output: {output_path}")
    print("=" * 64)
    print("Real lexical candidate selection: PASS")

if __name__ == "__main__":
    main()