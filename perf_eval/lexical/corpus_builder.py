from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import re

LEXICAL_MARKERS = re.compile(
    r"(?m)^(TITLE|CONTEXT|STEPS|RULES):\s*"
)

@dataclass(frozen=True)
class LexicalDocument:
    source: str
    document_id: str
    text: str

def clean_manual_text_for_lexical(
    text: str,
) -> str:
    """
    Remove embedding-only structural markers while preserving
    their textual content.
    Example:

        TITLE: Login
        CONTEXT:
        Sales user...
    becomes:

        Login
        Sales user...
    """
    if not isinstance(text, str):
        raise TypeError(
            "manual chunk text must be a string"
        )
    return LEXICAL_MARKERS.sub(
        "",
        text,
    ).strip()

def reconstruct_manuals_for_lexical(
    metadata: Sequence[dict[str, Any]],
) -> list[LexicalDocument]:
    """
    Reconstruct complete manuals from ordered metadata chunks.
    Chunks are grouped by doc_id and ordered by vector_id. The
    resulting list contains one lexical document per complete manual.
    """
    chunks_by_doc: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row_index, item in enumerate(metadata):
        if "doc_id" not in item:
            raise ValueError(
                "Manual metadata row "
                f"{row_index} is missing 'doc_id'"
            )
        if "vector_id" not in item:
            raise ValueError(
                "Manual metadata row "
                f"{row_index} is missing 'vector_id'"
            )
        if "text" not in item:
            raise ValueError(
                "Manual metadata row "
                f"{row_index} is missing 'text'"
            )
        doc_id = str(item["doc_id"])
        chunks_by_doc[doc_id].append(item)
    reconstructed_documents: list[
        LexicalDocument
    ] = []
    for doc_id, chunks in chunks_by_doc.items():
        ordered_chunks = sorted(
            chunks,
            key=lambda item: item["vector_id"],
        )
        cleaned_chunks = [
            clean_manual_text_for_lexical(
                chunk["text"]
            )
            for chunk in ordered_chunks
        ]
        nonempty_chunks = [
            chunk_text
            for chunk_text in cleaned_chunks
            if chunk_text
        ]
        complete_text = "\n\n".join(
            nonempty_chunks
        ).strip()
        if not complete_text:
            raise ValueError(
                f"Reconstructed manual '{doc_id}' "
                "contains no lexical text"
            )
        reconstructed_documents.append(
            LexicalDocument(
                source="manuals",
                document_id=doc_id,
                text=complete_text,
            )
        )
    return reconstructed_documents

def build_direct_lexical_documents(
    source: str,
    metadata: Sequence[dict[str, Any]],
) -> list[LexicalDocument]:
    """
    Build one lexical document per metadata row.
    Direct sources are expected to contain a complete independently
    retrievable document in each row's 'text' field.
    """
    documents: list[LexicalDocument] = []
    for row_index, item in enumerate(metadata):
        if "text" not in item:
            raise ValueError(
                f"Metadata row {row_index} for source "
                f"'{source}' is missing 'text'"
            )
        text = item["text"]
        if not isinstance(text, str):
            raise TypeError(
                f"Metadata row {row_index} for source "
                f"'{source}' has non-string text"
            )
        cleaned_text = text.strip()
        if not cleaned_text:
            continue
        document_id = _resolve_document_id(
            source=source,
            item=item,
            row_index=row_index,
        )
        documents.append(
            LexicalDocument(
                source=source,
                document_id=document_id,
                text=cleaned_text,
            )
        )
    return documents

def _resolve_document_id(
    *,
    source: str,
    item: dict[str, Any],
    row_index: int,
) -> str:
    """
    Resolve a stable descriptive ID for experiment metadata.
    The package still uses positional result IDs. These IDs are retained
    only for corpus auditing and mapping benchmark rows back to their
    source records.
    """
    preferred_fields = (
        "doc_id",
        "document_id",
        "sop_id",
        "faq_id",
        "video_id",
        "chunk_id",
        "unit_id",
        "vector_id",
        "id",
    )
    for field_name in preferred_fields:
        if (
            field_name in item
            and item[field_name] is not None
        ):
            return str(item[field_name])
    return f"{source}-{row_index:06d}"

def prepare_lexical_documents(
    *,
    source: str,
    metadata: Sequence[dict[str, Any]],
    builder_name: str,
) -> list[LexicalDocument]:
    """
    Dispatch one source to its configured lexical builder.
    """
    if builder_name == "direct":
        return build_direct_lexical_documents(
            source,
            metadata,
        )
    if builder_name == "reconstruct_manuals":
        if source != "manuals":
            raise ValueError(
                "'reconstruct_manuals' is currently "
                "supported only for the manuals source"
            )
        return reconstruct_manuals_for_lexical(
            metadata
        )
    raise ValueError(
        f"Unsupported lexical builder "
        f"'{builder_name}' for source '{source}'"
    )

def combine_lexical_documents(
    documents_by_source: dict[
        str,
        Sequence[LexicalDocument],
    ],
) -> list[LexicalDocument]:
    """
    Combine source-specific documents while preserving the configured
    source order and each source's internal document order.
    """
    combined_documents: list[
        LexicalDocument
    ] = []
    for documents in documents_by_source.values():
        combined_documents.extend(documents)
    return combined_documents