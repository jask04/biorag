"""Build a Qdrant dense-retrieval index from the local NFCorpus.

Run from the project root (requires QDRANT_URL / QDRANT_API_KEY in .env and
``scripts/build_corpus.py`` to have been run first):

    uv run python scripts/build_index.py                 # full general index
    uv run python scripts/build_index.py --embedder biomedical
    uv run python scripts/build_index.py --limit 500     # quick smoke build
"""

from __future__ import annotations

import argparse

from biorag.corpus import load_documents
from biorag.embed import (
    CachedEmbedder,
    biomedical_embedder,
    general_embedder,
)
from biorag.index import QdrantIndex

EMBEDDERS = {
    "general": general_embedder,
    "biomedical": biomedical_embedder,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedder",
        choices=sorted(EMBEDDERS),
        default="general",
        help="Which embedding model to index with (default: general).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N documents (for a quick smoke build).",
    )
    args = parser.parse_args()

    documents = load_documents()
    if args.limit is not None:
        documents = documents[: args.limit]

    embedder = CachedEmbedder(EMBEDDERS[args.embedder]())
    index = QdrantIndex(embedder)

    print(f"Indexing {len(documents)} documents into '{index.collection}'")
    index.recreate()
    count = index.index_documents(documents)
    embedder.save()
    print(f"Upserted {count} chunks. Cache holds {len(embedder.cache)} vectors.")


if __name__ == "__main__":
    main()
