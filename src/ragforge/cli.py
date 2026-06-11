"""Command-line interface for ragforge.

Subcommands:
    ingest   Load a file or directory and build an on-disk index.
    query    Retrieve the most relevant chunks for a question from an index.

The CLI deliberately stays at the retrieval layer so it works with zero
configuration and no API key; the agent layer (Claude-backed answers) builds on
the same index format.
"""

from __future__ import annotations

import argparse
import sys

from ragforge.config import settings
from ragforge.embeddings import build_embedder
from ragforge.ingestion.chunking import chunk_document
from ragforge.ingestion.loaders import load_path
from ragforge.logging import configure, get_logger
from ragforge.store.memory import InMemoryVectorStore

log = get_logger("cli")
_DEFAULT_INDEX = ".ragforge_index.json"


def _cmd_ingest(args: argparse.Namespace) -> int:
    docs = load_path(args.path)
    if not docs:
        log.error("no readable documents found at %s", args.path)
        return 1
    embedder = build_embedder(settings)
    store = InMemoryVectorStore(dim=embedder.dim)
    chunks = []
    for doc in docs:
        chunks.extend(
            chunk_document(doc, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        )
    vectors = embedder.embed([c.text for c in chunks])
    store.add(chunks, vectors)
    store.save(args.index)
    log.info("ingested %d documents -> %d chunks, saved to %s", len(docs), len(chunks), args.index)
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    store = InMemoryVectorStore.load(args.index)
    embedder = build_embedder(settings)
    if embedder.dim != store.dim:
        log.error("embedder dim %d != index dim %d; re-ingest", embedder.dim, store.dim)
        return 1
    results = store.search(embedder.embed_one(args.question), top_k=args.top_k)
    if not results:
        print("No results.")
        return 0
    for rank, r in enumerate(results, start=1):
        snippet = r.chunk.text.replace("\n", " ")[:200]
        print(f"[{rank}] score={r.score:.4f}  doc={r.chunk.doc_id}")
        print(f"    {snippet}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragforge", description=__doc__)
    parser.add_argument("--log-level", default=settings.log_level)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index a file or directory")
    p_ingest.add_argument("path", help="path to a text file or directory")
    p_ingest.add_argument("--index", default=_DEFAULT_INDEX, help="output index path")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_query = sub.add_parser("query", help="retrieve chunks for a question")
    p_query.add_argument("question", help="the query string")
    p_query.add_argument("--index", default=_DEFAULT_INDEX, help="index path to search")
    p_query.add_argument("--top-k", type=int, default=settings.top_k)
    p_query.set_defaults(func=_cmd_query)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure(args.log_level)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
