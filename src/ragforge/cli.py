"""Command-line interface for ragforge.

Subcommands:
    ingest   Load a file or directory and build an on-disk index.
    query    Retrieve the most relevant chunks for a question from an index.
    ask      Answer a question with the agent (Claude when a key is set, else
             a deterministic offline stand-in), citing the corpus.

``ingest`` and ``query`` work with zero configuration and no API key; ``ask``
adds the agent layer on top of the same index format.
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
    try:
        docs = load_path(args.path)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
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
    try:
        store = InMemoryVectorStore.load(args.index)
    except (FileNotFoundError, ValueError) as exc:
        log.error("could not load index %s: %s", args.index, exc)
        return 1
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


def _cmd_ask(args: argparse.Namespace) -> int:
    from ragforge.agent import RagAgent
    from ragforge.llm import build_llm
    from ragforge.pipeline import Pipeline

    try:
        pipeline = Pipeline.from_index(args.index)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    agent = RagAgent(pipeline, build_llm(settings))

    if args.stream:
        answer = None
        for event in agent.iter_events(args.question):
            if event.type == "search":
                print(f"  - searching: {event.message}")
            elif event.type == "results":
                print(f"  - retrieved {event.data.get('count', 0)} passage(s)")
            if event.answer is not None:
                answer = event.answer
        print()
    else:
        answer = agent.answer(args.question)

    assert answer is not None
    print(answer.text)
    if answer.citations:
        print("\nSources:")
        for i, c in enumerate(answer.citations[: args.top_k], start=1):
            print(f"  [{i}] doc={c.chunk.doc_id}  score={c.score:.4f}")
    log.info("answered with model=%s", answer.model)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    import json

    from ragforge.agent import RagAgent
    from ragforge.eval import evaluate_answers, evaluate_retrieval, load_dataset
    from ragforge.llm import build_llm
    from ragforge.pipeline import Pipeline

    try:
        docs = load_path(args.corpus)
        dataset = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    if not docs:
        log.error("no documents found at %s", args.corpus)
        return 1

    if args.compare:
        from ragforge.eval import (
            compare_retrieval_configs,
            default_matrix,
            format_comparison,
        )

        results = compare_retrieval_configs(docs, dataset, default_matrix(), k=args.top_k)
        print(format_comparison(results))
        return 0

    pipeline = Pipeline()
    pipeline.ingest(docs)

    if args.retrieval_only:
        report = evaluate_retrieval(pipeline, dataset, k=args.top_k)
    else:
        agent = RagAgent(pipeline, build_llm(settings))
        report = evaluate_answers(pipeline, agent, dataset, k=args.top_k)

    print(report.summary())
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report.model_dump(), fh, indent=2)
        log.info("wrote report to %s", args.json_out)
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

    p_ask = sub.add_parser("ask", help="answer a question with the RAG agent")
    p_ask.add_argument("question", help="the question to answer")
    p_ask.add_argument("--index", default=_DEFAULT_INDEX, help="index path to search")
    p_ask.add_argument("--top-k", type=int, default=settings.top_k)
    p_ask.add_argument(
        "--stream", action="store_true", help="print the agent's progress as it works"
    )
    p_ask.set_defaults(func=_cmd_ask)

    p_eval = sub.add_parser("eval", help="evaluate retrieval and answer quality")
    p_eval.add_argument("--corpus", required=True, help="path to a corpus file or directory")
    p_eval.add_argument("--dataset", required=True, help="path to a JSONL QA dataset")
    p_eval.add_argument("--top-k", type=int, default=settings.top_k)
    p_eval.add_argument(
        "--retrieval-only", action="store_true", help="skip answer generation"
    )
    p_eval.add_argument(
        "--compare",
        action="store_true",
        help="compare retrieval configs (dense/hybrid/rerank/mmr) and print a table",
    )
    p_eval.add_argument("--json-out", help="optional path to write the full report as JSON")
    p_eval.set_defaults(func=_cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure(args.log_level)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
