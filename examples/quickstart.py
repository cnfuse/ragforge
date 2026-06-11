"""End-to-end ragforge demo: ingest → query → ask → evaluate.

Runs entirely offline against the bundled sample corpus (no API key needed).
With ANTHROPIC_API_KEY set, the `ask` step is answered by Claude instead of the
deterministic offline mock.

    python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from ragforge.agent import RagAgent
from ragforge.config import settings
from ragforge.eval import evaluate_answers, load_dataset
from ragforge.eval.report import EvalReport
from ragforge.ingestion.loaders import load_path
from ragforge.llm import build_llm
from ragforge.pipeline import Pipeline

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS = _ROOT / "data" / "sample" / "corpus"
_DATASET = _ROOT / "data" / "sample" / "qa.jsonl"


def run(corpus_dir: Path = _CORPUS, dataset_path: Path = _DATASET) -> EvalReport:
    """Run the full demo and return the evaluation report."""
    pipeline = Pipeline()
    docs = load_path(corpus_dir)
    indexed = pipeline.ingest(docs)
    print(f"Ingested {len(docs)} docs -> {indexed} chunks\n")

    print("Retrieval for 'memory safe systems language':")
    for rank, hit in enumerate(pipeline.retrieve("memory safe systems language", top_k=3), 1):
        print(f"  [{rank}] {hit.chunk.doc_id}  score={hit.score:.3f}")

    agent = RagAgent(pipeline, build_llm(settings))
    answer = agent.answer("Which language guarantees memory safety without a GC?")
    print(f"\nAgent answer ({answer.model}):\n  {answer.text}\n")

    report = evaluate_answers(pipeline, agent, load_dataset(dataset_path), k=3)
    print(report.summary())
    return report


if __name__ == "__main__":
    run()
