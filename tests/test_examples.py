"""Smoke test: the bundled quickstart example runs end to end offline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_QUICKSTART = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"


def _load_quickstart():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("quickstart", _QUICKSTART)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quickstart_runs_and_scores_sample() -> None:
    module = _load_quickstart()
    report = module.run()
    # The sample corpus/dataset are aligned, so retrieval should be perfect.
    assert report.num_examples == 4
    assert report.retrieval.hit_rate == 1.0
    assert report.mean_grounding is not None
