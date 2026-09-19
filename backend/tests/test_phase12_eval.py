import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.evaluate import run_evaluation


def test_benchmark_dataset_integrity():
    dataset_path = ROOT_DIR / "demo_data" / "benchmark_dataset.json"
    assert dataset_path.exists(), f"Benchmark dataset file must exist at {dataset_path}"

    import json
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert len(cases) == 10, "Dataset must contain exactly 10 synthetic examples"
    categories = {c["category"] for c in cases}
    assert "Exact Copy" in categories
    assert "Word-Level Paraphrase" in categories
    assert "Heavy Paraphrase" in categories
    assert "English -> Hindi Translation" in categories
    assert "English -> Telugu Translation" in categories
    assert "Unrelated Text" in categories
    assert "Audio Transcript Similarity" in categories
    assert "Python Variable Renaming" in categories
    assert "Python Loop Refactoring" in categories
    assert "Completely Unrelated Code" in categories


def test_run_evaluation_metrics():
    report = run_evaluation()
    metrics = report["metrics"]

    assert metrics["total_cases"] == 10
    assert metrics["false_positives"] == 0, "No unrelated texts/code should trigger false positives"
    assert metrics["precision"] >= 0.80
    assert metrics["recall"] >= 0.80
    assert metrics["f1_score"] >= 0.80

    for r in report["results"]:
        assert "semantic_similarity" in r
        assert "lexical_similarity" in r
        assert "classification" in r
        assert "verdict" in r
        assert r["verdict"] in ["TP", "TN", "FP", "FN"]
