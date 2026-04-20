"""Tests for RAGEvaluator metrics."""

import pytest

from rag_system.evaluator import RAGEvaluator, GroundTruth, EvaluationResult


@pytest.fixture
def evaluator():
    return RAGEvaluator()


# ── Precision@K ────────────────────────────────────────────────────────────


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert RAGEvaluator.precision_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0

    def test_zero_precision(self):
        assert RAGEvaluator.precision_at_k(["x", "y", "z"], ["a", "b", "c"], 3) == 0.0

    def test_partial_precision(self):
        result = RAGEvaluator.precision_at_k(["a", "x", "b"], ["a", "b", "c"], 3)
        assert abs(result - 2 / 3) < 1e-6

    def test_k_zero(self):
        assert RAGEvaluator.precision_at_k(["a"], ["a"], 0) == 0.0

    def test_k_greater_than_retrieved(self):
        # Only 2 retrieved, k=5 → should still work (2 hits out of 5 slots)
        result = RAGEvaluator.precision_at_k(["a", "b"], ["a", "b"], 5)
        assert abs(result - 2 / 5) < 1e-6


# ── Recall@K ───────────────────────────────────────────────────────────────


class TestRecallAtK:
    def test_perfect_recall(self):
        assert RAGEvaluator.recall_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0

    def test_zero_recall(self):
        assert RAGEvaluator.recall_at_k(["x", "y"], ["a", "b"], 2) == 0.0

    def test_partial_recall(self):
        result = RAGEvaluator.recall_at_k(["a", "x"], ["a", "b"], 2)
        assert abs(result - 0.5) < 1e-6

    def test_empty_relevant(self):
        assert RAGEvaluator.recall_at_k(["a"], [], 1) == 0.0


# ── Root Cause Match ───────────────────────────────────────────────────────


class TestRootCauseMatch:
    def test_perfect_match(self):
        result = RAGEvaluator.root_cause_match(
            "RRC Reconfiguration failure code 4",
            ["rrc", "failure", "code 4"],
        )
        assert result == 1.0

    def test_partial_match(self):
        result = RAGEvaluator.root_cause_match(
            "Some RRC issue happened",
            ["rrc", "failure", "code 4"],
        )
        assert abs(result - 1 / 3) < 1e-6

    def test_no_match(self):
        result = RAGEvaluator.root_cause_match(
            "Everything is fine",
            ["rrc", "failure", "code 4"],
        )
        assert result == 0.0

    def test_empty_keywords(self):
        assert RAGEvaluator.root_cause_match("anything", []) == 0.0

    def test_case_insensitive(self):
        result = RAGEvaluator.root_cause_match(
            "RRC FAILURE",
            ["rrc", "failure"],
        )
        assert result == 1.0


# ── Ground truth management ───────────────────────────────────────────────


class TestGroundTruth:
    def test_add_and_retrieve(self, evaluator):
        gt = GroundTruth(
            query="test query",
            relevant_doc_ids=["doc1"],
            root_cause="test cause",
            root_cause_keywords=["test"],
        )
        evaluator.add_ground_truth(gt)
        assert "test query" in evaluator.ground_truths

    def test_add_multiple(self, evaluator):
        gts = [
            GroundTruth(query="q1", relevant_doc_ids=["d1"], root_cause="c1", root_cause_keywords=["k1"]),
            GroundTruth(query="q2", relevant_doc_ids=["d2"], root_cause="c2", root_cause_keywords=["k2"]),
        ]
        evaluator.add_ground_truths(gts)
        assert len(evaluator.ground_truths) == 2
