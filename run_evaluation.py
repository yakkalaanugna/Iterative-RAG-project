#!/usr/bin/env python3
"""
run_evaluation.py — End-to-end evaluation of Baseline vs Adaptive RAG.

Usage:
    python run_evaluation.py                    # requires GROQ_API_KEY in .env
    python run_evaluation.py --logs data/logs/  # custom log directory
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG methods on telecom logs")
    parser.add_argument("--logs", default="data/logs", help="Path to log files directory")
    parser.add_argument("--api-key", default=None, help="Groq API key (or set GROQ_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: Set GROQ_API_KEY in .env or pass --api-key")
        sys.exit(1)

    from rag_system.adaptive_agent import AdaptiveIterativeRAGAgent
    from rag_system.evaluator import RAGEvaluator, GroundTruth

    # ── Ground truth ───────────────────────────────────────────────────────
    ground_truths = [
        GroundTruth(
            query="Why did UE4 fail?",
            relevant_doc_ids=["log1.txt", "log2.txt", "log3.txt"],
            root_cause="RRC Reconfiguration failure code 4 in rfma_impl.cpp caused UE4 release",
            root_cause_keywords=[
                "rrcreconfiguration", "failure", "code 4", "rfma_impl",
                "ue4", "release",
            ],
            severity="CRITICAL",
        ),
        GroundTruth(
            query="What caused the UE context release?",
            relevant_doc_ids=["log1.txt", "log3.txt"],
            root_cause="UE Context Release triggered by CU-CP after RRC failure",
            root_cause_keywords=[
                "ue context release", "cu-cp", "rrc", "failure",
            ],
            severity="CRITICAL",
        ),
        GroundTruth(
            query="Why were packets lost for UE4?",
            relevant_doc_ids=["log3.txt", "log1.txt"],
            root_cause="Packet loss due to long forward jump after UE4 RRC failure and release",
            root_cause_keywords=[
                "forward jump", "packet", "lost", "ue4",
            ],
            severity="CRITICAL",
        ),
    ]

    queries = [gt.query for gt in ground_truths]

    # ── Setup ──────────────────────────────────────────────────────────────
    evaluator = RAGEvaluator()
    evaluator.add_ground_truths(ground_truths)

    agent = AdaptiveIterativeRAGAgent(
        groq_api_key=api_key,
        max_iterations=3,
        top_k=6,
    )

    print(f"Loading logs from {args.logs}...")
    records = agent.load_logs(args.logs)
    print(f"Parsed {len(records)} log records.\n")

    # ── Baseline RAG ───────────────────────────────────────────────────────
    print("=" * 60)
    print("Running Baseline RAG (single-pass, vector-only)...")
    print("=" * 60)

    for query in queries:
        start = time.time()
        result = agent.analyze_baseline(query)
        latency = time.time() - start
        evaluator.evaluate_single(
            method="Baseline",
            query=query,
            retrieved_doc_contents=result.get("supporting_logs", []),
            predicted_root_cause=result.get("root_cause", ""),
            confidence=result.get("confidence", 0.0),
            num_iterations=1,
            latency=latency,
            confidence_trajectory=result.get("confidence_trajectory", []),
            retrieval_scores=result.get("retrieval_scores", []),
        )
        print(f"  [{latency:.1f}s] {query[:50]}... → conf={result.get('confidence', 0):.3f}")

    # ── Adaptive RAG ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Running Adaptive RAG (iterative + cross-encoder reranking)...")
    print("=" * 60)

    for query in queries:
        start = time.time()
        result = agent.analyze(query)
        latency = time.time() - start
        evaluator.evaluate_single(
            method="Adaptive",
            query=query,
            retrieved_doc_contents=result.get("supporting_logs", []),
            predicted_root_cause=result.get("root_cause", ""),
            confidence=result.get("confidence", 0.0),
            num_iterations=result.get("total_iterations", 1),
            latency=latency,
            confidence_trajectory=result.get("confidence_trajectory", []),
            retrieval_scores=result.get("retrieval_scores", []),
        )
        iters = result.get("total_iterations", 1)
        print(f"  [{latency:.1f}s] {query[:50]}... → conf={result.get('confidence', 0):.3f} ({iters} iters)")

    # ── Results ────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    summary = evaluator.summary_by_method()
    print(summary.to_string())

    print()
    print("Per-query details:")
    df = evaluator.to_dataframe()
    print(df[["Method", "Query", "Precision@K", "Root Cause Accuracy", "Latency (s)"]].to_string(index=False))


if __name__ == "__main__":
    main()
