#!/usr/bin/env python3
"""
run_all_experiments.py — Master experiment pipeline.

Runs all experiments, computes statistics, generates plots, and outputs
LaTeX-ready tables for the SIGIR workshop paper.

Experiments:
    1. Ablation study (6 configurations)
    2. Multi-LLM comparison (4 models, fixed retrieval)
    3. Retrieval-reasoning correlation analysis
    4. Failure mode classification
    5. Improved metric evaluation & justification
    6. Statistical significance testing

Output:
    - results/ablation_results.csv
    - results/multi_llm_results.csv
    - results/correlation_analysis.json
    - results/failure_modes.json
    - results/metric_analysis.json
    - results/significance_tests.json
    - results/latex_tables.tex
    - results/correlation_scatter.png
    - results/failure_mode_dist.png
    - results/metric_correlation.png

Usage:
    python run_all_experiments.py [--max-queries N] [--skip-multi-llm]
"""

import argparse
import json
import os
import sys
import time
import warnings
import functools
import re
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

print = functools.partial(print, flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def precision_at_k(retrieved_files, relevant_files, k):
    if k == 0: return 0.0
    top_k = retrieved_files[:k]
    relevant_set = set(relevant_files)
    hits = sum(1 for f in top_k if f in relevant_set)
    return hits / k

def recall_at_k(retrieved_files, relevant_files, k):
    if not relevant_files: return 0.0
    top_k = set(retrieved_files[:k])
    relevant_set = set(relevant_files)
    return len(top_k & relevant_set) / len(relevant_set)

def mrr_score(retrieved_files, relevant_files):
    relevant_set = set(relevant_files)
    for rank, f in enumerate(retrieved_files, 1):
        if f in relevant_set:
            return 1.0 / rank
    return 0.0

def root_cause_match(predicted, keywords):
    if not keywords or not predicted: return 0.0
    predicted_lower = predicted.lower()
    matches = sum(1 for kw in keywords if kw.lower() in predicted_lower)
    return matches / len(keywords)

def rouge_l_f1(prediction, reference):
    """ROUGE-L F1 score using LCS."""
    if not prediction or not reference: return 0.0
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens: return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0: return 0.0
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec)

def semantic_similarity(text1, text2, model):
    """Compute cosine similarity using sentence-transformers."""
    emb = model.encode([text1, text2], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))

def structured_score(analysis_text):
    """Score structured completeness of analysis output."""
    scores = {}
    text_lower = analysis_text.lower()
    # Root cause identification
    scores["root_cause"] = 1.0 if any(k in text_lower for k in ["root cause", "caused by", "failure originated"]) else 0.0
    # Timeline presence
    scores["timeline"] = 1.0 if any(k in text_lower for k in ["timeline", "sequence", "chain of events", "chronolog"]) else 0.0
    # Reasoning depth
    reasoning_markers = ["because", "therefore", "consequently", "this led to", "which caused", "resulting in"]
    scores["reasoning"] = min(sum(1 for m in reasoning_markers if m in text_lower) / 3.0, 1.0)
    # Specificity (file refs, error codes, timestamps)
    file_refs = len(re.findall(r'\w+\.\w+', text_lower))
    ts_refs = len(re.findall(r'\d{2}:\d{2}:\d{2}', text_lower))
    scores["specificity"] = min((file_refs + ts_refs) / 8.0, 1.0)
    # Evidence alignment
    scores["alignment"] = min(len(analysis_text) / 500.0, 1.0)
    return np.mean(list(scores.values()))

def composite_metric(semantic_sim, struct_score, rouge_l, keyword_acc):
    """Composite score: 0.35*sem + 0.30*struct + 0.20*rouge + 0.15*kw."""
    return 0.35 * semantic_sim + 0.30 * struct_score + 0.20 * rouge_l + 0.15 * keyword_acc


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_retrieved_files(result, relevant_files):
    """Extract file names from retrieved documents for metric computation."""
    retrieved = []
    for log_text in result.get("supporting_logs", []):
        matched = False
        for rel_file in relevant_files:
            if rel_file.lower() in log_text.lower():
                retrieved.append(rel_file)
                matched = True
                break
        if not matched:
            retrieved.append(f"__irrelevant_{len(retrieved)}")
    return retrieved

def run_single_query(config_fn, query_text, gt, keywords, rate_limit=0.3):
    """Run a single query through a config function with error handling."""
    time.sleep(rate_limit)  # Rate limiting
    start = time.time()
    try:
        result = config_fn(query_text)
    except Exception as e:
        print(f"    ERROR: {e}")
        return {
            "supporting_logs": [], "root_cause": "", "confidence": 0.0,
            "retrieval_scores": [], "total_iterations": 1,
            "latency_seconds": time.time() - start,
            "full_analysis": "", "reasoning_steps": [],
        }
    
    latency = time.time() - start
    result["latency_seconds"] = latency
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation(agent, queries, out_dir, rate_limit=0.3):
    """Run 6-configuration ablation study."""
    from run_ablation import AblationRunner
    
    runner = AblationRunner(agent)
    configs = [
        ("Dense-Only", runner.run_dense_only),
        ("BM25-Only", runner.run_bm25_only),
        ("Hybrid", runner.run_hybrid_no_rerank),
        ("Hybrid+Rerank", runner.run_hybrid_rerank),
        ("Hybrid+Iter", runner.run_hybrid_iterative),
        ("Full-System", runner.run_full_system),
    ]
    
    all_results = []
    
    for config_name, config_fn in configs:
        print(f"\n{'='*60}")
        print(f"  ABLATION: {config_name}")
        print(f"{'='*60}")
        
        for qi, q in enumerate(queries):
            result = run_single_query(
                config_fn, q["query"],
                q["ground_truth"], q.get("keywords", []),
                rate_limit=rate_limit
            )
            
            retrieved = extract_retrieved_files(result, q["ground_truth"]["relevant_files"])
            k = len(retrieved)
            
            row = {
                "config": config_name,
                "query_id": q["id"],
                "query": q["query"],
                "query_type": q["type"],
                "difficulty": q["difficulty"],
                "scenario_id": q["scenario_id"],
                "precision_at_k": round(precision_at_k(retrieved, q["ground_truth"]["relevant_files"], k), 4),
                "recall_at_k": round(recall_at_k(retrieved, q["ground_truth"]["relevant_files"], k), 4),
                "mrr": round(mrr_score(retrieved, q["ground_truth"]["relevant_files"]), 4),
                "root_cause_accuracy": round(root_cause_match(result.get("root_cause", ""), q.get("keywords", [])), 4),
                "confidence": round(result.get("confidence", 0.0), 4),
                "iterations": result.get("total_iterations", 1),
                "latency_s": round(result.get("latency_seconds", 0), 2),
                "predicted_root_cause": result.get("root_cause", ""),
                "full_analysis": result.get("full_analysis", ""),
            }
            all_results.append(row)
            
            if (qi + 1) % 10 == 0 or qi == len(queries) - 1:
                print(f"  [{qi+1:3d}/{len(queries)}] P@K={row['precision_at_k']:.3f} R@K={row['recall_at_k']:.3f} RCA={row['root_cause_accuracy']:.3f}")
    
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "ablation_results.csv", index=False)
    
    # Summary with std
    summary = df.groupby("config").agg(
        pk_mean=("precision_at_k", "mean"),
        pk_std=("precision_at_k", "std"),
        rk_mean=("recall_at_k", "mean"),
        rk_std=("recall_at_k", "std"),
        mrr_mean=("mrr", "mean"),
        mrr_std=("mrr", "std"),
        rca_mean=("root_cause_accuracy", "mean"),
        rca_std=("root_cause_accuracy", "std"),
        conf_mean=("confidence", "mean"),
        conf_std=("confidence", "std"),
        iter_mean=("iterations", "mean"),
        latency_mean=("latency_s", "mean"),
    ).round(4).reset_index()
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    
    # By query type
    type_summary = df.groupby(["config", "query_type"]).agg(
        pk_mean=("precision_at_k", "mean"),
        rk_mean=("recall_at_k", "mean"),
        rca_mean=("root_cause_accuracy", "mean"),
    ).round(4).reset_index()
    type_summary.to_csv(out_dir / "ablation_by_query_type.csv", index=False)
    
    print(f"\n  Ablation done. {len(all_results)} rows saved.")
    return df, summary


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MULTI-LLM COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def run_multi_llm(agent, queries, out_dir, rate_limit=0.5):
    """Compare 4 LLMs with identical fixed retrieval."""
    
    MODELS = [
        ("llama-3.3-70b-versatile", "Llama-3.3-70B", "70B"),
        ("llama-3.1-8b-instant", "Llama-3.1-8B", "8B"),
        ("mixtral-8x7b-32768", "Mixtral-8x7B", "46.7B"),
        ("gemma2-9b-it", "Gemma2-9B", "9B"),
    ]
    
    api_key = os.getenv("GROQ_API_KEY")
    all_results = []
    
    # Pre-compute fixed retrieval contexts for all queries
    print(f"\n{'='*60}")
    print(f"  MULTI-LLM: Pre-computing fixed retrieval")
    print(f"{'='*60}")
    
    fixed_contexts = []
    for qi, q in enumerate(queries):
        scored_docs = agent.retriever.retrieve_and_rerank(q["query"], top_k=agent.top_k, n_candidates=20)
        context_text = agent.retriever.format_retrieved(scored_docs)
        retrieved = extract_retrieved_files(
            {"supporting_logs": [f"[{sd.metadata.get('source', '')}] {sd.content}" for sd in scored_docs]},
            q["ground_truth"]["relevant_files"]
        )
        fixed_contexts.append({
            "context": context_text,
            "scored_docs": scored_docs,
            "retrieved_files": retrieved,
        })
    print(f"  Fixed retrieval computed for {len(queries)} queries.")
    
    from langchain_groq import ChatGroq
    from langchain_core.output_parsers import StrOutputParser
    
    for model_id, model_name, param_count in MODELS:
        print(f"\n  MODEL: {model_name} ({param_count})")
        print(f"  {'-'*40}")
        
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_id,
            max_tokens=2048,
            temperature=0,
        )
        chain = agent.ANALYSIS_PROMPT | llm | StrOutputParser()
        
        for qi, q in enumerate(queries):
            time.sleep(rate_limit)
            ctx = fixed_contexts[qi]
            start = time.time()
            
            try:
                analysis = chain.invoke({
                    "context": ctx["context"],
                    "question": q["query"],
                    "memory_context": "No prior incidents.",
                    "iteration": 1,
                    "max_iterations": 1,
                    "previous_findings": "",
                })
            except Exception as e:
                print(f"    [{qi+1}] ERROR: {e}")
                analysis = ""
            
            latency = time.time() - start
            parsed = agent._parse_analysis(analysis)
            rc = parsed.get("root_cause", "")
            rca = root_cause_match(rc, q.get("keywords", []))
            
            # Completeness scoring
            sections = ["root cause", "severity", "timeline", "details", "recommendation"]
            al = analysis.lower()
            completeness = sum(1 for s in sections if s in al) / len(sections)
            
            k = len(ctx["retrieved_files"])
            row = {
                "model": model_name,
                "model_id": model_id,
                "params": param_count,
                "query_id": q["id"],
                "query": q["query"],
                "query_type": q["type"],
                "precision_at_k": round(precision_at_k(ctx["retrieved_files"], q["ground_truth"]["relevant_files"], k), 4),
                "recall_at_k": round(recall_at_k(ctx["retrieved_files"], q["ground_truth"]["relevant_files"], k), 4),
                "root_cause_accuracy": round(rca, 4),
                "completeness": round(completeness, 4),
                "latency_s": round(latency, 2),
                "predicted_root_cause": rc,
                "full_analysis": analysis,
            }
            all_results.append(row)
            
            if (qi + 1) % 10 == 0 or qi == len(queries) - 1:
                print(f"    [{qi+1:3d}/{len(queries)}] RCA={rca:.3f} Comp={completeness:.2f} [{latency:.1f}s]")
    
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "multi_llm_results.csv", index=False)
    
    summary = df.groupby("model").agg(
        rca_mean=("root_cause_accuracy", "mean"),
        rca_std=("root_cause_accuracy", "std"),
        comp_mean=("completeness", "mean"),
        comp_std=("completeness", "std"),
        latency_mean=("latency_s", "mean"),
        pk_mean=("precision_at_k", "mean"),
    ).round(4).reset_index()
    summary.to_csv(out_dir / "multi_llm_summary.csv", index=False)
    
    print(f"\n  Multi-LLM done. {len(all_results)} rows saved.")
    return df, summary


# ═══════════════════════════════════════════════════════════════════════════════
# 3. IMPROVED METRICS EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_improved_metrics(ablation_df, queries, out_dir):
    """Compute improved evaluation metrics on ablation results."""
    print(f"\n{'='*60}")
    print(f"  IMPROVED METRICS EVALUATION")
    print(f"{'='*60}")
    
    # Load semantic model
    try:
        from sentence_transformers import SentenceTransformer
        sem_model = SentenceTransformer("all-MiniLM-L6-v2")
        has_sem = True
    except Exception:
        has_sem = False
        print("  WARNING: sentence-transformers not available, skipping semantic similarity")
    
    query_map = {q["id"]: q for q in queries}
    rows = []
    
    for _, r in ablation_df.iterrows():
        q = query_map.get(r["query_id"])
        if not q:
            continue
        
        predicted = r.get("predicted_root_cause", "") or ""
        gt_text = q["ground_truth"].get("root_cause", "")
        analysis = r.get("full_analysis", "") or ""
        kw_acc = r["root_cause_accuracy"]
        
        # Semantic similarity
        sem_sim = 0.0
        if has_sem and predicted and gt_text:
            sem_sim = semantic_similarity(predicted, gt_text, sem_model)
        
        # ROUGE-L
        rl = rouge_l_f1(predicted, gt_text)
        
        # Structured score
        ss = structured_score(analysis) if analysis else 0.0
        
        # Composite
        comp = composite_metric(sem_sim, ss, rl, kw_acc)
        
        rows.append({
            "config": r["config"],
            "query_id": r["query_id"],
            "query_type": r.get("query_type", ""),
            "keyword_accuracy": round(kw_acc, 4),
            "semantic_similarity": round(sem_sim, 4),
            "rouge_l": round(rl, 4),
            "structured_score": round(ss, 4),
            "composite_score": round(comp, 4),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "improved_metrics_results.csv", index=False)
    
    # Summary by config
    summary = df.groupby("config").agg(
        kw_mean=("keyword_accuracy", "mean"),
        kw_std=("keyword_accuracy", "std"),
        sem_mean=("semantic_similarity", "mean"),
        sem_std=("semantic_similarity", "std"),
        rouge_mean=("rouge_l", "mean"),
        rouge_std=("rouge_l", "std"),
        struct_mean=("structured_score", "mean"),
        struct_std=("structured_score", "std"),
        comp_mean=("composite_score", "mean"),
        comp_std=("composite_score", "std"),
    ).round(4).reset_index()
    summary.to_csv(out_dir / "improved_metrics_summary.csv", index=False)
    
    # Metric inter-correlation
    if len(df) > 5:
        from scipy import stats
        corrs = {}
        pairs = [
            ("keyword_accuracy", "semantic_similarity"),
            ("keyword_accuracy", "composite_score"),
            ("semantic_similarity", "rouge_l"),
            ("semantic_similarity", "composite_score"),
            ("rouge_l", "composite_score"),
        ]
        for m1, m2 in pairs:
            v1, v2 = df[m1].values, df[m2].values
            if np.std(v1) > 0 and np.std(v2) > 0:
                pr, pp = stats.pearsonr(v1, v2)
                sr, sp = stats.spearmanr(v1, v2)
            else:
                pr, pp, sr, sp = 0, 1, 0, 1
            corrs[f"{m1}_vs_{m2}"] = {
                "pearson_r": round(pr, 4), "pearson_p": round(pp, 4),
                "spearman_rho": round(sr, 4), "spearman_p": round(sp, 4),
            }
        
        with open(out_dir / "metric_correlations.json", "w") as f:
            json.dump(corrs, f, indent=2)
    
    print(f"  Improved metrics done. {len(rows)} rows.")
    return df, summary


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_correlation_analysis(ablation_df, out_dir):
    """Compute retrieval-reasoning correlation with Pearson & Spearman."""
    print(f"\n{'='*60}")
    print(f"  CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    from scipy import stats
    
    results = {}
    retrieval_metrics = ["precision_at_k", "recall_at_k", "mrr"]
    reasoning_metric = "root_cause_accuracy"
    
    for rm in retrieval_metrics:
        x = ablation_df[rm].values
        y = ablation_df[reasoning_metric].values
        
        # Filter out rows where both are 0 (uninformative)
        mask = ~((x == 0) & (y == 0))
        x_f, y_f = x[mask], y[mask]
        
        if len(x_f) > 2 and np.std(x_f) > 0 and np.std(y_f) > 0:
            pr, pp = stats.pearsonr(x_f, y_f)
            sr, sp = stats.spearmanr(x_f, y_f)
        else:
            pr, pp, sr, sp = 0.0, 1.0, 0.0, 1.0
        
        results[rm] = {
            "pearson_r": round(pr, 4),
            "pearson_p": round(pp, 6),
            "spearman_rho": round(sr, 4),
            "spearman_p": round(sp, 6),
            "n": int(len(x_f)),
        }
        print(f"  {rm:16s}: Pearson r={pr:.4f} (p={pp:.4f}), Spearman ρ={sr:.4f} (p={sp:.4f})")
    
    with open(out_dir / "correlation_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate scatter plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        labels = {"precision_at_k": "Precision@K", "recall_at_k": "Recall@K", "mrr": "MRR"}
        
        for ax, rm in zip(axes, retrieval_metrics):
            x = ablation_df[rm].values
            y = ablation_df[reasoning_metric].values
            mask = ~((x == 0) & (y == 0))
            x_f, y_f = x[mask], y[mask]
            
            # Color by config
            configs = ablation_df.loc[mask, "config"].values
            config_names = sorted(set(configs))
            colors = plt.cm.Set2(np.linspace(0, 1, len(config_names)))
            color_map = {c: colors[i] for i, c in enumerate(config_names)}
            
            for cfg in config_names:
                m = configs == cfg
                ax.scatter(x_f[m], y_f[m], c=[color_map[cfg]], label=cfg, alpha=0.6, s=30, edgecolors="gray", linewidth=0.3)
            
            # Regression line
            if len(x_f) > 2 and np.std(x_f) > 0:
                z = np.polyfit(x_f, y_f, 1)
                p_fn = np.poly1d(z)
                x_line = np.linspace(x_f.min(), x_f.max(), 50)
                ax.plot(x_line, p_fn(x_line), "r--", alpha=0.5, linewidth=1.5)
            
            r_val = results[rm]["pearson_r"]
            ax.set_xlabel(labels[rm], fontsize=11)
            ax.set_ylabel("Root Cause Accuracy", fontsize=11)
            ax.set_title(f"r = {r_val:.3f}", fontsize=10)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.2)
        
        axes[0].legend(fontsize=7, loc="upper left", framealpha=0.8)
        fig.suptitle("Retrieval vs Reasoning: The Retrieval–Reasoning Gap", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / "correlation_scatter.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved correlation_scatter.png")
    except Exception as e:
        print(f"  Plot error: {e}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FAILURE MODE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_failure_modes(ablation_df, out_dir, retrieval_thresh=0.5, reasoning_thresh=0.4):
    """Classify queries into 4 failure regimes."""
    print(f"\n{'='*60}")
    print(f"  FAILURE MODE ANALYSIS")
    print(f"{'='*60}")
    
    # Use adaptive thresholds: median of actual data
    pk_median = ablation_df["precision_at_k"].median()
    rca_median = ablation_df["root_cause_accuracy"].median()
    
    # Use max of provided threshold and median for meaningful splits
    r_thresh = max(retrieval_thresh, pk_median) if pk_median > 0 else retrieval_thresh
    a_thresh = max(reasoning_thresh, rca_median) if rca_median > 0 else reasoning_thresh
    
    print(f"  Thresholds: retrieval P@K > {r_thresh:.3f}, reasoning RCA > {a_thresh:.3f}")
    
    categories = {
        "High-Ret, High-Reas": (0, 0),
        "High-Ret, Low-Reas": (0, 0),
        "Low-Ret, High-Reas": (0, 0),
        "Low-Ret, Low-Reas": (0, 0),
    }
    
    query_classifications = []
    for _, row in ablation_df.iterrows():
        pk = row["precision_at_k"]
        rca = row["root_cause_accuracy"]
        
        high_ret = pk >= r_thresh
        high_reas = rca >= a_thresh
        
        if high_ret and high_reas:
            cat = "High-Ret, High-Reas"
        elif high_ret and not high_reas:
            cat = "High-Ret, Low-Reas"
        elif not high_ret and high_reas:
            cat = "Low-Ret, High-Reas"
        else:
            cat = "Low-Ret, Low-Reas"
        
        query_classifications.append({
            "config": row["config"],
            "query_id": row["query_id"],
            "precision_at_k": pk,
            "root_cause_accuracy": rca,
            "regime": cat,
        })
    
    cls_df = pd.DataFrame(query_classifications)
    
    # Count distribution
    dist = cls_df["regime"].value_counts().to_dict()
    total = len(cls_df)
    
    failure_results = {
        "thresholds": {"retrieval": r_thresh, "reasoning": a_thresh},
        "distribution": {},
        "total": total,
    }
    for regime in ["High-Ret, High-Reas", "High-Ret, Low-Reas", "Low-Ret, High-Reas", "Low-Ret, Low-Reas"]:
        count = dist.get(regime, 0)
        failure_results["distribution"][regime] = {
            "count": count,
            "percentage": round(100 * count / total, 1) if total > 0 else 0,
        }
        print(f"  {regime:25s}: {count:4d} ({100*count/total:.1f}%)")
    
    # Distribution by config
    config_dist = cls_df.groupby(["config", "regime"]).size().reset_index(name="count")
    failure_results["by_config"] = config_dist.to_dict(orient="records")
    
    with open(out_dir / "failure_modes.json", "w") as f:
        json.dump(failure_results, f, indent=2)
    
    cls_df.to_csv(out_dir / "failure_mode_classifications.csv", index=False)
    
    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        regimes = ["High-Ret, High-Reas", "High-Ret, Low-Reas", "Low-Ret, High-Reas", "Low-Ret, Low-Reas"]
        counts = [failure_results["distribution"][r]["count"] for r in regimes]
        colors = ["#2ecc71", "#e74c3c", "#3498db", "#95a5a6"]
        
        bars = ax.bar(range(4), counts, color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["High-Ret\nHigh-Reas", "High-Ret\nLow-Reas", "Low-Ret\nHigh-Reas", "Low-Ret\nLow-Reas"], fontsize=9)
        ax.set_ylabel("Query Count (across all configs)", fontsize=11)
        ax.set_title("Failure Mode Distribution", fontsize=13, fontweight="bold")
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{count}\n({100*count/total:.0f}%)", ha="center", va="bottom", fontsize=9)
        
        ax.grid(True, axis="y", alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / "failure_mode_dist.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved failure_mode_dist.png")
    except Exception as e:
        print(f"  Plot error: {e}")
    
    return failure_results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. STATISTICAL SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════════════════════

def run_significance_tests(ablation_df, out_dir):
    """Paired t-tests and bootstrap CIs between key configurations."""
    print(f"\n{'='*60}")
    print(f"  STATISTICAL SIGNIFICANCE TESTING")
    print(f"{'='*60}")
    
    from scipy import stats
    
    comparisons = [
        ("Hybrid", "Hybrid+Rerank"),
        ("Hybrid+Rerank", "Full-System"),
        ("Dense-Only", "Full-System"),
        ("Hybrid", "Full-System"),
    ]
    
    results = {}
    for config_a, config_b in comparisons:
        df_a = ablation_df[ablation_df["config"] == config_a].sort_values("query_id")
        df_b = ablation_df[ablation_df["config"] == config_b].sort_values("query_id")
        
        if len(df_a) == 0 or len(df_b) == 0:
            print(f"  SKIP {config_a} vs {config_b}: no data")
            continue
        
        # Align on query_id
        common_ids = set(df_a["query_id"]) & set(df_b["query_id"])
        df_a = df_a[df_a["query_id"].isin(common_ids)].sort_values("query_id")
        df_b = df_b[df_b["query_id"].isin(common_ids)].sort_values("query_id")
        
        comp_key = f"{config_a}_vs_{config_b}"
        results[comp_key] = {}
        
        for metric in ["precision_at_k", "root_cause_accuracy"]:
            a_vals = df_a[metric].values
            b_vals = df_b[metric].values
            diff = b_vals - a_vals
            
            # Paired t-test
            if np.std(diff) > 0:
                t_stat, p_val = stats.ttest_rel(a_vals, b_vals)
            else:
                t_stat, p_val = 0.0, 1.0
            
            # Bootstrap CI
            n_boot = 1000
            boot_diffs = []
            for _ in range(n_boot):
                idx = np.random.choice(len(diff), size=len(diff), replace=True)
                boot_diffs.append(np.mean(diff[idx]))
            ci_lower = np.percentile(boot_diffs, 2.5)
            ci_upper = np.percentile(boot_diffs, 97.5)
            
            results[comp_key][metric] = {
                "mean_a": round(float(np.mean(a_vals)), 4),
                "mean_b": round(float(np.mean(b_vals)), 4),
                "mean_diff": round(float(np.mean(diff)), 4),
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val), 6),
                "significant_005": bool(p_val < 0.05),
                "ci_95_lower": round(float(ci_lower), 4),
                "ci_95_upper": round(float(ci_upper), 4),
                "n": int(len(diff)),
            }
            
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            label = metric.replace("_", " ").title()
            print(f"  {config_a} vs {config_b} [{label}]: Δ={np.mean(diff):+.4f}, p={p_val:.4f} {sig}")
    
    with open(out_dir / "significance_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"  Significance tests done.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LATEX TABLE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_latex_tables(ablation_summary, multi_llm_summary, correlation_results,
                          failure_results, metric_summary, significance_results,
                          out_dir):
    """Generate LaTeX-ready table code."""
    print(f"\n{'='*60}")
    print(f"  GENERATING LATEX TABLES")
    print(f"{'='*60}")
    
    lines = []
    lines.append("% ═══ AUTO-GENERATED LATEX TABLES ═══\n")
    
    # --- Table 2: Ablation Results ---
    lines.append("% Table 2: Ablation Results")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Ablation results (65 queries, Llama 3.3 70B). $\\pm$ values are std. dev.}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Config} & \\textbf{P@K} & \\textbf{R@K} & \\textbf{MRR} & \\textbf{RCA\\textsubscript{kw}} & \\textbf{Conf.} & \\textbf{Iters} \\\\")
    lines.append("\\midrule")
    
    config_order = ["Dense-Only", "BM25-Only", "Hybrid", "Hybrid+Rerank", "Hybrid+Iter", "Full-System"]
    for cfg in config_order:
        row = ablation_summary[ablation_summary["config"] == cfg]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        
        def fmt(mean_col, std_col):
            m = r[mean_col]
            s = r[std_col]
            if pd.isna(s) or s == 0:
                return f"{m:.3f}"
            return f"{m:.3f}$\\pm${s:.2f}"
        
        line = f"{cfg:20s} & {fmt('pk_mean','pk_std')} & {fmt('rk_mean','rk_std')} & {fmt('mrr_mean','mrr_std')} & {fmt('rca_mean','rca_std')} & {r['conf_mean']:.3f} & {r['iter_mean']:.1f} \\\\"
        lines.append(line)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}\n")
    
    # --- Table 3: Multi-LLM ---
    if multi_llm_summary is not None:
        lines.append("% Table 3: Multi-LLM Comparison")
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\caption{Multi-LLM results with identical retrieval (Hybrid+Rerank, top-6).}")
        lines.append("\\label{tab:multi-llm}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("\\textbf{Model} & \\textbf{RCA\\textsubscript{kw}} & \\textbf{Completeness} & \\textbf{Latency (s)} & \\textbf{Params} \\\\")
        lines.append("\\midrule")
        
        model_order = ["Llama-3.3-70B", "Mixtral-8x7B", "Gemma2-9B", "Llama-3.1-8B"]
        for model in model_order:
            row = multi_llm_summary[multi_llm_summary["model"] == model]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            rca_str = f"{r['rca_mean']:.3f}$\\pm${r['rca_std']:.2f}" if not pd.isna(r.get('rca_std', 0)) else f"{r['rca_mean']:.3f}"
            comp_str = f"{r['comp_mean']:.3f}$\\pm${r['comp_std']:.2f}" if not pd.isna(r.get('comp_std', 0)) else f"{r['comp_mean']:.3f}"
            
            # Find params from model name
            param_map = {"Llama-3.3-70B": "70B", "Mixtral-8x7B": "46.7B", "Gemma2-9B": "9B", "Llama-3.1-8B": "8B"}
            params = param_map.get(model, "")
            
            lines.append(f"{model:20s} & {rca_str} & {comp_str} & {r['latency_mean']:.1f} & {params} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}\n")
    
    # --- Table 4: Correlation ---
    if correlation_results:
        lines.append("% Table 4: Retrieval-Reasoning Correlation")
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\caption{Correlation between retrieval and reasoning metrics.}")
        lines.append("\\label{tab:correlation}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("\\textbf{Metric} & \\textbf{Pearson $r$} & \\textbf{$p$} & \\textbf{Spearman $\\rho$} & \\textbf{$p$} \\\\")
        lines.append("\\midrule")
        
        label_map = {"precision_at_k": "Precision@K", "recall_at_k": "Recall@K", "mrr": "MRR"}
        for metric_name, label in label_map.items():
            if metric_name in correlation_results:
                c = correlation_results[metric_name]
                p_fmt = lambda v: f"{v:.4f}" if v >= 0.0001 else f"<0.001"
                lines.append(f"{label:12s} & {c['pearson_r']:.3f} & {p_fmt(c['pearson_p'])} & {c['spearman_rho']:.3f} & {p_fmt(c['spearman_p'])} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}\n")
    
    # --- Table 5: Failure Modes ---
    if failure_results:
        lines.append("% Table 5: Failure Mode Distribution")
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\caption{Per-query regime classification.}")
        lines.append("\\label{tab:failure-modes}")
        lines.append("\\begin{tabular}{lcc}")
        lines.append("\\toprule")
        lines.append("\\textbf{Regime} & \\textbf{Count} & \\textbf{\\%} \\\\")
        lines.append("\\midrule")
        
        for regime in ["High-Ret, High-Reas", "High-Ret, Low-Reas", "Low-Ret, High-Reas", "Low-Ret, Low-Reas"]:
            d = failure_results["distribution"].get(regime, {"count": 0, "percentage": 0})
            lines.append(f"{regime:25s} & {d['count']} & {d['percentage']:.1f} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}\n")
    
    # --- Table 6: Metric Inter-correlation ---
    if metric_summary is not None:
        lines.append("% Table 6: Improved Metrics Summary by Config")
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\caption{Evaluation metrics comparison across configurations.}")
        lines.append("\\label{tab:metrics}")
        lines.append("\\begin{tabular}{lccccc}")
        lines.append("\\toprule")
        lines.append("\\textbf{Config} & \\textbf{KW Acc} & \\textbf{Semantic} & \\textbf{ROUGE-L} & \\textbf{Struct.} & \\textbf{Composite} \\\\")
        lines.append("\\midrule")
        
        for cfg in config_order:
            row = metric_summary[metric_summary["config"] == cfg]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            lines.append(f"{cfg:20s} & {r['kw_mean']:.3f} & {r['sem_mean']:.3f} & {r['rouge_mean']:.3f} & {r['struct_mean']:.3f} & {r['comp_mean']:.3f} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}\n")
    
    # --- Table 7: Statistical Significance ---
    if significance_results:
        lines.append("% Table 7: Statistical Significance Tests")
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\caption{Paired t-tests between configurations. *$p<0.05$, **$p<0.01$, ***$p<0.001$.}")
        lines.append("\\label{tab:significance}")
        lines.append("\\begin{tabular}{llcccc}")
        lines.append("\\toprule")
        lines.append("\\textbf{Comparison} & \\textbf{Metric} & \\textbf{$\\Delta$} & \\textbf{$p$-value} & \\textbf{95\\% CI} & \\textbf{Sig.} \\\\")
        lines.append("\\midrule")
        
        for comp_key, metrics in significance_results.items():
            comp_label = comp_key.replace("_vs_", " vs ")
            for metric_name, vals in metrics.items():
                label = "P@K" if "precision" in metric_name else "RCA"
                sig = "***" if vals["p_value"] < 0.001 else "**" if vals["p_value"] < 0.01 else "*" if vals["p_value"] < 0.05 else "ns"
                ci_str = f"[{vals['ci_95_lower']:+.3f}, {vals['ci_95_upper']:+.3f}]"
                lines.append(f"{comp_label:30s} & {label} & {vals['mean_diff']:+.3f} & {vals['p_value']:.4f} & {ci_str} & {sig} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}\n")
    
    latex_code = "\n".join(lines)
    with open(out_dir / "latex_tables.tex", "w") as f:
        f.write(latex_code)
    
    print(f"  LaTeX tables saved to latex_tables.tex")
    print(f"\n{'='*60}")
    print(latex_code)
    print(f"{'='*60}")
    
    return latex_code


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--logs", default="data/synthetic_logs", help="Log directory")
    parser.add_argument("--queries", default="data/synthetic_eval_queries.json", help="Queries file")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit queries")
    parser.add_argument("--skip-multi-llm", action="store_true", help="Skip multi-LLM comparison")
    parser.add_argument("--rate-limit", type=float, default=0.4, help="Seconds between API calls")
    args = parser.parse_args()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY in .env")
        sys.exit(1)
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load queries
    with open(args.queries, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    queries = dataset["queries"]
    if args.max_queries:
        queries = queries[:args.max_queries]
    print(f"\nLoaded {len(queries)} queries")
    
    # Setup agent
    from rag_system.adaptive_agent import AdaptiveIterativeRAGAgent
    agent = AdaptiveIterativeRAGAgent(
        groq_api_key=api_key,
        max_iterations=3,
        top_k=6,
    )
    print(f"Loading logs from {args.logs}...")
    records = agent.load_logs(args.logs)
    print(f"Parsed {len(records)} records, indexed.\n")
    
    t0 = time.time()
    
    # ── 1. Ablation ──
    ablation_df, ablation_summary = run_ablation(agent, queries, out_dir, rate_limit=args.rate_limit)
    
    # ── 2. Multi-LLM ──
    multi_llm_df, multi_llm_summary = None, None
    if not args.skip_multi_llm:
        multi_llm_df, multi_llm_summary = run_multi_llm(agent, queries, out_dir, rate_limit=max(args.rate_limit, 0.5))
    
    # ── 3. Improved Metrics ──
    metrics_df, metric_summary = run_improved_metrics(ablation_df, queries, out_dir)
    
    # ── 4. Correlation ──
    correlation_results = run_correlation_analysis(ablation_df, out_dir)
    
    # ── 5. Failure Modes ──
    failure_results = run_failure_modes(ablation_df, out_dir)
    
    # ── 6. Significance ──
    significance_results = run_significance_tests(ablation_df, out_dir)
    
    # ── 7. LaTeX Tables ──
    latex_code = generate_latex_tables(
        ablation_summary, multi_llm_summary, correlation_results,
        failure_results, metric_summary, significance_results, out_dir
    )
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")
    print(f"  Files in {out_dir}/:")
    for f in sorted(out_dir.glob("*")):
        print(f"    {f.name} ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
