#!/usr/bin/env python3
"""
update_paper_tables.py — Fill paper.tex tables with actual experiment results.

Reads results from:
    - results/ablation_summary.csv
    - results/multi_llm_summary.csv  
    - results/correlation_analysis.json
    - results/failure_modes.json
    - results/significance_tests.json
    - results/improved_metrics_summary.csv
    - results/metric_correlations.json

Updates paper.tex with actual values, then recompiles to PDF.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def load_results(results_dir):
    """Load all result files."""
    r = {}
    rd = Path(results_dir)
    
    # Ablation
    f = rd / "ablation_summary.csv"
    if f.exists():
        r["ablation"] = pd.read_csv(f)
        print(f"  Loaded ablation_summary.csv ({len(r['ablation'])} configs)")
    
    # Multi-LLM
    f = rd / "multi_llm_summary.csv"
    if f.exists():
        r["multi_llm"] = pd.read_csv(f)
        print(f"  Loaded multi_llm_summary.csv ({len(r['multi_llm'])} models)")
    
    # Correlation
    f = rd / "correlation_analysis.json"
    if f.exists():
        with open(f) as fh:
            r["correlation"] = json.load(fh)
        print(f"  Loaded correlation_analysis.json")
    
    # Failure modes
    f = rd / "failure_modes.json"
    if f.exists():
        with open(f) as fh:
            r["failure"] = json.load(fh)
        print(f"  Loaded failure_modes.json")
    
    # Significance
    f = rd / "significance_tests.json"
    if f.exists():
        with open(f) as fh:
            r["significance"] = json.load(fh)
        print(f"  Loaded significance_tests.json")
    
    # Improved metrics
    f = rd / "improved_metrics_summary.csv"
    if f.exists():
        r["metrics"] = pd.read_csv(f)
        print(f"  Loaded improved_metrics_summary.csv")
    
    # Metric correlations
    f = rd / "metric_correlations.json"
    if f.exists():
        with open(f) as fh:
            r["metric_corr"] = json.load(fh)
        print(f"  Loaded metric_correlations.json")
    
    return r


def fmt_val(mean, std=None):
    """Format as mean±std."""
    if std is not None and not np.isnan(std) and std > 0.001:
        return f"{mean:.3f}$\\pm${std:.2f}"
    return f"{mean:.3f}"


def build_ablation_table(df):
    """Generate LaTeX ablation table body."""
    config_order = ["Dense-Only", "BM25-Only", "Hybrid", "Hybrid+Rerank", "Hybrid+Iter", "Full-System"]
    lines = []
    for cfg in config_order:
        row = df[df["config"] == cfg]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        pk = fmt_val(r["pk_mean"], r.get("pk_std"))
        rk = fmt_val(r["rk_mean"], r.get("rk_std"))
        mrr = fmt_val(r["mrr_mean"], r.get("mrr_std"))
        rca = fmt_val(r["rca_mean"], r.get("rca_std"))
        conf = f"{r['conf_mean']:.3f}"
        iters = f"{r['iter_mean']:.1f}"
        # Short display names
        disp = cfg.replace("Hybrid+Iter", "Hybrid+Iter.")
        lines.append(f"{disp:20s} & {pk} & {rk} & {mrr} & {rca} & {conf} & {iters} \\\\")
    return "\n".join(lines)


def build_multi_llm_table(df):
    """Generate LaTeX multi-LLM table body."""
    model_order = ["Llama-3.3-70B", "Mixtral-8x7B", "Gemma2-9B", "Llama-3.1-8B"]
    param_map = {"Llama-3.3-70B": "70B", "Mixtral-8x7B": "46.7B", "Gemma2-9B": "9B", "Llama-3.1-8B": "8B"}
    lines = []
    for model in model_order:
        row = df[df["model"] == model]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        rca = fmt_val(r["rca_mean"], r.get("rca_std"))
        comp = fmt_val(r["comp_mean"], r.get("comp_std"))
        lat = f"{r['latency_mean']:.1f}"
        params = param_map.get(model, "")
        lines.append(f"{model:20s} & {rca} & {comp} & {lat} & {params} \\\\")
    return "\n".join(lines)


def build_correlation_table(corr):
    """Generate LaTeX correlation table body."""
    label_map = {"precision_at_k": "Precision@K", "recall_at_k": "Recall@K", "mrr": "MRR"}
    lines = []
    for key, label in label_map.items():
        if key in corr:
            c = corr[key]
            pp = f"{c['pearson_p']:.4f}" if c['pearson_p'] >= 0.0001 else "$<$0.001"
            sp = f"{c['spearman_p']:.4f}" if c['spearman_p'] >= 0.0001 else "$<$0.001"
            lines.append(f"{label:12s} & {c['pearson_r']:.3f} & {pp} & {c['spearman_rho']:.3f} & {sp} \\\\")
    return "\n".join(lines)


def build_failure_table(failure):
    """Generate LaTeX failure mode table body."""
    lines = []
    for regime in ["High-Ret, High-Reas", "High-Ret, Low-Reas", "Low-Ret, High-Reas", "Low-Ret, Low-Reas"]:
        d = failure["distribution"].get(regime, {"count": 0, "percentage": 0})
        # LaTeX-friendly name
        disp = regime.replace("High-Ret", "High retrieval").replace("Low-Ret", "Low retrieval").replace("High-Reas", "high reasoning").replace("Low-Reas", "low reasoning")
        lines.append(f"{disp:40s} & {d['count']} & {d['percentage']:.1f} \\\\")
    return "\n".join(lines)


def build_metric_corr_table(metric_corr):
    """Generate LaTeX metric inter-correlation table body."""
    pairs = [
        ("keyword_accuracy_vs_semantic_similarity", "Keyword overlap vs.\\ Semantic similarity"),
        ("keyword_accuracy_vs_composite_score", "Keyword overlap vs.\\ Composite score"),
        ("semantic_similarity_vs_rouge_l", "Semantic sim.\\ vs.\\ ROUGE-L F1"),
    ]
    lines = []
    for key, label in pairs:
        if key in metric_corr:
            c = metric_corr[key]
            lines.append(f"{label:45s} & {c['pearson_r']:.3f} \\\\")
    return "\n".join(lines)


def update_paper(paper_path, results):
    """Replace placeholder tables in paper.tex with actual values."""
    with open(paper_path, "r", encoding="utf-8") as f:
        tex = f.read()
    
    original = tex
    
    # --- Ablation Table ---
    if "ablation" in results:
        body = build_ablation_table(results["ablation"])
        # Replace the table body between \midrule and \bottomrule in tab:ablation
        pattern = r'(\\label\{tab:ablation\}.*?\\midrule\n)(.*?)(\\bottomrule)'
        def repl_ablation(m):
            return m.group(1) + body + "\n" + m.group(3)
        tex = re.sub(pattern, repl_ablation, tex, flags=re.DOTALL)
        print("  Updated: Ablation table")
    
    # --- Multi-LLM Table ---
    if "multi_llm" in results:
        body = build_multi_llm_table(results["multi_llm"])
        pattern = r'(\\label\{tab:multi-llm\}.*?\\midrule\n)(.*?)(\\bottomrule)'
        def repl_llm(m):
            return m.group(1) + body + "\n" + m.group(3)
        tex = re.sub(pattern, repl_llm, tex, flags=re.DOTALL)
        print("  Updated: Multi-LLM table")
    
    # --- Correlation Table ---
    if "correlation" in results:
        body = build_correlation_table(results["correlation"])
        pattern = r'(\\label\{tab:correlation\}.*?\\midrule\n)(.*?)(\\bottomrule)'
        def repl_corr(m):
            return m.group(1) + body + "\n" + m.group(3)
        tex = re.sub(pattern, repl_corr, tex, flags=re.DOTALL)
        print("  Updated: Correlation table")
    
    # --- Failure Mode Table ---
    if "failure" in results:
        body = build_failure_table(results["failure"])
        pattern = r'(\\label\{tab:failure-modes\}.*?\\midrule\n)(.*?)(\\bottomrule)'
        def repl_fail(m):
            return m.group(1) + body + "\n" + m.group(3)
        tex = re.sub(pattern, repl_fail, tex, flags=re.DOTALL)
        print("  Updated: Failure mode table")
    
    # --- Metric Correlation Table ---
    if "metric_corr" in results:
        body = build_metric_corr_table(results["metric_corr"])
        pattern = r'(\\label\{tab:metric-corr\}.*?\\midrule\n)(.*?)(\\bottomrule)'
        def repl_metric(m):
            return m.group(1) + body + "\n" + m.group(3)
        tex = re.sub(pattern, repl_metric, tex, flags=re.DOTALL)
        print("  Updated: Metric correlation table")
    
    # --- Update abstract with actual numbers ---
    if "ablation" in results:
        df = results["ablation"]
        pk_range = df["pk_mean"].max() - df["pk_mean"].min()
        rca_range = df["rca_mean"].max() - df["rca_mean"].min()
        
        # Update "Precision@K varies by up to X" in abstract
        tex = re.sub(
            r'Precision@K varies by up to \d+\.\d+',
            f'Precision@K varies by up to {pk_range:.2f}',
            tex
        )
        # Update RCA flat range - use str.replace to avoid regex issues with \sim
        rca_min, rca_max = df["rca_mean"].min(), df["rca_mean"].max()
        old_pattern = re.search(
            r'root-cause accuracy remains flat \(\$\\sim\$[\d\.]+--[\d\.]+\)',
            tex
        )
        if old_pattern:
            tex = tex[:old_pattern.start()] + \
                f'root-cause accuracy remains flat ($\\sim${rca_min:.2f}--{rca_max:.2f})' + \
                tex[old_pattern.end():]
    
    # --- Update findings paragraph ---
    if "ablation" in results:
        df = results["ablation"]
        pk_range = df["pk_mean"].max() - df["pk_mean"].min()
        rca_range = df["rca_mean"].max() - df["rca_mean"].min()
        
        old_pattern = re.search(
            r'RCA\\textsubscript\{kw\} varies by \$< [\d\.]+\$ while P@K varies\nby up to [\d\.]+',
            tex
        )
        if old_pattern:
            tex = tex[:old_pattern.start()] + \
                f'RCA\\textsubscript{{kw}} varies by $< {rca_range:.2f}$ while P@K varies\nby up to {pk_range:.2f}' + \
                tex[old_pattern.end():]
    
    if tex != original:
        with open(paper_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"\n  Paper updated: {paper_path}")
    else:
        print("\n  No changes made to paper.")
    
    return tex


def main():
    results_dir = "results"
    paper_path = "paper.tex"
    
    print("Loading experiment results...")
    results = load_results(results_dir)
    
    if not results:
        print("No results found. Run experiments first.")
        sys.exit(1)
    
    print(f"\nUpdating {paper_path}...")
    update_paper(paper_path, results)
    
    print("\nDone. Run 'pdflatex paper.tex' to recompile.")


if __name__ == "__main__":
    main()
