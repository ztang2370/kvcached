#!/usr/bin/env python3
"""
Analyze and compare benchmark results: with kvcached vs without kvcached.

Loads all JSON result files from sweep/ and sweep_without_kvcached/,
matches common configurations, aggregates across model replicas,
and produces:
  1. Summary CSV tables
  2. Comparison charts (latency, throughput, etc.)

Usage:
    python analyze_results.py [--results-dir RESULTS_DIR] [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Helpers ──────────────────────────────────────────────────────────────────

FILENAME_RE = re.compile(
    r"^(?P<pattern>poisson|uniform|ramp)"
    r"_rate(?P<rate>\d+)"
    r"_prompt(?P<prompt>\d+)"
    r"_gen(?P<gen>\d+)"
    r"_model(?P<model>\d+)\.json$"
)

METRICS = [
    # (json_key, display_name, unit, higher_is_better)
    ("mean_ttft_ms",   "Mean TTFT",   "ms", False),
    ("median_ttft_ms", "Median TTFT", "ms", False),
    ("p99_ttft_ms",    "P99 TTFT",    "ms", False),
    ("mean_tpot_ms",   "Mean TPOT",   "ms", False),
    ("median_tpot_ms", "Median TPOT", "ms", False),
    ("p99_tpot_ms",    "P99 TPOT",    "ms", False),
    ("mean_itl_ms",    "Mean ITL",    "ms", False),
    ("median_itl_ms",  "Median ITL",  "ms", False),
    ("p99_itl_ms",     "P99 ITL",     "ms", False),
    ("mean_e2el_ms",   "Mean E2EL",   "ms", False),
    ("median_e2el_ms", "Median E2EL", "ms", False),
    ("p99_e2el_ms",    "P99 E2EL",    "ms", False),
    ("request_throughput",     "Request Throughput",     "req/s", True),
    ("output_throughput",      "Output Throughput",      "tok/s", True),
    ("total_token_throughput", "Total Token Throughput",  "tok/s", True),
]

METRIC_KEYS = [m[0] for m in METRICS]


def parse_filename(fname: str) -> dict | None:
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    return {
        "pattern": m.group("pattern"),
        "rate": int(m.group("rate")),
        "prompt": int(m.group("prompt")),
        "gen": int(m.group("gen")),
        "model": int(m.group("model")),
    }


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all JSON result files from a directory into a DataFrame."""
    rows = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        info = parse_filename(fname)
        if info is None:
            continue
        fpath = os.path.join(results_dir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: skipping {fpath}: {e}", file=sys.stderr)
            continue

        row = {**info}
        for key in METRIC_KEYS:
            row[key] = data.get(key)
        row["completed"] = data.get("completed")
        row["num_prompts"] = data.get("num_prompts")
        row["duration"] = data.get("duration")
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def config_key(row):
    """Unique key for a (pattern, rate, prompt, gen) configuration."""
    return (row["pattern"], row["rate"], row["prompt"], row["gen"])


def aggregate_across_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (pattern, rate, prompt, gen), aggregate metrics across model replicas.
    Uses mean for latencies and sum for throughputs (since models run concurrently).
    """
    group_cols = ["pattern", "rate", "prompt", "gen"]

    # Throughput metrics should be summed (total across all instances)
    throughput_keys = ["request_throughput", "output_throughput", "total_token_throughput"]
    latency_keys = [k for k in METRIC_KEYS if k not in throughput_keys]

    agg_dict = {}
    for k in latency_keys:
        agg_dict[k] = "mean"  # average latency across replicas
    for k in throughput_keys:
        agg_dict[k] = "sum"   # sum throughput across replicas
    agg_dict["completed"] = "sum"
    agg_dict["model"] = "count"  # number of replicas

    agg = df.groupby(group_cols, as_index=False).agg(agg_dict)
    agg = agg.rename(columns={"model": "num_replicas"})
    return agg


def build_comparison_df(df_with: pd.DataFrame, df_without: pd.DataFrame,
                        merge_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Join the two DataFrames on merge_cols and compute deltas.
    """
    if merge_cols is None:
        merge_cols = ["pattern", "rate", "prompt", "gen"]
    comp = pd.merge(
        df_with, df_without,
        on=merge_cols,
        suffixes=("_with", "_without"),
        how="inner",
    )

    # Compute percentage change for each metric:
    # For latency (lower is better): negative % = kvcached is better
    # For throughput (higher is better): positive % = kvcached is better
    for key in METRIC_KEYS:
        w_col = f"{key}_with"
        wo_col = f"{key}_without"
        if w_col in comp.columns and wo_col in comp.columns:
            w = comp[w_col]
            wo = comp[wo_col]
            # pct_change = (with - without) / without * 100
            comp[f"{key}_pct"] = ((w - wo) / wo.replace(0, np.nan)) * 100

    return comp


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_metric_comparison_by_rate(
    comp: pd.DataFrame,
    metric_key: str,
    metric_name: str,
    unit: str,
    higher_is_better: bool,
    pattern: str,
    output_dir: str,
):
    """
    Bar chart: for a given pattern, group by rate, show with vs without for
    each (prompt, gen) combo, averaged across prompt/gen combos per rate.
    """
    sub = comp[comp["pattern"] == pattern].copy()
    if sub.empty:
        return

    rates = sorted(sub["rate"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: absolute values by rate (averaged over prompt/gen combos)
    ax = axes[0]
    rate_means_with = [sub[sub["rate"] == r][f"{metric_key}_with"].mean() for r in rates]
    rate_means_without = [sub[sub["rate"] == r][f"{metric_key}_without"].mean() for r in rates]

    x = np.arange(len(rates))
    width = 0.35
    bars1 = ax.bar(x - width/2, rate_means_with, width, label="With KVCached", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, rate_means_without, width, label="Without KVCached", color="#FF9800", alpha=0.85)
    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel(f"{metric_name} ({unit})")
    ax.set_title(f"{metric_name} by Rate ({pattern})")
    ax.set_xticks(x)
    ax.set_xticklabels(rates)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: percentage difference by rate
    ax = axes[1]
    pct_means = [sub[sub["rate"] == r][f"{metric_key}_pct"].mean() for r in rates]
    colors = []
    for pct in pct_means:
        if higher_is_better:
            colors.append("#4CAF50" if pct > 0 else "#F44336")
        else:
            colors.append("#4CAF50" if pct < 0 else "#F44336")

    ax.bar(x, pct_means, 0.6, color=colors, alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("% Change (with vs without)")
    benefit_label = "↓ = kvcached better" if not higher_is_better else "↑ = kvcached better"
    ax.set_title(f"{metric_name} % Change ({pattern}) [{benefit_label}]")
    ax.set_xticks(x)
    ax.set_xticklabels(rates)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fname = f"{pattern}_{metric_key}_by_rate.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()


def plot_heatmap(
    comp: pd.DataFrame,
    metric_key: str,
    metric_name: str,
    higher_is_better: bool,
    pattern: str,
    rate: int,
    output_dir: str,
):
    """
    Heatmap of % change for a given pattern+rate, with prompt on y-axis and gen on x-axis.
    Green = kvcached better, Red = kvcached worse.
    """
    sub = comp[(comp["pattern"] == pattern) & (comp["rate"] == rate)].copy()
    if sub.empty:
        return

    prompts = sorted(sub["prompt"].unique())
    gens = sorted(sub["gen"].unique())

    pct_col = f"{metric_key}_pct"
    matrix = np.full((len(prompts), len(gens)), np.nan)
    for _, row in sub.iterrows():
        pi = prompts.index(row["prompt"])
        gi = gens.index(row["gen"])
        matrix[pi, gi] = row[pct_col]

    fig, ax = plt.subplots(figsize=(max(8, len(gens) * 1.5), max(5, len(prompts) * 1.2)))

    # Color: for latency, negative is good (green); for throughput, positive is good (green)
    if higher_is_better:
        cmap = plt.cm.RdYlGn  # red=bad(negative), green=good(positive)
    else:
        cmap = plt.cm.RdYlGn_r  # red=bad(positive), green=good(negative)

    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 1)
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(gens)))
    ax.set_xticklabels(gens)
    ax.set_yticks(range(len(prompts)))
    ax.set_yticklabels(prompts)
    ax.set_xlabel("Generation Length")
    ax.set_ylabel("Prompt Length")

    # Annotate cells
    for i in range(len(prompts)):
        for j in range(len(gens)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if abs(val) > vmax * 0.6 else "black")

    benefit_label = "green = kvcached better" if True else ""
    ax.set_title(f"{metric_name} % Change | {pattern} rate={rate}\n({benefit_label})")
    plt.colorbar(im, ax=ax, label="% change (with vs without)")
    plt.tight_layout()
    fname = f"heatmap_{pattern}_rate{rate}_{metric_key}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()


def plot_overall_summary(comp: pd.DataFrame, output_dir: str):
    """
    Single chart showing average % change across all configs for each key metric.
    """
    key_metrics = [
        ("mean_ttft_ms",   "Mean TTFT",   False),
        ("p99_ttft_ms",    "P99 TTFT",    False),
        ("mean_tpot_ms",   "Mean TPOT",   False),
        ("p99_tpot_ms",    "P99 TPOT",    False),
        ("mean_itl_ms",    "Mean ITL",    False),
        ("p99_itl_ms",     "P99 ITL",     False),
        ("mean_e2el_ms",   "Mean E2EL",   False),
        ("p99_e2el_ms",    "P99 E2EL",    False),
        ("request_throughput",  "Req Throughput",  True),
        ("output_throughput",   "Out Throughput",  True),
    ]

    labels = []
    means = []
    medians = []
    colors_mean = []
    for key, name, higher_is_better in key_metrics:
        pct_col = f"{key}_pct"
        if pct_col not in comp.columns:
            continue
        m = comp[pct_col].mean()
        md = comp[pct_col].median()
        labels.append(name)
        means.append(m)
        medians.append(md)
        if higher_is_better:
            colors_mean.append("#4CAF50" if m > 0 else "#F44336")
        else:
            colors_mean.append("#4CAF50" if m < 0 else "#F44336")

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, means, width, label="Mean % Change", color=colors_mean, alpha=0.85)
    bars2 = ax.bar(x + width/2, medians, width, label="Median % Change", color="#9E9E9E", alpha=0.6)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("% Change (kvcached vs baseline)")
    ax.set_title("Overall % Change: KVCached vs Without KVCached\n(Latency: negative=better | Throughput: positive=better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars1, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:+.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_instance_summary(comp_by_model: dict[int, pd.DataFrame], output_dir: str):
    """
    Bar chart: for each model instance, show mean % change for key metrics.
    One cluster per instance, bars for each metric.
    """
    key_metrics = [
        ("mean_ttft_ms",   "TTFT",  False),
        ("mean_tpot_ms",   "TPOT",  False),
        ("mean_e2el_ms",   "E2EL",  False),
        ("output_throughput", "Out Tput", True),
    ]
    instances = sorted(comp_by_model.keys())

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(instances))
    n = len(key_metrics)
    width = 0.8 / n
    metric_colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]

    for i, (key, name, higher_is_better) in enumerate(key_metrics):
        pct_col = f"{key}_pct"
        vals = []
        for mi in instances:
            comp = comp_by_model[mi]
            vals.append(comp[pct_col].mean() if pct_col in comp.columns else 0)

        bars = ax.bar(x + i * width - (n-1)*width/2, vals, width,
                      label=name, color=metric_colors[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:+.1f}%", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Model Instance")
    ax.set_ylabel("Mean % Change (kvcached vs baseline)")
    ax.set_title("Per-Instance Comparison: KVCached vs Baseline\n"
                 "(Latency: negative=better | Throughput: positive=better)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Instance {mi}" for mi in instances])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_instance_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_instance_metric_by_rate(
    comp_by_model: dict[int, pd.DataFrame],
    metric_key: str,
    metric_name: str,
    unit: str,
    higher_is_better: bool,
    pattern: str,
    output_dir: str,
):
    """
    For a given pattern and metric, show one subplot per instance:
    with vs without bars grouped by rate.
    """
    instances = sorted(comp_by_model.keys())
    # Collect common rates across all instances for this pattern
    all_rates = set()
    for mi in instances:
        sub = comp_by_model[mi]
        sub_pat = sub[sub["pattern"] == pattern]
        all_rates.update(sub_pat["rate"].unique())
    if not all_rates:
        return
    rates = sorted(all_rates)

    fig, axes = plt.subplots(1, len(instances), figsize=(7 * len(instances), 6), sharey=True)
    if len(instances) == 1:
        axes = [axes]

    for idx, mi in enumerate(instances):
        ax = axes[idx]
        comp = comp_by_model[mi]
        sub = comp[comp["pattern"] == pattern]

        vals_with = []
        vals_without = []
        for r in rates:
            row = sub[sub["rate"] == r]
            vals_with.append(row[f"{metric_key}_with"].mean() if len(row) > 0 else 0)
            vals_without.append(row[f"{metric_key}_without"].mean() if len(row) > 0 else 0)

        x = np.arange(len(rates))
        width = 0.35
        ax.bar(x - width/2, vals_with, width, label="With KVCached", color="#2196F3", alpha=0.85)
        ax.bar(x + width/2, vals_without, width, label="Without KVCached", color="#FF9800", alpha=0.85)
        ax.set_xlabel("Request Rate (req/s)")
        if idx == 0:
            ax.set_ylabel(f"{metric_name} ({unit})")
        ax.set_title(f"Instance {mi}")
        ax.set_xticks(x)
        ax.set_xticklabels(rates)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{metric_name} by Rate — {pattern} (per instance)", fontsize=14, y=1.02)
    plt.tight_layout()
    fname = f"per_instance_{pattern}_{metric_key}_by_rate.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_instance_pct_by_rate(
    comp_by_model: dict[int, pd.DataFrame],
    metric_key: str,
    metric_name: str,
    higher_is_better: bool,
    pattern: str,
    output_dir: str,
):
    """
    Line chart: % change for each instance overlaid on the same plot, x-axis = rate.
    """
    instances = sorted(comp_by_model.keys())
    all_rates = set()
    for mi in instances:
        sub = comp_by_model[mi]
        sub_pat = sub[sub["pattern"] == pattern]
        all_rates.update(sub_pat["rate"].unique())
    if not all_rates:
        return
    rates = sorted(all_rates)
    pct_col = f"{metric_key}_pct"

    fig, ax = plt.subplots(figsize=(10, 6))
    instance_colors = ["#2196F3", "#FF9800", "#4CAF50"]
    markers = ["o", "s", "^"]

    for idx, mi in enumerate(instances):
        comp = comp_by_model[mi]
        sub = comp[comp["pattern"] == pattern]
        vals = []
        for r in rates:
            row = sub[sub["rate"] == r]
            vals.append(row[pct_col].mean() if len(row) > 0 and pct_col in row.columns else np.nan)
        ax.plot(rates, vals, marker=markers[idx], color=instance_colors[idx],
                linewidth=2, markersize=8, label=f"Instance {mi}", alpha=0.85)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("% Change (kvcached vs baseline)")
    benefit_label = "↓ = kvcached better" if not higher_is_better else "↑ = kvcached better"
    ax.set_title(f"{metric_name} % Change per Instance — {pattern}\n({benefit_label})")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fname = f"per_instance_{pattern}_{metric_key}_pct_by_rate.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()


def plot_intra_setup_instances(
    df_raw: pd.DataFrame,
    setup_label: str,
    output_dir: str,
):
    """
    Compare instance 1, 2, 3 within the same setup (with or without kvcached).
    Grouped bar chart: one cluster per metric, one bar per instance.
    """
    key_metrics = [
        ("mean_ttft_ms",   "Mean TTFT (ms)"),
        ("mean_tpot_ms",   "Mean TPOT (ms)"),
        ("mean_itl_ms",    "Mean ITL (ms)"),
        ("mean_e2el_ms",   "Mean E2EL (ms)"),
        ("output_throughput", "Out Tput (tok/s)"),
        ("request_throughput", "Req Tput (req/s)"),
    ]
    instances = sorted(df_raw["model"].unique())
    instance_colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(key_metrics))
    n = len(instances)
    width = 0.8 / n

    for idx, mi in enumerate(instances):
        sub = df_raw[df_raw["model"] == mi]
        vals = [sub[key].mean() for key, _ in key_metrics]
        bars = ax.bar(x + idx * width - (n-1)*width/2, vals, width,
                      label=f"Instance {mi}", color=instance_colors[idx], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_ylabel("Value")
    ax.set_title(f"Instance 1 vs 2 vs 3 — {setup_label}\n(absolute metric values, averaged across all configs)")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in key_metrics], rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    safe_label = setup_label.lower().replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"intra_{safe_label}_instances.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_intra_setup_instances_by_rate(
    df_raw: pd.DataFrame,
    metric_key: str,
    metric_name: str,
    unit: str,
    setup_label: str,
    output_dir: str,
):
    """
    Line chart: metric value per instance across rates (averaged over prompt/gen combos),
    one line per instance. Shows how each instance degrades under load.
    """
    instances = sorted(df_raw["model"].unique())
    instance_colors = ["#2196F3", "#FF9800", "#4CAF50"]
    markers = ["o", "s", "^"]

    patterns = sorted(df_raw["pattern"].unique())
    n_patterns = len(patterns)

    fig, axes = plt.subplots(1, n_patterns, figsize=(7 * n_patterns, 6), sharey=True)
    if n_patterns == 1:
        axes = [axes]

    for pi, pattern in enumerate(patterns):
        ax = axes[pi]
        sub_pat = df_raw[df_raw["pattern"] == pattern]
        rates = sorted(sub_pat["rate"].unique())

        for idx, mi in enumerate(instances):
            sub = sub_pat[sub_pat["model"] == mi]
            vals = [sub[sub["rate"] == r][metric_key].mean() for r in rates]
            ax.plot(rates, vals, marker=markers[idx], color=instance_colors[idx],
                    linewidth=2, markersize=8, label=f"Instance {mi}", alpha=0.85)

        ax.set_xlabel("Request Rate (req/s)")
        if pi == 0:
            ax.set_ylabel(f"{metric_name} ({unit})")
        ax.set_title(f"{pattern}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f"{metric_name} per Instance by Rate — {setup_label}", fontsize=14, y=1.02)
    plt.tight_layout()
    safe_label = setup_label.lower().replace(" ", "_")
    fname = f"intra_{safe_label}_{metric_key}_by_rate.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()


def plot_intra_setup_degradation(
    df_raw: pd.DataFrame,
    setup_label: str,
    output_dir: str,
):
    """
    Show % degradation of instance 2 and 3 relative to instance 1 (the baseline).
    Grouped bar: one cluster per metric, bars for inst2 vs inst1 and inst3 vs inst1.
    """
    key_metrics = [
        ("mean_ttft_ms",   "Mean TTFT",    False),
        ("mean_tpot_ms",   "Mean TPOT",    False),
        ("mean_e2el_ms",   "Mean E2EL",    False),
        ("output_throughput", "Out Tput",   True),
    ]
    instances = sorted(df_raw["model"].unique())
    if 1 not in instances:
        return

    # Get per-config values for each instance
    merge_cols = ["pattern", "rate", "prompt", "gen"]
    inst1 = df_raw[df_raw["model"] == 1][merge_cols + METRIC_KEYS].copy()
    inst1 = inst1.rename(columns={k: f"{k}_i1" for k in METRIC_KEYS})

    comparison_instances = [mi for mi in instances if mi != 1]
    inst_colors = {"2": "#FF9800", "3": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(key_metrics))
    n = len(comparison_instances)
    width = 0.8 / n

    for idx, mi in enumerate(comparison_instances):
        inst_mi = df_raw[df_raw["model"] == mi][merge_cols + METRIC_KEYS].copy()
        inst_mi = inst_mi.rename(columns={k: f"{k}_i{mi}" for k in METRIC_KEYS})
        merged = pd.merge(inst1, inst_mi, on=merge_cols, how="inner")

        pcts = []
        for key, name, hib in key_metrics:
            v1 = merged[f"{key}_i1"]
            vi = merged[f"{key}_i{mi}"]
            pct = ((vi - v1) / v1.replace(0, np.nan) * 100).mean()
            pcts.append(pct)

        bars = ax.bar(x + idx * width - (n-1)*width/2, pcts, width,
                      label=f"Instance {mi} vs Instance 1",
                      color=inst_colors.get(str(mi), "#9E9E9E"), alpha=0.85)
        for bar, val in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:+.1f}%", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=9, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("% Change vs Instance 1")
    ax.set_title(f"Instance Degradation Relative to Instance 1 — {setup_label}\n"
                 "(Latency: positive=worse | Throughput: negative=worse)")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name, _ in key_metrics])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    safe_label = setup_label.lower().replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"intra_{safe_label}_degradation.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_intra_setup_side_by_side(
    df_with_raw: pd.DataFrame,
    df_without_raw: pd.DataFrame,
    output_dir: str,
):
    """
    Side-by-side comparison: how instances degrade within each setup.
    Two subplots: left = with kvcached, right = without kvcached.
    Each shows instance 1, 2, 3 absolute values for key metrics.
    """
    key_metrics = [
        ("mean_ttft_ms",   "Mean TTFT"),
        ("mean_tpot_ms",   "Mean TPOT"),
        ("mean_e2el_ms",   "Mean E2EL"),
        ("output_throughput", "Out Tput"),
    ]

    instances = sorted(set(df_with_raw["model"].unique()) & set(df_without_raw["model"].unique()))
    instance_colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax_idx, (df, label) in enumerate([(df_with_raw, "With KVCached"), (df_without_raw, "Without KVCached")]):
        ax = axes[ax_idx]
        x = np.arange(len(key_metrics))
        n = len(instances)
        width = 0.8 / n

        for idx, mi in enumerate(instances):
            sub = df[df["model"] == mi]
            vals = [sub[key].mean() for key, _ in key_metrics]
            bars = ax.bar(x + idx * width - (n-1)*width/2, vals, width,
                          label=f"Instance {mi}", color=instance_colors[idx], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in key_metrics], rotation=15, ha="right")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Instance 1 vs 2 vs 3: Side-by-Side (With vs Without KVCached)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intra_side_by_side.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_intra_degradation_comparison(
    df_with_raw: pd.DataFrame,
    df_without_raw: pd.DataFrame,
    output_dir: str,
):
    """
    Grouped bar: % degradation of inst 2,3 vs inst 1, side by side for with/without kvcached.
    Shows whether the stagger penalty is worse under kvcached.
    """
    key_metrics = [
        ("mean_ttft_ms",   "Mean TTFT"),
        ("mean_tpot_ms",   "Mean TPOT"),
        ("mean_e2el_ms",   "Mean E2EL"),
        ("output_throughput", "Out Tput"),
    ]
    merge_cols = ["pattern", "rate", "prompt", "gen"]
    instances = sorted(set(df_with_raw["model"].unique()) & set(df_without_raw["model"].unique()))
    comparison_instances = [mi for mi in instances if mi != 1]

    bar_groups = []  # (label, pcts)
    colors = []
    color_map = {
        ("With KVCached", 2): "#1565C0",
        ("With KVCached", 3): "#64B5F6",
        ("Without KVCached", 2): "#E65100",
        ("Without KVCached", 3): "#FFB74D",
    }

    for label, df in [("With KVCached", df_with_raw), ("Without KVCached", df_without_raw)]:
        inst1 = df[df["model"] == 1][merge_cols + METRIC_KEYS].copy()
        inst1 = inst1.rename(columns={k: f"{k}_i1" for k in METRIC_KEYS})
        for mi in comparison_instances:
            inst_mi = df[df["model"] == mi][merge_cols + METRIC_KEYS].copy()
            inst_mi = inst_mi.rename(columns={k: f"{k}_i{mi}" for k in METRIC_KEYS})
            merged = pd.merge(inst1, inst_mi, on=merge_cols, how="inner")
            pcts = []
            for key, name in key_metrics:
                v1 = merged[f"{key}_i1"]
                vi = merged[f"{key}_i{mi}"]
                pct = ((vi - v1) / v1.replace(0, np.nan) * 100).mean()
                pcts.append(pct)
            bar_groups.append((f"{label}\nInst {mi} vs 1", pcts))
            colors.append(color_map.get((label, mi), "#9E9E9E"))

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(key_metrics))
    n = len(bar_groups)
    width = 0.8 / n

    for idx, ((lbl, pcts), color) in enumerate(zip(bar_groups, colors)):
        bars = ax.bar(x + idx * width - (n-1)*width/2, pcts, width,
                      label=lbl, color=color, alpha=0.85)
        for bar, val in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:+.1f}%", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=7, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("% Change vs Instance 1")
    ax.set_title("Stagger Penalty: Instance 2,3 vs Instance 1\n"
                 "With KVCached vs Without KVCached\n"
                 "(Latency: positive=worse | Throughput: negative=worse)")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in key_metrics])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intra_degradation_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_overall_summary_by_pattern(comp: pd.DataFrame, output_dir: str):
    """
    Grouped bar chart: one cluster per pattern, showing average % change for key metrics.
    """
    key_metrics = [
        ("mean_ttft_ms",   "TTFT",  False),
        ("mean_tpot_ms",   "TPOT",  False),
        ("mean_e2el_ms",   "E2EL",  False),
        ("output_throughput", "Out Tput", True),
    ]
    patterns = sorted(comp["pattern"].unique())

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(patterns))
    n = len(key_metrics)
    width = 0.8 / n

    for i, (key, name, higher_is_better) in enumerate(key_metrics):
        pct_col = f"{key}_pct"
        vals = []
        for pat in patterns:
            sub = comp[comp["pattern"] == pat]
            vals.append(sub[pct_col].mean())

        colors = []
        for v in vals:
            if higher_is_better:
                colors.append("#4CAF50" if v > 0 else "#F44336")
            else:
                colors.append("#4CAF50" if v < 0 else "#F44336")

        bars = ax.bar(x + i * width - (n-1)*width/2, vals, width, label=name, alpha=0.85)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Sending Pattern")
    ax.set_ylabel("Mean % Change")
    ax.set_title("KVCached vs Baseline: % Change by Pattern\n(Latency: negative=better | Throughput: positive=better)")
    ax.set_xticks(x)
    ax.set_xticklabels(patterns)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_by_pattern.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze kvcached benchmark results")
    parser.add_argument("--results-dir", type=str,
                        default="results",
                        help="Root results directory (contains sweep/ and sweep_without_kvcached/)")
    parser.add_argument("--output-dir", type=str,
                        default="results/analysis",
                        help="Directory to write analysis outputs")
    args = parser.parse_args()

    sweep_dir = os.path.join(args.results_dir, "sweep")
    nosweep_dir = os.path.join(args.results_dir, "sweep_without_kvcached")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from: {sweep_dir}")
    df_with_raw = load_results(sweep_dir)
    print(f"  Loaded {len(df_with_raw)} result files")

    print(f"Loading results from: {nosweep_dir}")
    df_without_raw = load_results(nosweep_dir)
    print(f"  Loaded {len(df_without_raw)} result files")

    # ── Aggregate across model replicas ──
    print("\nAggregating across model replicas...")
    df_with = aggregate_across_models(df_with_raw)
    df_without = aggregate_across_models(df_without_raw)
    print(f"  With kvcached:    {len(df_with)} configs")
    print(f"  Without kvcached: {len(df_without)} configs")

    # ── Build comparison ──
    comp = build_comparison_df(df_with, df_without)
    print(f"  Common configs for comparison: {len(comp)}")

    if comp.empty:
        print("ERROR: No common configurations found!")
        sys.exit(1)

    # ── Save CSV ──
    csv_path = os.path.join(output_dir, "comparison.csv")
    comp.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"\nSaved comparison CSV: {csv_path}")

    # ── Summary statistics ──
    print("\n" + "=" * 80)
    print("  OVERALL SUMMARY (averaged across all common configs)")
    print("=" * 80)
    print(f"{'Metric':<30} {'With KVCached':>15} {'Without KVCached':>18} {'% Change':>12} {'Winner':>10}")
    print("-" * 85)

    for key, name, unit, higher_is_better in METRICS:
        w = comp[f"{key}_with"].mean()
        wo = comp[f"{key}_without"].mean()
        pct = comp[f"{key}_pct"].mean()
        if higher_is_better:
            winner = "kvcached" if pct > 0 else "baseline"
        else:
            winner = "kvcached" if pct < 0 else "baseline"
        print(f"{name:<30} {w:>12.1f} {unit:<3} {wo:>12.1f} {unit:<3}   {pct:>+8.1f}%   {winner}")

    # ── Summary by pattern ──
    print("\n" + "=" * 80)
    print("  SUMMARY BY PATTERN")
    print("=" * 80)
    for pattern in sorted(comp["pattern"].unique()):
        sub = comp[comp["pattern"] == pattern]
        print(f"\n  --- {pattern.upper()} ({len(sub)} configs) ---")
        print(f"  {'Metric':<30} {'With KVCached':>15} {'Without':>15} {'% Change':>12}")
        print(f"  {'-'*75}")
        for key, name, unit, higher_is_better in METRICS:
            if "p99" in key or "std" in key:
                continue  # skip verbose ones in pattern summary
            w = sub[f"{key}_with"].mean()
            wo = sub[f"{key}_without"].mean()
            pct = sub[f"{key}_pct"].mean()
            marker = "✓" if (higher_is_better and pct > 0) or (not higher_is_better and pct < 0) else "✗"
            print(f"  {name:<30} {w:>11.1f} {unit:<4} {wo:>11.1f} {unit:<4}  {pct:>+8.1f}% {marker}")

    # ── Summary by rate ──
    print("\n" + "=" * 80)
    print("  SUMMARY BY REQUEST RATE")
    print("=" * 80)
    for rate in sorted(comp["rate"].unique()):
        sub = comp[comp["rate"] == rate]
        print(f"\n  --- Rate {rate} req/s ({len(sub)} configs) ---")
        print(f"  {'Metric':<25} {'With':>12} {'Without':>12} {'%Chg':>10}")
        print(f"  {'-'*62}")
        for key, name, unit, hib in [
            ("mean_ttft_ms", "Mean TTFT", "ms", False),
            ("mean_tpot_ms", "Mean TPOT", "ms", False),
            ("mean_e2el_ms", "Mean E2EL", "ms", False),
            ("output_throughput", "Out Tput", "tok/s", True),
        ]:
            w = sub[f"{key}_with"].mean()
            wo = sub[f"{key}_without"].mean()
            pct = sub[f"{key}_pct"].mean()
            print(f"  {name:<25} {w:>10.1f} {unit:<5} {wo:>10.1f} {unit:<5} {pct:>+8.1f}%")

    # ── Per-instance comparison ──
    print("\n" + "=" * 80)
    print("  PER-INSTANCE COMPARISON (model 1, 2, 3 separately)")
    print("=" * 80)

    # Build per-instance comparison DataFrames
    all_models = sorted(set(df_with_raw["model"].unique()) & set(df_without_raw["model"].unique()))
    comp_by_model: dict[int, pd.DataFrame] = {}
    for mi in all_models:
        dfw = df_with_raw[df_with_raw["model"] == mi].copy()
        dfwo = df_without_raw[df_without_raw["model"] == mi].copy()
        comp_mi = build_comparison_df(dfw, dfwo, merge_cols=["pattern", "rate", "prompt", "gen"])
        comp_by_model[mi] = comp_mi
        print(f"\n  === Instance {mi} ({len(comp_mi)} common configs) ===")
        print(f"  {'Metric':<30} {'With KVCached':>15} {'Without':>15} {'% Change':>12}")
        print(f"  {'-'*75}")
        for key, name, unit, higher_is_better in METRICS:
            if "p99" in key or "std" in key:
                continue
            w_col = f"{key}_with"
            wo_col = f"{key}_without"
            pct_col = f"{key}_pct"
            if w_col not in comp_mi.columns:
                continue
            w = comp_mi[w_col].mean()
            wo = comp_mi[wo_col].mean()
            pct = comp_mi[pct_col].mean()
            marker = "✓" if (higher_is_better and pct > 0) or (not higher_is_better and pct < 0) else "✗"
            print(f"  {name:<30} {w:>11.1f} {unit:<4} {wo:>11.1f} {unit:<4}  {pct:>+8.1f}% {marker}")

    # Per-instance by pattern
    print("\n" + "=" * 80)
    print("  PER-INSTANCE × PATTERN BREAKDOWN")
    print("=" * 80)
    for mi in all_models:
        comp_mi = comp_by_model[mi]
        for pattern in sorted(comp_mi["pattern"].unique()):
            sub = comp_mi[comp_mi["pattern"] == pattern]
            print(f"\n  Instance {mi} / {pattern.upper()} ({len(sub)} configs)")
            print(f"  {'Metric':<25} {'With':>12} {'Without':>12} {'%Chg':>10}")
            print(f"  {'-'*62}")
            for key, name, unit, hib in [
                ("mean_ttft_ms", "Mean TTFT", "ms", False),
                ("mean_tpot_ms", "Mean TPOT", "ms", False),
                ("mean_e2el_ms", "Mean E2EL", "ms", False),
                ("output_throughput", "Out Tput", "tok/s", True),
            ]:
                w_col = f"{key}_with"
                wo_col = f"{key}_without"
                pct_col = f"{key}_pct"
                if w_col not in sub.columns:
                    continue
                w = sub[w_col].mean()
                wo = sub[wo_col].mean()
                pct = sub[pct_col].mean()
                print(f"  {name:<25} {w:>10.1f} {unit:<5} {wo:>10.1f} {unit:<5} {pct:>+8.1f}%")

    # Save per-instance CSVs
    for mi in all_models:
        csv_path_mi = os.path.join(output_dir, f"comparison_instance{mi}.csv")
        comp_by_model[mi].to_csv(csv_path_mi, index=False, float_format="%.2f")
    print("\n  Saved per-instance CSVs: comparison_instance{1,2,3}.csv")

    # ── Intra-setup comparison (instances within the same setup) ──
    for label, df_raw in [("With KVCached", df_with_raw), ("Without KVCached", df_without_raw)]:
        print("\n" + "=" * 80)
        print(f"  INTRA-SETUP: INSTANCES 1 vs 2 vs 3 — {label.upper()}")
        print("=" * 80)
        inst_models = sorted(df_raw["model"].unique())
        merge_cols = ["pattern", "rate", "prompt", "gen"]

        # Overall absolute values per instance
        print(f"\n  {'Metric':<25}", end="")
        for mi in inst_models:
            print(f" {'Inst '+str(mi):>14}", end="")
        print()
        print(f"  {'-'*(25 + 15*len(inst_models))}")
        for key, name, unit, _ in METRICS:
            if "p99" in key or "std" in key:
                continue
            print(f"  {name:<25}", end="")
            for mi in inst_models:
                sub = df_raw[df_raw["model"] == mi]
                v = sub[key].mean()
                print(f" {v:>11.1f} {unit:<2}", end="")
            print()

        # % degradation relative to instance 1
        if 1 in inst_models:
            inst1 = df_raw[df_raw["model"] == 1][merge_cols + METRIC_KEYS].copy()
            inst1_renamed = inst1.rename(columns={k: f"{k}_i1" for k in METRIC_KEYS})
            print("\n  % Change relative to Instance 1:")
            print(f"  {'Metric':<25}", end="")
            for mi in inst_models:
                print(f" {'Inst '+str(mi):>14}", end="")
            print()
            print(f"  {'-'*(25 + 15*len(inst_models))}")
            for key, name, unit, hib in [
                ("mean_ttft_ms",   "Mean TTFT",   "ms",    False),
                ("mean_tpot_ms",   "Mean TPOT",   "ms",    False),
                ("mean_itl_ms",    "Mean ITL",    "ms",    False),
                ("mean_e2el_ms",   "Mean E2EL",   "ms",    False),
                ("output_throughput", "Out Tput", "tok/s",  True),
                ("request_throughput", "Req Tput", "req/s", True),
            ]:
                print(f"  {name:<25}", end="")
                for mi in inst_models:
                    if mi == 1:
                        print(f" {'(baseline)':>14}", end="")
                    else:
                        inst_mi = df_raw[df_raw["model"] == mi][merge_cols + METRIC_KEYS].copy()
                        inst_mi_renamed = inst_mi.rename(columns={k: f"{k}_i{mi}" for k in METRIC_KEYS})
                        merged = pd.merge(inst1_renamed, inst_mi_renamed, on=merge_cols, how="inner")
                        v1 = merged[f"{key}_i1"]
                        vi = merged[f"{key}_i{mi}"]
                        pct = ((vi - v1) / v1.replace(0, np.nan) * 100).mean()
                        print(f" {pct:>+11.1f}%  ", end="")
                print()

    # ── Generate plots ──
    print("\n\nGenerating plots...")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Overall summary
    plot_overall_summary(comp, plots_dir)
    print("  ✓ overall_summary.png")

    # 2. Summary by pattern
    plot_overall_summary_by_pattern(comp, plots_dir)
    print("  ✓ summary_by_pattern.png")

    # 3. Per-pattern metric comparisons by rate
    key_plot_metrics = [
        ("mean_ttft_ms",   "Mean TTFT",   "ms",    False),
        ("mean_tpot_ms",   "Mean TPOT",   "ms",    False),
        ("mean_e2el_ms",   "Mean E2EL",   "ms",    False),
        ("mean_itl_ms",    "Mean ITL",    "ms",    False),
        ("output_throughput", "Output Throughput", "tok/s", True),
        ("request_throughput", "Request Throughput", "req/s", True),
    ]
    for pattern in sorted(comp["pattern"].unique()):
        for key, name, unit, hib in key_plot_metrics:
            plot_metric_comparison_by_rate(comp, key, name, unit, hib, pattern, plots_dir)
        print(f"  ✓ {pattern}_*.png (bar charts)")

    # 4. Heatmaps for key metrics at each rate
    for pattern in sorted(comp["pattern"].unique()):
        for rate in sorted(comp[comp["pattern"] == pattern]["rate"].unique()):
            for key, name, hib in [
                ("mean_ttft_ms", "Mean TTFT", False),
                ("mean_e2el_ms", "Mean E2EL", False),
                ("output_throughput", "Output Throughput", True),
            ]:
                plot_heatmap(comp, key, name, hib, pattern, rate, plots_dir)
        print(f"  ✓ heatmap_{pattern}_*.png")

    # 5. Per-instance summary chart
    plot_per_instance_summary(comp_by_model, plots_dir)
    print("  ✓ per_instance_summary.png")

    # 6. Per-instance metric comparisons by rate
    per_inst_metrics = [
        ("mean_ttft_ms",   "Mean TTFT",   "ms",    False),
        ("mean_tpot_ms",   "Mean TPOT",   "ms",    False),
        ("mean_e2el_ms",   "Mean E2EL",   "ms",    False),
        ("output_throughput", "Output Throughput", "tok/s", True),
    ]
    for pattern in sorted(comp["pattern"].unique()):
        for key, name, unit, hib in per_inst_metrics:
            plot_per_instance_metric_by_rate(comp_by_model, key, name, unit, hib, pattern, plots_dir)
            plot_per_instance_pct_by_rate(comp_by_model, key, name, hib, pattern, plots_dir)
        print(f"  ✓ per_instance_{pattern}_*.png")

    # 7. Intra-setup instance comparison plots
    for label, df_raw in [("With KVCached", df_with_raw), ("Without KVCached", df_without_raw)]:
        plot_intra_setup_instances(df_raw, label, plots_dir)
        plot_intra_setup_degradation(df_raw, label, plots_dir)
        for key, name, unit in [
            ("mean_ttft_ms",   "Mean TTFT",   "ms"),
            ("mean_tpot_ms",   "Mean TPOT",   "ms"),
            ("mean_e2el_ms",   "Mean E2EL",   "ms"),
            ("output_throughput", "Output Throughput", "tok/s"),
        ]:
            plot_intra_setup_instances_by_rate(df_raw, key, name, unit, label, plots_dir)
        safe = label.lower().replace(" ", "_")
        print(f"  ✓ intra_{safe}_*.png")

    # 8. Side-by-side and degradation comparison
    plot_intra_setup_side_by_side(df_with_raw, df_without_raw, plots_dir)
    print("  ✓ intra_side_by_side.png")

    plot_intra_degradation_comparison(df_with_raw, df_without_raw, plots_dir)
    print("  ✓ intra_degradation_comparison.png")

    print(f"\nAll outputs saved to: {output_dir}/")
    print("  - comparison.csv")
    print("  - comparison_instance{1,2,3}.csv")
    print(f"  - plots/ ({len(os.listdir(plots_dir))} charts)")
    print("\nDone!")


if __name__ == "__main__":
    main()
