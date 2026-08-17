"""Aggregate every eval_results_*.json into a comparison table + recall@k chart."""

import sys
from pathlib import Path

# repo root on sys.path so `import src...` works both when this file is run
# directly and when it is imported (e.g. from a notebook)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import matplotlib.pyplot as plt

from src.utils.io import PROCESSED_DIR, load_json, resolve

RETRIEVAL_RESULTS = {
    "Row-level (baseline)": "data/processed/eval_results_baseline_row_level.json",
    "Naive fixed-size": "data/processed/eval_results_naive_fixed.json",
    "Whole-table": "data/processed/eval_results_whole_table.json",
    "Sentence-window": "data/processed/eval_results_sentence_window.json",
}
GENERATION_RESULTS = "data/processed/generation_comparison.json"
GENERATION_DETAIL = "data/processed/eval_results_generation.json"
CHART_PATH = "data/processed/chunking_comparison.png"

# generation results are keyed by module-style names; map them onto the display labels
GEN_KEY_BY_LABEL = {
    "Row-level (baseline)": "row_level",
    "Naive fixed-size": "naive_fixed",
    "Whole-table": "whole_table",
    "Sentence-window": "sentence_window",
}


def load_retrieval_results() -> dict[str, dict[int, dict]]:
    results = {}
    for label, path in RETRIEVAL_RESULTS.items():
        if not resolve(path).exists():
            print(f"  skipping {label}: {path} not found")
            continue
        results[label] = {int(k): v for k, v in load_json(path).items()}
    return results


def load_generation_accuracy() -> dict[str, float]:
    """Accepts either the slim {strategy: accuracy} file or the full per-question
    file, and either key style, since runs from earlier sessions used the display
    labels as keys."""
    for path in (GENERATION_RESULTS, GENERATION_DETAIL):
        if not resolve(path).exists():
            continue
        raw = load_json(path)
        accuracies = {k: (v["accuracy"] if isinstance(v, dict) else v) for k, v in raw.items()}

        out = {}
        for label, key in GEN_KEY_BY_LABEL.items():
            for candidate in (key, label):
                if candidate in accuracies:
                    out[label] = accuracies[candidate]
                    break
        return out
    return {}


def print_table(retrieval: dict, generation: dict) -> None:
    print(f"{'Strategy':<25} {'P@5':>8} {'R@5':>8} {'P@10':>8} {'R@10':>8} {'gen acc':>9}")
    for label, res in retrieval.items():
        gen = generation.get(label)
        gen_str = f"{gen:>8.1%}" if gen is not None else f"{'n/a':>8}"
        print(f"{label:<25} {res[5]['precision']:>8.3f} {res[5]['recall']:>8.3f} "
              f"{res[10]['precision']:>8.3f} {res[10]['recall']:>8.3f} {gen_str}")


def plot_recall_curves(retrieval: dict, out_path=CHART_PATH, show: bool = False):
    fig, ax = plt.subplots(figsize=(8, 5))
    k_values = sorted(next(iter(retrieval.values())).keys())
    for label, res in retrieval.items():
        ax.plot(k_values, [res[k]["recall"] for k in k_values], marker="o", label=label)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Chunking Strategy Comparison \u2014 Recall@k on FinQA dev set")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    target = PROCESSED_DIR / "chunking_comparison.png"
    fig.savefig(target, dpi=120)
    if show:
        plt.show()
    return fig, target


def build_report(show: bool = False):
    retrieval = load_retrieval_results()
    if not retrieval:
        raise SystemExit("no eval_results_*.json found -- run run_retrieval_eval.py first")
    generation = load_generation_accuracy()

    print_table(retrieval, generation)
    _, target = plot_recall_curves(retrieval, show=show)
    print(f"\nchart written to {target}")
    return retrieval, generation


def main() -> None:
    matplotlib.use("Agg")   # headless when run as a script; notebooks keep their inline backend
    build_report(show=False)


if __name__ == "__main__":
    main()
