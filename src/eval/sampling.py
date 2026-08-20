"""Question sampling for the (slow, local-LLM) generation eval.

A uniform sample would be dominated by divide/subtract (~75% of the dev set)
and would barely touch table_min/table_max. Stratifying by operation type keeps
a 44-question sample informative.
"""

import random
import re
from collections import defaultdict


def primary_op(program: str) -> str:
    """First operation in a FinQA program, e.g. 'divide(46.6, 54.9)' -> 'divide'."""
    m = re.match(r"([a-z_]+)\(", program or "")
    return m.group(1) if m else "unknown"


def group_by_op(eval_examples: list[dict]) -> dict[str, list[dict]]:
    by_op = defaultdict(list)
    for ex in eval_examples:
        by_op[primary_op(ex.get("program", ""))].append(ex)
    return dict(by_op)


def stratified_sample_by_op(eval_examples: list[dict], per_op_n: int = 5, seed: int = 42) -> list[dict]:
    """Up to per_op_n questions per operation type. Seeded so reruns stay comparable."""
    by_op = group_by_op(eval_examples)
    random.seed(seed)

    sample = []
    for exs in by_op.values():
        sample.extend(random.sample(exs, min(per_op_n, len(exs))))
    return sample


def describe_sample(eval_examples: list[dict], per_op_n: int = 5) -> list[tuple[str, int, int]]:
    """[(op, n_sampled, n_available), ...] for printing sample composition."""
    by_op = group_by_op(eval_examples)
    return [(op, min(per_op_n, len(exs)), len(exs)) for op, exs in by_op.items()]
