"""Strategy 5 (planned, not yet built): small-to-big / parent-child retrieval.

Motivation from the strategy 1 vs 3 results: row-level chunks retrieve more
precisely because a single row is a tight match for a question, but whole-table
chunks answer better because the arithmetic usually needs a second cell from the
same table. Parent-child is meant to get both -- embed and match on the row
(child), then hand the LLM the whole table it came from (parent).

Implementation sketch:
  1. Index the row-level chunks from row_level.build_row_level_chunks unchanged,
     so retrieval precision and gold-id alignment are preserved as-is.
  2. Keep a child chunk_id -> parent chunk_id map, where the parent for a
     table_row is its doc's whole_table chunk and the parent for a text_line is
     its sentence window.
  3. At query time, retrieve top-k children, map to parents, dedupe parents
     (several winning rows often share one table), then build the prompt from
     the parents while still scoring retrieval against the child ids.
"""


def build_parent_child_chunks(doc: dict) -> list[dict]:
    raise NotImplementedError(
        "Strategy 5 is not implemented yet; see the module docstring for the intended design."
    )
