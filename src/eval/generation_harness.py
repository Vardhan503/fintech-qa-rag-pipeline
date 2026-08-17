"""End-to-end eval: retrieve, ask the LLM, grade the answer.

This is the metric that actually settled which chunking strategy is better.
Retrieval precision/recall only measure whether the evidence was fetched;
accuracy here measures whether the model could then compute the answer from it.

Ground truth is exe_ans, the executed result of FinQA's program, not the
`answer` string field -- `answer` is human-typed and noisy (rounded to '1%' when
exe_ans is 0.015, occasionally unrelated to exe_ans entirely).
"""

import re

import requests

from src.eval.retrieval_harness import cosine_top_k, embed_texts

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:7b-instruct"   # swap for llama3.1:8b if you prefer

PROMPT_TEMPLATE = """You are a financial analyst. Use ONLY the context below to answer the question.
Show your arithmetic briefly, then finish with a line of exactly the form:
ANSWER: <single number, or yes/no>
Give percentages as a percent value (e.g. 14.3%).

Context:
{context}

Question: {question}
"""


def parse_number(s):
    """Pull the last number out of an LLM's text answer -> float, or None."""
    if isinstance(s, (int, float)):
        return float(s)
    if not isinstance(s, str):
        return None

    nums = re.findall(r"-?\d*\.?\d+\s*%?", s.replace(",", "").replace("$", ""))
    if not nums:
        return None
    tok = nums[-1].strip()
    val = float(tok.rstrip("% "))
    return val / 100.0 if tok.endswith("%") else val


def answers_match(gold, pred, tol: float = 0.02) -> bool:
    """gold is exe_ans: a float for numeric questions, "yes"/"no" for boolean ones.
    pred is the raw LLM string, so it must be parsed before any arithmetic --
    subtracting a str from a float is exactly how this used to blow up."""
    if isinstance(gold, str):
        g = gold.strip().lower()
        if g in ("yes", "no"):
            # only the opening words, so a hedged "... not yes" can't score a hit
            return g in re.findall(r"[a-z]+", (pred or "").lower())[:6]
        return g == (pred or "").strip().lower()

    if gold is None:
        return False

    g = float(gold)
    p = parse_number(pred)
    if p is None:
        return False

    # FinQA is inconsistent about percent scaling (0.85 vs 85) and sign, so a
    # bare tolerance check on the literal value under-reports accuracy badly
    for cand in (p, p / 100.0, p * 100.0):
        for gg in (g, abs(g)):
            if abs(gg - abs(cand)) <= tol * max(abs(gg), 1e-6) or abs(gg - cand) <= 1e-4:
                return True
    return False


def call_llm(prompt: str, llm_model: str = LLM_MODEL, timeout: int = 180) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": llm_model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.0, "num_predict": 300}},   # deterministic for fair comparison
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
    return PROMPT_TEMPLATE.format(context=context, question=question)


def extract_final_answer(raw: str) -> str:
    """Prefer the explicit ANSWER: line; fall back to the whole response."""
    hits = re.findall(r"ANSWER:\s*(.+)", raw or "", flags=re.IGNORECASE)
    return hits[-1].strip() if hits else (raw or "").strip()


def run_generation_eval(strategy_chunks: list[dict], sample: list[dict], model,
                         top_k: int = 5, llm_model: str = LLM_MODEL, verbose: bool = True) -> dict:
    """Retrieve top_k for each sampled question, generate, grade against exe_ans."""
    chunk_ids = [c["chunk_id"] for c in strategy_chunks]
    chunk_texts = [c["text"] for c in strategy_chunks]
    id_to_text = dict(zip(chunk_ids, chunk_texts))

    chunk_vecs = embed_texts(chunk_texts, model)
    query_vecs = embed_texts([e["question"] for e in sample], model)

    results = []
    for i, ex in enumerate(sample):
        retrieved_ids = cosine_top_k(query_vecs[i], chunk_vecs, chunk_ids, k=top_k)
        context = [id_to_text[cid] for cid in retrieved_ids]
        try:
            raw = call_llm(build_prompt(ex["question"], context), llm_model=llm_model)
        except Exception as exc:                      # a dead Ollama shouldn't void the whole sweep
            raw = ""
            print(f"  LLM call failed on {ex.get('id')}: {exc}")

        pred = extract_final_answer(raw)
        correct = answers_match(ex["exe_ans"], pred)
        results.append({
            "id": ex.get("id"),
            "question": ex["question"],
            "exe_ans": ex["exe_ans"],
            "generated": pred,
            "correct": correct,
            "retrieved_ids": retrieved_ids,
            "context_chars": sum(len(c) for c in context),
            # TODO: exact chunk_id match, so this is only comparable within one id
            # scheme. It does not affect `accuracy`, but do not read
            # retrieval_hit_rate across differently-ID'd strategies (a whole_table
            # chunk can never equal a table_row gold id).
            "gold_hit": bool(set(retrieved_ids) & set(ex.get("gold_chunk_ids", []))),
        })
        if verbose:
            mark = "\u2713" if correct else "\u2717"
            print(f"{mark}  exe_ans={ex['exe_ans']:<12} generated={pred[:40]!r}")

    n = len(results)
    return {
        "accuracy": sum(r["correct"] for r in results) / n if n else 0.0,
        "retrieval_hit_rate": sum(r["gold_hit"] for r in results) / n if n else 0.0,
        "n": n,
        "results": results,
    }
