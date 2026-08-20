"""End-to-end eval: retrieve, ask the LLM, grade the answer.

LLM calls go through LangChain ChatOllama. Retrieval uses the shared embedder
and cosine top-k. Ground truth is exe_ans, the executed FinQA program result,
not the human-typed `answer` field.
"""

import re

from langchain_core.prompts import ChatPromptTemplate

from src.eval.retrieval_harness import cosine_top_k, embed_texts, word_overlap_relevance
from src.models import LLM_MODEL, load_llm

PROMPT = ChatPromptTemplate.from_template(
    """You are a financial analyst. Use ONLY the context below to answer the question.
Show your arithmetic briefly, then finish with a line of exactly the form:
ANSWER: <single number, or yes/no>
Give percentages as a percent value (e.g. 14.3%).

Context:
{context}

Question: {question}
"""
)


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
    """gold is exe_ans: a float for numeric questions, "yes"/"no" for boolean ones."""
    if isinstance(gold, str):
        g = gold.strip().lower()
        if g in ("yes", "no"):
            return g in re.findall(r"[a-z]+", (pred or "").lower())[:6]
        return g == (pred or "").strip().lower()

    if gold is None:
        return False

    g = float(gold)
    p = parse_number(pred)
    if p is None:
        return False

    for cand in (p, p / 100.0, p * 100.0):
        for gg in (g, abs(g)):
            if abs(gg - abs(cand)) <= tol * max(abs(gg), 1e-6) or abs(gg - cand) <= 1e-4:
                return True
    return False


def call_llm(prompt: str, llm_model: str = LLM_MODEL, timeout: int = 180) -> str:
    del timeout
    llm = load_llm(llm_model)
    return llm.invoke(prompt).content.strip()


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
    return PROMPT.format(context=context, question=question)


def extract_final_answer(raw: str) -> str:
    """Prefer the explicit ANSWER: line; fall back to the whole response."""
    hits = re.findall(r"ANSWER:\s*(.+)", raw or "", flags=re.IGNORECASE)
    return hits[-1].strip() if hits else (raw or "").strip()


def _grade_one(ex, retrieved_ids, context, llm_model, verbose, gold_texts=None):
    try:
        raw = call_llm(build_prompt(ex["question"], context), llm_model=llm_model)
    except Exception as exc:
        raw = ""
        print(f"  LLM call failed on {ex.get('id')}: {exc}")

    pred = extract_final_answer(raw)
    correct = answers_match(ex["exe_ans"], pred)
    if verbose:
        mark = "\u2713" if correct else "\u2717"
        print(f"{mark}  exe_ans={ex['exe_ans']:<12} generated={pred[:40]!r}")

    if gold_texts is not None:
        gold_hit = any(word_overlap_relevance(ctx, gt) for ctx in context for gt in gold_texts)
    else:
        gold_hit = bool(set(retrieved_ids) & set(ex.get("gold_chunk_ids", [])))

    return {
        "id": ex.get("id"),
        "question": ex["question"],
        "exe_ans": ex["exe_ans"],
        "generated": pred,
        "correct": correct,
        "retrieved_ids": retrieved_ids,
        "context_chars": sum(len(c) for c in context),
        "gold_hit": gold_hit,
    }


def _summarize(results: list[dict]) -> dict:
    n = len(results)
    return {
        "accuracy": sum(r["correct"] for r in results) / n if n else 0.0,
        "retrieval_hit_rate": sum(r["gold_hit"] for r in results) / n if n else 0.0,
        "n": n,
        "results": results,
    }


def run_generation_eval(strategy_chunks: list[dict], sample: list[dict], model,
                         top_k: int = 5, llm_model: str = LLM_MODEL, verbose: bool = True,
                         gold_text_by_id: dict | None = None) -> dict:
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
        gold_texts = None
        if gold_text_by_id is not None:
            gold_texts = [gold_text_by_id[g] for g in ex.get("gold_chunk_ids", []) if g in gold_text_by_id]
        results.append(_grade_one(ex, retrieved_ids, context, llm_model, verbose, gold_texts=gold_texts))
    return _summarize(results)
