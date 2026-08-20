"""Strategy 2 (control): LangChain RecursiveCharacterTextSplitter over flat text.

Deliberately structure-blind -- concatenates pre_text, table, and post_text
into one blob and splits it the way a generic RAG pipeline would. This is the
baseline the structure-aware FinQA strategies have to beat.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking import attach_chunk_ids

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def flatten_document(doc: dict) -> str:
    pre = " ".join(item["text"] for item in doc["pre_text"])
    table_flat = " ".join(" ".join(row) for row in doc["table"])
    post = " ".join(item["text"] for item in doc["post_text"])
    return f"{pre} {table_flat} {post}"


def naive_fixed_size_chunks(doc: dict, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    splitter = SPLITTER if (chunk_size == 500 and overlap == 50) else RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = splitter.split_text(flatten_document(doc))
    chunks = [
        {"chunk_type": "fixed_size", "row_index": i, "text": text, "is_noise": False}
        for i, text in enumerate(texts)
    ]
    return attach_chunk_ids(doc["doc_id"], chunks)
