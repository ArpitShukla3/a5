"""Simple CLI RAG app: Anthropic-style Contextual Retrieval with LangChain + OpenRouter.
Hybrid retrieval (vector + BM25) followed by a cross-encoder reranker.
"""
import hashlib
import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env first so LangSmith picks up LANGSMITH_* variables before any tracing happens.
load_dotenv()

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from sentence_transformers import CrossEncoder

OPENROUTER_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
# Local cross-encoder. Alternatives: "BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CACHE_FILE = Path("contextualized_chunks.json")
CHROMA_DIR = "chroma_db"
RETRIEVE_K = 4    # per retriever (vector and BM25)
RERANK_TOP_N = 3  # chunks sent to the final answer step
BASE_DIR = Path(__file__).resolve().parent
POLICY_FILE = Path(os.getenv("POLICY_FILE", BASE_DIR / "policy.txt"))


def load_policy(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Policy file is empty: {path}")
    return text


POLICY = load_policy(POLICY_FILE)

llm = ChatOpenAI(
    model=os.environ["OPENROUTER_MODEL"],
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=OPENROUTER_URL,
    temperature=0,
)

# Loaded once at startup (downloads the model on first run, then uses the local cache).
reranker = CrossEncoder(RERANKER_MODEL)


def create_chunks() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100, separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.create_documents([POLICY.strip()])
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk-{i}"
    return chunks


@traceable(
    name="contextualize_chunks",
    run_type="chain",
    process_inputs=lambda inputs: {"num_chunks": len(inputs["chunks"])},
    process_outputs=lambda out: {"num_docs": len(out)},
)
def contextualize_chunks(chunks: list[Document]) -> list[Document]:
    """Prepend an LLM-written context to each chunk. Cached on disk."""
    key = hashlib.sha256((POLICY + "".join(c.page_content for c in chunks)).encode()).hexdigest()
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
        if cache.get("key") == key:
            print("Loaded contextualized chunks from cache.")
            return [Document(page_content=d["text"], metadata=d["metadata"]) for d in cache["docs"]]

    print(f"Contextualizing {len(chunks)} chunks (one-off, cached afterwards)...")
    docs = []
    for chunk in chunks:
        prompt = (
            f"<policy_document>\n{POLICY.strip()}\n</policy_document>\n\n"
            f"<chunk>\n{chunk.page_content}\n</chunk>\n\n"
            "Given the complete policy document and this specific chunk, write a short context "
            "(1-2 sentences) explaining what this chunk is about and how it relates to the overall "
            "policy. Do not add information that is not present in the document. "
            "Reply with the context only."
        )
        context = llm.invoke(prompt).content.strip()
        docs.append(
            Document(
                page_content=f"{context}\n\n{chunk.page_content}",
                metadata={**chunk.metadata, "original": chunk.page_content},
            )
        )

    CACHE_FILE.write_text(
        json.dumps(
            {"key": key, "docs": [{"text": d.page_content, "metadata": d.metadata} for d in docs]},
            indent=2,
        )
    )
    return docs


def build_vector_store(docs: list[Document]) -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_URL,
        check_embedding_ctx_length=False,  # needed for non-OpenAI models via OpenRouter
    )
    # Collection name includes a content hash, so changed chunks get a fresh collection.
    digest = hashlib.sha256("".join(d.page_content for d in docs).encode()).hexdigest()[:12]
    store = Chroma(
        collection_name=f"policy_{digest}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    if store._collection.count() == 0:
        store.add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs])
    return store


def build_bm25(docs: list[Document]) -> BM25Retriever:
    return BM25Retriever.from_documents(docs, k=RETRIEVE_K)


@traceable(
    name="hybrid_retrieve",
    run_type="retriever",
    process_inputs=lambda inputs: {"question": inputs["question"]},  # hide store/bm25 objects
)
def retrieve(question: str, store: Chroma, bm25: BM25Retriever) -> list[Document]:
    """Hybrid retrieval: vector top-k + BM25 top-k, deduplicated by chunk ID."""
    results = store.similarity_search(question, k=RETRIEVE_K) + bm25.invoke(question)
    unique = {}
    for doc in results:
        unique.setdefault(doc.metadata["chunk_id"], doc)
    return list(unique.values())


@traceable(name="cross_encoder_rerank", run_type="chain", metadata={"reranker": RERANKER_MODEL})
def rerank(question: str, candidates: list[Document]) -> list[Document]:
    """Score each (question, chunk) pair with a cross-encoder and keep the top N."""
    if not candidates:
        return []
    # Score against the contextualized text (context + chunk). To score the raw chunk
    # instead, use d.metadata["original"].
    pairs = [(question, d.page_content) for d in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    # Copy docs with the score in metadata so it shows up in the LangSmith trace.
    return [
        Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": float(s)})
        for d, s in ranked[:RERANK_TOP_N]
    ]


@traceable(name="generate_answer", run_type="chain")
def generate_answer(question: str, chunks: list[Document]) -> str:
    context = "\n\n".join(f"[{d.metadata['chunk_id']}] {d.metadata['original']}" for d in chunks)
    prompt = (
        "You are an assistant for Meridian's patient policy. Answer the question using ONLY the "
        "policy excerpts below. Be concise and precise. If the excerpts contain a general rule "
        "and an exception that both apply, state both. If the policy does not specify the "
        "answer, say that the policy does not specify it. Do not use outside knowledge, and "
        "do not guess.\n\n"
        f"Policy excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return llm.invoke(prompt).content.strip()


@traceable(
    name="rag_pipeline",
    run_type="chain",
    process_inputs=lambda inputs: {"question": inputs["question"]},  # hide store/bm25 objects
)
def answer_question(question: str, store: Chroma, bm25: BM25Retriever) -> str:
    """Root trace for one question: retrieve -> rerank -> generate."""
    candidates = retrieve(question, store, bm25)
    top_chunks = rerank(question, candidates)
    return generate_answer(question, top_chunks)


def chat() -> None:
    docs = contextualize_chunks(create_chunks())
    store = build_vector_store(docs)
    bm25 = build_bm25(docs)
    session_id = str(uuid.uuid4())  # groups all questions from this run into one LangSmith thread
    print("Policy assistant ready. Type 'exit' to quit.\n")

    while True:
        question = input("User: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer = answer_question(
            question,
            store,
            bm25,
            langsmith_extra={"metadata": {"session_id": session_id}, "tags": ["policy-rag"]},
        )
        print(f"A: {answer}\n")


if __name__ == "__main__":
    chat()