"""Agentic CLI RAG app (v2): Contextual Retrieval + tool-calling agent + conversation compression.

What changed from v1
- The LLM is now an agent. It decides whether to search the policy, how many times, and with
  which query. It can also answer small talk directly or ask a clarifying question.
- Retrieval (hybrid vector + BM25, LLM rerank) is exposed to the agent as a tool.
- Conversation memory: recent turns are kept verbatim; once the history grows past a limit,
  older turns are compressed into a running summary.

Requires a chat model that supports tool calling (set OPENROUTER_MODEL accordingly).
"""
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
CACHE_FILE = Path("contextualized_chunks.json")
CHROMA_DIR = "chroma_db"
BASE_DIR = Path(__file__).resolve().parent
POLICY_FILE = Path(os.getenv("POLICY_FILE", BASE_DIR / "policy.txt"))

RETRIEVE_K = 4        # per retriever (vector and BM25)
RERANK_TOP_N = 4      # chunks returned by one search_policy call
MAX_AGENT_STEPS = 5   # max LLM turns per question (tool calls + final answer)

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))  # compress above this
KEEP_RECENT_MESSAGES = 4                                             # kept verbatim after compression
VERBOSE = os.getenv("VERBOSE", "1") == "1"                           # print agent decisions


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


def log(message: str) -> None:
    if VERBOSE:
        print(f"  [{message}]")


def text_of(message: BaseMessage) -> str:
    """Return message content as plain text (some models return a list of content parts)."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content).strip()


# --------------------------------------------------------------------------------------
# Indexing (same pipeline as v1)
# --------------------------------------------------------------------------------------
def create_chunks() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100, separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.create_documents([POLICY.strip()])
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk-{i}"
    return chunks


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
        context = text_of(llm.invoke(prompt))
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


# --------------------------------------------------------------------------------------
# Retrieval (same as v1, now used as an agent tool)
# --------------------------------------------------------------------------------------
def retrieve(question: str, store: Chroma, bm25: BM25Retriever) -> list[Document]:
    """Hybrid retrieval: vector top-k + BM25 top-k, deduplicated by chunk ID."""
    results = store.similarity_search(question, k=RETRIEVE_K) + bm25.invoke(question)
    unique = {}
    for doc in results:
        unique.setdefault(doc.metadata["chunk_id"], doc)
    return list(unique.values())


def rerank(question: str, candidates: list[Document]) -> list[Document]:
    """Ask the LLM to pick the most relevant chunk IDs."""
    listing = "\n\n".join(f"[{d.metadata['chunk_id']}]\n{d.page_content}" for d in candidates)
    prompt = (
        f"Question: {question}\n\nCandidate chunks:\n\n{listing}\n\n"
        f"Return the IDs of the {RERANK_TOP_N} chunks most useful for answering the question, "
        "most relevant first. Include any chunk that states an exception or special rule "
        "affecting the answer. Respond ONLY with a JSON list of IDs, e.g. "
        '["chunk-1", "chunk-4"].'
    )
    reply = text_of(llm.invoke(prompt))
    by_id = {d.metadata["chunk_id"]: d for d in candidates}
    ranked_ids = [i for i in re.findall(r"chunk-\d+", reply) if i in by_id]
    ranked_ids = list(dict.fromkeys(ranked_ids))[:RERANK_TOP_N]
    if not ranked_ids:  # fallback if the LLM output can't be parsed
        return candidates[:RERANK_TOP_N]
    return [by_id[i] for i in ranked_ids]


def build_tools(store: Chroma, bm25: BM25Retriever) -> dict:
    @tool
    def search_policy(query: str) -> str:
        """Search Meridian's patient policy and return the most relevant excerpts.

        Use a short, self-contained query (include the key terms, not pronouns).
        Call it again with different wording if the excerpts do not answer the question,
        or once per sub-question when the user asks about several things.
        """
        candidates = retrieve(query, store, bm25)
        top = rerank(query, candidates)
        log(f"search_policy({query!r}) -> {[d.metadata['chunk_id'] for d in top]}")
        return "\n\n".join(f"[{d.metadata['chunk_id']}] {d.metadata['original']}" for d in top)

    return {"search_policy": search_policy}


# --------------------------------------------------------------------------------------
# Conversation memory with compression
# --------------------------------------------------------------------------------------
@dataclass
class Memory:
    summary: str = ""
    messages: list[BaseMessage] = field(default_factory=list)  # final user/assistant turns only

    def add_turn(self, question: str, answer: str) -> None:
        self.messages += [HumanMessage(question), AIMessage(answer)]
        self.compress_if_needed()

    def compress_if_needed(self) -> None:
        """Once history exceeds the limit, fold older turns into the running summary."""
        if len(self.messages) <= MAX_HISTORY_MESSAGES:
            return
        old, recent = self.messages[:-KEEP_RECENT_MESSAGES], self.messages[-KEEP_RECENT_MESSAGES:]
        transcript = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {text_of(m)}" for m in old
        )
        prompt = (
            "Update the running summary of a conversation between a patient and Meridian's "
            "policy assistant. Keep: facts the user stated about themselves or their situation, "
            "what they asked, what the assistant concluded, and anything still unresolved. "
            "Be brief (under 150 words). Do not add information that is not in the conversation.\n\n"
            f"Current summary:\n{self.summary or '(none)'}\n\n"
            f"Older messages to fold in:\n{transcript}\n\nUpdated summary:"
        )
        self.summary = text_of(llm.invoke(prompt))
        self.messages = recent
        log(f"memory compressed: {len(old)} messages summarised, {len(recent)} kept")


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------
AGENT_SYSTEM_PROMPT = """You are an assistant for Meridian's patient policy. You decide how to \
handle each message.

Tools: search_policy(query) returns excerpts from the policy.

How to work:
- Greetings, thanks and small talk: reply briefly, no search.
- Questions about the policy: ALWAYS search before answering. The conversation summary and \
earlier messages are context only, never a source of policy facts.
- If the user's question depends on earlier messages, write a self-contained search query.
- If a question has several parts, search for each part separately.
- If the excerpts do not answer the question, search again with different wording before \
concluding anything. Only say "the policy does not specify it" after at least one search \
(and a second, differently worded search when the first came back empty-handed).
- If the question is too ambiguous to search or answer, ask one short clarifying question.
- If the policy says the requested information is confidential, say that it is confidential.

Answer rules:
- Use ONLY the retrieved excerpts. No outside knowledge, no guessing.
- If a general rule and an exception both apply, state both.
- Only state an exception if the excerpts state it explicitly.
- Be concise and precise.
"""


def run_agent(question: str, memory: Memory, tools: dict) -> str:
    system = AGENT_SYSTEM_PROMPT
    if memory.summary:
        system += f"\nSummary of the earlier conversation:\n{memory.summary}\n"

    agent = llm.bind_tools(list(tools.values()))
    messages: list[BaseMessage] = [SystemMessage(system), *memory.messages, HumanMessage(question)]

    for _ in range(MAX_AGENT_STEPS):
        ai = agent.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            if not text_of(ai):
                break
            return text_of(ai)
        for call in ai.tool_calls:
            try:
                result = tools[call["name"]].invoke(call["args"])
            except Exception as exc:  # keep the agent loop alive on tool errors
                result = f"Tool error: {exc}"
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    # Step budget used up (or empty reply): force a final answer from what was retrieved.
    log("step limit reached, forcing final answer")
    messages.append(
        HumanMessage(
            "Give your final answer now, using only the excerpts retrieved above. "
            "If they do not answer the question, say the policy does not specify it."
        )
    )
    return text_of(llm.invoke(messages))


def chat() -> None:
    docs = contextualize_chunks(create_chunks())
    store = build_vector_store(docs)
    bm25 = build_bm25(docs)
    tools = build_tools(store, bm25)
    memory = Memory()
    print("Policy agent ready. Type 'exit' to quit, '/reset' to clear the conversation.\n")

    while True:
        question = input("User: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question == "/reset":
            memory = Memory()
            print("Conversation cleared.\n")
            continue
        if not question:
            continue
        answer = run_agent(question, memory, tools)
        memory.add_turn(question, answer)
        print(f"A: {answer}\n")


if __name__ == "__main__":
    chat()