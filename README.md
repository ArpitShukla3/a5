# Meridian Policy Assistant

A command-line RAG app that answers questions about Meridian's patient policy, using Contextual Retrieval with LangChain, Chroma and BM25. Models are accessed through OpenRouter.

## What it does

1. **Load policy:** reads the policy text from `policy.txt`.
2. **Chunk:** splits the text with `RecursiveCharacterTextSplitter` (600 characters, 100 overlap).
3. **Contextualize:** for each chunk, the LLM writes a 1-2 sentence context using the full policy, which is prepended to the chunk. Results are cached in `contextualized_chunks.json`.
4. **Index:** the contextualized chunks are stored in a Chroma vector store (`chroma_db/`) and a BM25 index.
5. **Retrieve:** for each question, takes the top 4 from vector search and the top 4 from BM25, deduplicated by chunk ID.
6. **Rerank:** the LLM selects the 3 most useful chunks, including any that state an exception.
7. **Answer:** the LLM answers using only the original text of those chunks. If the policy does not cover the question, it says so.

## Setup

Install dependencies:

```bash
pip install python-dotenv langchain-core langchain-community langchain-openai \
            langchain-chroma langchain-text-splitters chromadb rank_bm25
```

Place `policy.txt` next to the script (UTF-8, blank line between sections).

Create a `.env` file:

```env
OPENROUTER_API_KEY=your-key
OPENROUTER_MODEL=your-chat-model

# Optional
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
POLICY_FILE=/path/to/policy.txt
```

## Run

```bash
python app.py
```

The first run contextualizes every chunk (one LLM call each). Later runs load from the cache. Type `exit` or `quit` to leave.

## Caching

- The contextualization cache is keyed by a hash of the policy text and chunks, so editing `policy.txt` regenerates it.
- The Chroma collection name includes a content hash, so changed content gets a new collection. Old collections stay in `chroma_db/`.
- To rebuild from scratch, delete `contextualized_chunks.json` and `chroma_db/`.

## Files

| File | Purpose |
|------|---------|
| `app.py` | The application |
| `policy.txt` | Policy source text |
| `.env` | API key and model settings |
| `contextualized_chunks.json` | Generated cache |
| `chroma_db/` | Generated vector store |# a5
