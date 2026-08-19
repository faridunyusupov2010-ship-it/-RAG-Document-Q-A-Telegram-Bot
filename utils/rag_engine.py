"""
rag_engine.py

Core retrieval-augmented generation logic:
  1. Embed each chunk (sentence-transformers, multilingual model)
  2. Store in an in-memory Chroma collection (ephemeral — scoped to one
     Streamlit session, not shared across users, and not written to disk)
  3. On a question: embed the question, retrieve the top-k most similar
     chunks, and pass ONLY those chunks to Gemini as context
  4. The prompt explicitly instructs the model to say when the answer isn't
     in the provided context, rather than making something up — this is the
     main difference between "a chatbot" and "a chatbot you can trust with
     your company's documents"
"""

import os
import uuid
import logging
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

load_dotenv()
logger = logging.getLogger(__name__)

# =========================
# Configuration
# =========================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 4


# Load embedding model only once per process
_embedding_model = None


# =========================
# Embedding Model
# =========================

def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and reuse it.

    Loading the model is relatively slow, while generating
    embeddings after loading is much faster.
    """
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _embedding_model


# =========================
# ChromaDB
# =========================

def build_collection(chunks: list[str]):
    """
    Create a persistent Chroma collection for a document.
    """

    model = get_embedding_model()

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection_name = f"doc_{uuid.uuid4().hex[:8]}"

    collection = client.get_or_create_collection(
        name=collection_name
    )

    embeddings = model.encode(
        chunks,
        show_progress_bar=False
    ).tolist()

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
    )

    return collection


def get_collection(collection_name: str):
    """
    Load an existing persistent Chroma collection by name.
    """

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    return client.get_collection(
        name=collection_name
    )


def delete_collection(collection_name: str) -> None:
    """
    Delete a document collection from persistent Chroma.
    """

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    client.delete_collection(
        name=collection_name
    )


# =========================
# Retrieval
# =========================

def retrieve(
    collection,
    question: str,
    top_k: int = TOP_K,
) -> list[str]:
    """
    Convert the user's question into an embedding
    and retrieve the most relevant document chunks.
    """

    model = get_embedding_model()

    query_embedding = model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(
            top_k,
            collection.count(),
        ),
    )

    return (
        results["documents"][0]
        if results["documents"]
        else []
    )


# =========================
# Question Answering
# =========================

def answer_question(
    collection,
    question: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Retrieve relevant chunks and ask Groq to answer
    using only those chunks.
    """

    retrieved_chunks = retrieve(
        collection,
        question,
    )

    if not retrieved_chunks:
        return {
            "answer": (
                "I couldn't find anything relevant "
                "to that question in the document."
            ),
            "sources": [],
        }

    # -------------------------
    # Build context
    # -------------------------

    context = "\n\n---\n\n".join(
        retrieved_chunks
    )

    # -------------------------
    # Conversation history
    # -------------------------

    history_text = ""

    if chat_history:
        # Keep only the last 6 messages
        recent = chat_history[-6:]

        history_text = "\n".join(
            (
                f"{'User' if turn['role'] == 'user' else 'Assistant'}: "
                f"{turn['content']}"
            )
            for turn in recent
        )

        history_text = (
            "\nConversation so far:\n"
            f"{history_text}\n"
        )

    # -------------------------
    # Prompt
    # -------------------------

    prompt = f"""
You are a helpful assistant that answers questions
using ONLY the document excerpts provided below.

Follow these rules strictly:

1. Answer only using information found in the excerpts.
2. If the answer is not contained in the excerpts,
   clearly say that the document doesn't cover it.
3. Do not guess.
4. Do not use outside knowledge.
5. Answer in the same language as the user's question.
6. Be concise and direct.

{history_text}

Document excerpts:
{context}

Question:
{question}
"""

    # -------------------------
    # GROQ API KEY
    # -------------------------

    api_key = os.environ.get(
        "GROQ_API_KEY"
    )

    if not api_key:
        return {
            "answer": (
                "Error: GROQ_API_KEY not found. "
                "Add it to your .env file."
            ),
            "sources": retrieved_chunks,
        }

    # -------------------------
    # Groq LLM
    # -------------------------

    try:
        llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.2,
            api_key=api_key,
        )

        response = llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": retrieved_chunks,
        }

    except Exception:
        logger.exception(
            "Groq request failed"
        )

        return {
            "answer": (
                "❌ I couldn't generate an answer right now. "
                "Please try again."
            ),
            "sources": retrieved_chunks,
        }