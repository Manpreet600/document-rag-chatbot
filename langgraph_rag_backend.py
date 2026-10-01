from __future__ import annotations

import os
import sqlite3
import time
import threading
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# Local embeddings: free, no rate limit, so big books can be indexed.
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

# -------------------
# SQLite (checkpointer + thread titles + document metadata)
# -------------------
conn = sqlite3.connect(database="connect.db", check_same_thread=False)
_db_lock = threading.Lock()  # FastAPI runs sync endpoints in a thread pool

with _db_lock:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chat_threads(thread_id TEXT PRIMARY KEY, title TEXT)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS thread_documents(
            thread_id TEXT PRIMARY KEY, filename TEXT, documents INTEGER, chunks INTEGER)"""
    )
    conn.commit()

checkpointer = SqliteSaver(conn=conn)

# -------------------
# PDF retriever store (per thread). FAISS indexes are saved to disk so
# they survive a server restart, and are lazily reloaded.
# -------------------
INDEX_DIR = "faiss_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_INGEST: Dict[str, dict] = {}  # thread_id -> live indexing status


def _index_path(thread_id: str) -> str:
    return os.path.join(INDEX_DIR, str(thread_id))


def _as_retriever(store):
    # MMR returns varied passages, which works better on a big book than plain top-k.
    return store.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 40})


def _get_retriever(thread_id: Optional[str]):
    if not thread_id:
        return None
    thread_id = str(thread_id)
    if thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    path = _index_path(thread_id)
    if os.path.isdir(path):
        store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        _THREAD_RETRIEVERS[thread_id] = _as_retriever(store)
        return _THREAD_RETRIEVERS[thread_id]
    return None


def get_ingest_status(thread_id: str) -> dict:
    return _INGEST.get(str(thread_id), {"state": "idle"})


def start_ingest(pdf_path: str, thread_id: str, filename: str) -> None:
    """Index a PDF in a background thread so the request returns immediately."""
    tid = str(thread_id)
    if _INGEST.get(tid, {}).get("state") == "running":
        raise ValueError("A PDF is already being indexed for this chat.")
    _INGEST[tid] = {"state": "running", "stage": "Reading PDF", "progress": 0}
    threading.Thread(target=_ingest_worker, args=(pdf_path, tid, filename), daemon=True).start()


def _ingest_worker(pdf_path: str, tid: str, filename: str, batch_size: int = 128) -> None:
    try:
        docs = PyPDFLoader(pdf_path).load()
        chars = sum(len(d.page_content.strip()) for d in docs)
        if not docs or chars / len(docs) < 100:
            raise ValueError(
                "This PDF has almost no text (it looks scanned). Run OCR on it first, "
                "for example: ocrmypdf --force-ocr in.pdf out.pdf"
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
        )
        chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]

        store = None
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            store = FAISS.from_documents(batch, embeddings) if store is None else store
            if i > 0:
                store.add_documents(batch)
            done = min(i + batch_size, len(chunks))
            _INGEST[tid] = {
                "state": "running",
                "stage": f"Embedding {done}/{len(chunks)} chunks",
                "progress": int(done * 100 / len(chunks)),
            }

        store.save_local(_index_path(tid))
        _THREAD_RETRIEVERS[tid] = _as_retriever(store)

        with _db_lock:
            conn.execute(
                "INSERT OR REPLACE INTO thread_documents VALUES (?, ?, ?, ?)",
                (tid, filename, len(docs), len(chunks)),
            )
            conn.commit()
        _INGEST[tid] = {"state": "done", "progress": 100}
    except Exception as e:
        _INGEST[tid] = {"state": "error", "message": str(e)}
    finally:
        try:
            os.remove(pdf_path)
        except OSError:
            pass


# -------------------
# Tools
# -------------------
search_tool = DuckDuckGoSearchResults(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic on two numbers.
    Supported operations: add, subtract, multiplication, division
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "subtract":
            result = first_num - second_num
        elif operation == "multiplication":
            result = first_num * second_num
        elif operation == "division":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"unsupported operation '{operation}'"}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def stock_price_calculator(symbol: str) -> dict:
    """Fetch the latest stock price for a symbol (e.g. 'AAPL', 'TSLA') using Alpha Vantage."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "ALPHAVANTAGE_API_KEY is not set"}
    url = "https://www.alphavantage.co/query"
    r = requests.get(
        url,
        params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
        timeout=15,
    )
    return r.json()


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Use this tool whenever the user asks a question about an uploaded PDF or document.
    Always retrieve information from the uploaded document before answering.
    Never answer document questions from memory. The document may be a very long book,
    so use specific search queries and call this tool several times for broad questions.
    """
    thread_id = config["configurable"]["thread_id"]
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No document indexed for this chat. Upload a PDF first.", "query": query}

    result = retriever.invoke(query)
    return {
        "query": query,
        "context": [d.page_content for d in result],
        "pages": [int(d.metadata.get("page", 0)) + 1 for d in result],
        "source_file": thread_document_metadata(thread_id).get("filename"),
    }


tools = [search_tool, calculator, stock_price_calculator, rag_tool]
model_with_tools = model.bind_tools(tools)


# -------------------
# Graph
# -------------------
class ChatState(TypedDict):
    # thread_id is no longer stored in state: the config reaches the tools automatically.
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant. When the user asks about their uploaded document or book, "
    "use rag_tool. For broad questions (summaries, lists of chapters, comparisons) call it "
    "several times with different specific queries, then answer from what it returns and "
    "mention page numbers. If the passages do not contain the answer, say so."
))


def chat_node(state: ChatState, config: RunnableConfig):
    response = model_with_tools.invoke([SYSTEM_PROMPT] + state["messages"], config=config)
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# Helpers used by the FastAPI layer
# -------------------
def text_of(content) -> str:
    """Gemini may return a string or a list of content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) and p.get("type") == "text" else (p if isinstance(p, str) else "")
            for p in content
        )
    return ""


def make_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": str(thread_id)},
        "metadata": {"thread_id": str(thread_id)},
        "run_name": "chat_turn",
    }


def create_thread(thread_id: str, title: str = "New Chat") -> dict:
    with _db_lock:
        conn.execute(
            "INSERT OR IGNORE INTO chat_threads(thread_id, title) VALUES (?, ?)",
            (str(thread_id), title),
        )
        conn.commit()
    return {"thread_id": str(thread_id), "title": get_chat_title(thread_id)}


def get_chat_title(thread_id: str) -> Optional[str]:
    with _db_lock:
        row = conn.execute(
            "SELECT title FROM chat_threads WHERE thread_id = ?", (str(thread_id),)
        ).fetchone()
    return row[0] if row else None


def save_chat_title(thread_id: str, title: str) -> None:
    with _db_lock:
        conn.execute(
            "INSERT OR REPLACE INTO chat_threads(thread_id, title) VALUES (?, ?)",
            (str(thread_id), title),
        )
        conn.commit()


def generate_title(first_message: str) -> str:
    prompt = (
        "Generate a short title (3-5 words) for a conversation that starts with this message.\n\n"
        f"{first_message}\n\nOnly return the title."
    )
    try:
        return text_of(model.invoke(prompt).content).strip().strip('"') or "New Chat"
    except Exception:
        return first_message[:30] or "New Chat"


def retrieve_all_thread() -> list[dict]:
    with _db_lock:
        rows = conn.execute("SELECT thread_id, title FROM chat_threads").fetchall()
    return [{"thread_id": r[0], "title": r[1]} for r in reversed(rows)]  # newest first


def load_conversation(thread_id: str) -> list[dict]:
    """Return only user messages and final assistant text (no tool-call plumbing)."""
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    out = []
    for m in (state.values.get("messages", []) if state and state.values else []):
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": text_of(m.content)})
        elif isinstance(m, AIMessage):
            text = text_of(m.content)
            if text.strip():
                out.append({"role": "assistant", "content": text})
    return out


def thread_document_metadata(thread_id: str) -> dict:
    with _db_lock:
        row = conn.execute(
            "SELECT filename, documents, chunks FROM thread_documents WHERE thread_id = ?",
            (str(thread_id),),
        ).fetchone()
    return {"filename": row[0], "documents": row[1], "chunks": row[2]} if row else {}
