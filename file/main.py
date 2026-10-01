import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel

import langgraph_rag_backend as be

app = FastAPI(title="LangGraph RAG Chatbot")
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/threads")
def list_threads():
    return be.retrieve_all_thread()


@app.post("/api/threads")
def new_thread():
    return be.create_thread(str(uuid.uuid4()))


@app.get("/api/threads/{thread_id}/messages")
def get_messages(thread_id: str):
    return be.load_conversation(thread_id)


@app.get("/api/threads/{thread_id}/document")
def get_document(thread_id: str):
    return be.thread_document_metadata(thread_id)


@app.post("/api/threads/{thread_id}/upload")
def upload_pdf(thread_id: str, file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    # Stream to disk in 1 MB pieces so a 100 MB+ book never sits in memory.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp, length=1024 * 1024)
        path = tmp.name
    try:
        be.start_ingest(path, thread_id, file.filename)
    except ValueError as e:
        os.remove(path)
        raise HTTPException(409, str(e))
    return {"state": "running"}


@app.get("/api/threads/{thread_id}/ingest-status")
def ingest_status(thread_id: str):
    return be.get_ingest_status(thread_id)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/api/threads/{thread_id}/chat")
def chat(thread_id: str, req: ChatRequest):
    text = req.message.strip()
    if not text:
        raise HTTPException(400, "Message is empty.")

    be.create_thread(thread_id)
    new_title = None
    if be.get_chat_title(thread_id) == "New Chat":
        new_title = be.generate_title(text)
        be.save_chat_title(thread_id, new_title)

    def event_stream():
        if new_title:
            yield _sse({"type": "title", "title": new_title})
        try:
            for chunk, _meta in be.chatbot.stream(
                {"messages": [HumanMessage(content=text)]},
                config=be.make_config(thread_id),
                stream_mode="messages",
            ):
                if not isinstance(chunk, AIMessageChunk):
                    continue
                for tc in chunk.tool_call_chunks or []:
                    if tc.get("name"):
                        yield _sse({"type": "tool", "name": tc["name"]})
                token = be.text_of(chunk.content)
                if token:
                    yield _sse({"type": "token", "content": token})
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})
        yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
