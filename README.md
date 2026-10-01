# Document RAG Chatbot

Chat with your PDFs, including large books of 1000+ pages. The app is built with **LangGraph**, **Gemini**, **FAISS** and **FastAPI**, and has a streaming web interface.

## Features

- Upload a PDF and ask questions about it; answers include page numbers
- Handles large books: indexing runs in the background with live progress
- Free local embeddings (`BAAI/bge-small-en-v1.5`), so there are no API rate limits when indexing
- Streaming replies with a visible "Using tool" indicator
- Multiple chats with auto-generated titles; history is saved in SQLite
- Each chat has its own PDF, and indexes are saved to disk so they survive restarts
- Extra tools: web search (DuckDuckGo), calculator and stock prices (Alpha Vantage)

## Tech Stack

| Part | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Agent / graph | LangGraph + LangChain |
| Vector store | FAISS |
| Embeddings | Hugging Face `bge-small-en-v1.5` (local) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML, CSS, vanilla JavaScript (SSE streaming) |
| Storage | SQLite (chat history and threads) |

## Project Structure

```
files/
├── main.py                    # FastAPI app and API endpoints
├── langgraph_rag_backend.py   # LangGraph agent, tools, PDF indexing
├── requirements.txt
├── .env                       # your API keys (not committed)
└── static/
    └── index.html             # chat interface
```

## Setup

1. Clone the repository and open the project folder:
```bash
   git clone https://github.com/<your-username>/document-rag-chatbot.git
   cd document-rag-chatbot
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Create a `.env` file:
```
   GOOGLE_API_KEY=your_gemini_api_key
   ALPHAVANTAGE_API_KEY=your_alpha_vantage_key   # optional, for the stock tool
```
   Get a Gemini key at https://aistudio.google.com/apikey

4. Run the server:
```bash
   python -m uvicorn main:app --reload
```

5. Open http://127.0.0.1:8000 in your browser.

The first run downloads the embedding model (about 130 MB).

## How to Use

1. Click **New chat**.
2. Click **Upload a PDF** and wait until the sidebar shows `Using yourfile.pdf` with the chunk count.
3. Ask questions about the document.

A 1000+ page book can take 5 to 15 minutes to index on a CPU. Don't edit code files while indexing, because the server reload cancels the job.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/threads` | List chats |
| POST | `/api/threads` | Create a chat |
| GET | `/api/threads/{id}/messages` | Load chat history |
| POST | `/api/threads/{id}/upload` | Upload a PDF (indexed in background) |
| GET | `/api/threads/{id}/ingest-status` | Indexing progress |
| GET | `/api/threads/{id}/document` | Indexed PDF info |
| POST | `/api/threads/{id}/chat` | Send a message (streamed reply) |

## Limitations

- Scanned PDFs without a text layer need OCR first, for example:
```bash
  ocrmypdf --force-ocr input.pdf output.pdf
```
- One PDF per chat (uploading a new one replaces it)
- No authentication: don't expose the app to the public internet as it is

## Notes

- Never commit `.env`, `connect.db` or `faiss_indexes/`. Add them to `.gitignore`.

## License

MIT
