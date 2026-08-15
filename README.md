# 🤖 RAG Document Q&A — Telegram Bot

Same RAG pipeline as the [Streamlit demo](../rag-demo) (`chunking.py`,
`pdf_reader.py`, `rag_engine.py` are copied unchanged), wrapped as a
Telegram bot instead of a web app — proof that the retrieval/generation
logic isn't tied to one front-end.

## How it's different from the Streamlit version

Streamlit gives each browser tab its own `st.session_state` for free. A
Telegram bot is a single long-running process serving many users at once,
so this bot tracks each user's document and chat history itself, keyed by
`chat_id` (see `user_sessions` in `bot.py`). Tested for multi-user
isolation — one user's document never leaks into another user's session.

## Usage

- Send a PDF → bot extracts, chunks, and indexes it
- Or `/paste` → send text as your next message instead
- Then just type questions as normal messages
- `/new` — clear the current document and start over

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
TELEGRAM_BOT_TOKEN=your_bot_token_here
GEMINI_API_KEY=your_gemini_key_here
```

Get a bot token from [@BotFather](https://t.me/BotFather) on Telegram
(`/newbot`, follow the prompts).

Run:
```bash
python bot.py
```

## Testing without a live Telegram connection

`bot.py`'s handlers are plain `async def` functions that take
`(update, context)` — they don't have to run through Telegram's servers to
be tested. The state-machine logic (session creation, `/paste` flow,
multi-user isolation, `/new` reset) can be verified by calling them
directly with mocked `Update`/`Context` objects. This is how this bot was
tested during development, without needing a live bot token.

## Deployment note

This uses polling (`app.run_polling()`), which needs a process that stays
running continuously — it can't run on Streamlit Community Cloud, which is
built for request/response web apps. For a real deployment: a small
always-on host (a cheap VPS, Railway, Render) or a webhook-based setup
behind a proper HTTPS endpoint.

## Limitations

- **PDF only** for file uploads (same as the Streamlit demo — no OCR).
- **In-memory sessions.** If the bot process restarts, all active
  documents and chat histories are lost. For production use, this would
  need to move to persistent storage (Redis/a database).
- **Single-process only.** The `user_sessions` dict lives in one process's
  memory — this won't work correctly if the bot is ever scaled across
  multiple worker processes without moving session state to shared storage.
