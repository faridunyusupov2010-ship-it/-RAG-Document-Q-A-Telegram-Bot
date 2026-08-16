import time
from collections import defaultdict, deque
import logging
import os
import traceback
from utils.rag_engine import (
    build_collection,
    get_collection,
    delete_collection,
    answer_question,
)
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from utils.chunking import chunk_text
from utils.pdf_reader import extract_text_from_pdf
from utils.session_db import (
    init_db,
    save_session,
    set_awaiting_paste,
    get_session,
    delete_session,
    save_message,
    get_history,
    clear_history,
    update_activity,
    get_expired_sessions,
)
load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
MAX_PDF_SIZE = 10 * 1024 * 1024
MAX_PASTE_LENGTH = 100_000
# Rate limits
MAX_QUESTIONS_PER_MINUTE = 10
MAX_UPLOADS_PER_10_MINUTES = 3
# Per-chat state. In a single-process polling bot this is fine; if this
# were ever scaled to multiple worker processes, this would need to move
# to Redis or a database instead of an in-memory dict.
# Structure: {chat_id: {"collection": ..., "chunk_count": int, "history": [...], "awaiting_paste": bool}}
user_sessions: dict[int, dict] = {}
question_timestamps: dict[int, deque] = defaultdict(deque)
upload_timestamps: dict[int, deque] = defaultdict(deque)

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle unexpected errors without exposing technical details to users."""

    logger.error(
        "Unhandled exception while processing update",
        exc_info=context.error,
    )

    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=(
                    "❌ Something went wrong while processing your request. "
                    "Please try again."
                ),
            )
        except Exception:
            logger.exception(
                "Failed to send error message to user."
            )


def _get_session(chat_id: int) -> dict:
    if chat_id not in user_sessions:
        saved = get_session(chat_id)

        if saved:
            if saved["collection_name"]:
                try:
                    collection = get_collection(saved["collection_name"])

                    user_sessions[chat_id] = {
                        "collection": collection,
                        "chunk_count": saved["chunk_count"],
                        "history": get_history(chat_id),
                        "awaiting_paste": saved["awaiting_paste"],
                    }

                except Exception as e:
                    logger.error(
                        "Failed to restore collection for chat %s: %s",
                        chat_id,
                        e,
                    )

                    # NOTE: preserve awaiting_paste even if the collection
                    # itself failed to restore — a failed/missing document
                    # is unrelated to whether the user is mid-/paste.
                    user_sessions[chat_id] = {
                        "collection": None,
                        "chunk_count": 0,
                        "history": get_history(chat_id),
                        "awaiting_paste": saved["awaiting_paste"],
                    }
            else:
                # No document saved yet (e.g. user ran /paste but hasn't
                # sent the text yet) — nothing to restore from Chroma,
                # but awaiting_paste still needs to come from the DB.
                user_sessions[chat_id] = {
                    "collection": None,
                    "chunk_count": 0,
                    "history": get_history(chat_id),
                    "awaiting_paste": saved["awaiting_paste"],
                }

        else:
            user_sessions[chat_id] = {
                "collection": None,
                "chunk_count": 0,
                "history": [],
                "awaiting_paste": False,
            }

    return user_sessions[chat_id]

def check_question_rate_limit(chat_id: int) -> bool:
    now = time.monotonic()
    timestamps = question_timestamps[chat_id]

    while timestamps and now - timestamps[0] > 60:
        timestamps.popleft()

    if len(timestamps) >= MAX_QUESTIONS_PER_MINUTE:
        return False

    timestamps.append(now)
    return True

def check_upload_rate_limit(chat_id: int) -> bool:
    now = time.monotonic()
    timestamps = upload_timestamps[chat_id]

    while timestamps and now - timestamps[0] > 600:
        timestamps.popleft()

    if len(timestamps) >= MAX_UPLOADS_PER_10_MINUTES:
        return False

    timestamps.append(now)
    return True
 
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Hi! I answer questions about a document you give me.\n\n"
        "Send me a PDF, or use /paste to give me plain text instead.\n"
        "Once I have a document, just type your question as a normal message.\n\n"
        "Commands:\n"
        "/paste — I'll wait for you to paste text as the next message\n"
        "/new — clear the current document and start over"
    )

async def new_document(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    chat_id = update.effective_chat.id

    saved = get_session(chat_id)

    if saved and saved["collection_name"]:
        try:
            delete_collection(saved["collection_name"])
        except Exception as e:
            logger.warning(
                "Could not delete collection %s: %s",
                saved["collection_name"],
                e,
            )

    delete_session(chat_id)

    user_sessions[chat_id] = {
        "collection": None,
        "chunk_count": 0,
        "history": [],
        "awaiting_paste": False,
    }

    await update.message.reply_text(
        "🗑️ Cleared. Send a new PDF, or use /paste for text."
    )


async def paste_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    chat_id = update.effective_chat.id

    update_activity(chat_id)

    session = _get_session(chat_id)

    session["awaiting_paste"] = True
    set_awaiting_paste(chat_id, True)  # persist — /paste and the pasted
    # text arrive as two separate messages/updates, possibly handled by
    # two different process instances if the bot restarts in between
    # (e.g. a redeploy). Without this, the in-memory flag alone is lost
    # and the bot "forgets" it asked for pasted text.

    await update.message.reply_text(
        "📋 Send me the text now — "
        "I'll use your next message as the document."
    )


async def handle_document(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    chat_id = update.effective_chat.id
    document = update.message.document

    if not check_upload_rate_limit(chat_id):
        await update.message.reply_text(
            "⚠️ Upload limit reached. "
            "Please wait before uploading another document."
        )
        return

    if document.mime_type != "application/pdf":
        await update.message.reply_text(
            "⚠️ I can only read PDF files right now. "
            "Send a .pdf, or use /paste for plain text."
        )
        return

    if document.file_size and document.file_size > MAX_PDF_SIZE:
        await update.message.reply_text(
            "⚠️ This PDF is too large. "
            "The maximum allowed size is 10 MB."
        )
        return

    await update.message.reply_text(
        "📥 Reading your PDF..."
    )

    try:
        telegram_file = await context.bot.get_file(
            document.file_id
        )

        file_bytes = bytes(
            await telegram_file.download_as_bytearray()
        )

        raw_text = extract_text_from_pdf(
            file_bytes
        )

        if not raw_text.strip():
            await update.message.reply_text(
                "❌ I couldn't extract any text from this PDF. "
                "It may be a scanned/image-only document."
            )
            return

        await _process_and_confirm(
            update,
            chat_id,
            raw_text,
        )

    except Exception:
        logger.exception(
            "PDF processing failed for chat_id=%s",
            chat_id,
        )

        await update.message.reply_text(
            "❌ I couldn't process this PDF. "
            "Please try another PDF."
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    session = _get_session(chat_id)
    text = update.message.text

    # Case 1: we're expecting pasted document text (after /paste)
    if session["awaiting_paste"]:
        session["awaiting_paste"] = False

        if len(text) > MAX_PASTE_LENGTH:
            set_awaiting_paste(chat_id, False)  # clear in DB too, not just
            # in-memory — otherwise a restart before the next message
            # leaves the DB thinking we're still waiting for pasted text.
            await update.message.reply_text(
                "⚠️ The text is too long. "
                "The maximum allowed length is 100,000 characters."
            )
            return

        await _process_and_confirm(update, chat_id, text)
        return

    # Case 2: no document loaded yet — this text is not a question we can answer
    if session["collection"] is None:
        await update.message.reply_text(
            "I don't have a document yet. Send me a PDF, or use /paste to give me text."
        )
        return

    # Case 3: normal question against the loaded document

    if not check_question_rate_limit(chat_id):
        await update.message.reply_text(
            "⚠️ You're sending questions too quickly. "
            "Please wait a little and try again."
        )
        return

    update_activity(chat_id)

    await context.bot.send_chat_action(
        chat_id=chat_id,
        action="typing",
    )

    try:
        result = answer_question(
            session["collection"],
            text,
            chat_history=session["history"],
        )

    except Exception:
        logger.exception(
            "Question processing failed for chat_id=%s",
            chat_id,
        )

        await update.message.reply_text(
            "❌ I couldn't answer that question. "
            "Please try again."
        )
        return

    session["history"].append(
    {
        "role": "user",
        "content": text,
    }
)

    save_message(
        chat_id,
        "user",
        text,
    )

    session["history"].append(
        {
            "role": "assistant",
            "content": result["answer"],
        }
    )

    save_message(
        chat_id,
        "assistant",
        result["answer"],
    )

    await update.message.reply_text(
        result["answer"]
    )

async def _process_and_confirm(
    update: Update,
    chat_id: int,
    raw_text: str,
) -> None:

    if len(raw_text) > MAX_PASTE_LENGTH:
        await update.message.reply_text(
            "⚠️ The document text is too large to process."
        )
        return

    try:
        # ------------------------------------------------
        # 1. Get current session
        # ------------------------------------------------

        old_session = get_session(chat_id)

        old_collection_name = None

        if old_session:
            old_collection_name = old_session.get(
                "collection_name"
            )

        # ------------------------------------------------
        # 2. Chunk the NEW document
        # ------------------------------------------------

        chunks = chunk_text(raw_text)

        if not chunks:
            await update.message.reply_text(
                "❌ I couldn't create any chunks "
                "from this document."
            )
            return

        # ------------------------------------------------
        # 3. Build NEW collection FIRST
        # ------------------------------------------------
        #
        # Important:
        # We do NOT delete the old collection yet.
        #

        new_collection = build_collection(chunks)

        # ------------------------------------------------
        # 4. Save NEW session
        # ------------------------------------------------

        save_session(
            chat_id=chat_id,
            collection_name=new_collection.name,
            chunk_count=len(chunks),
        )

        # ------------------------------------------------
        # 5. Delete OLD collection
        # ------------------------------------------------

        if (
            old_collection_name
            and old_collection_name != new_collection.name
        ):
            try:
                delete_collection(
                    old_collection_name
                )

            except Exception:
                logger.exception(
                    "Failed to delete old collection "
                    "%s for chat_id=%s",
                    old_collection_name,
                    chat_id,
                )

        # ------------------------------------------------
        # 6. Clear old conversation history
        # ------------------------------------------------

        clear_history(chat_id)

        # ------------------------------------------------
        # 7. Update RAM session
        # ------------------------------------------------

        session = _get_session(chat_id)

        session["collection"] = new_collection
        session["chunk_count"] = len(chunks)
        session["history"] = []

        # ------------------------------------------------
        # 8. Confirmation
        # ------------------------------------------------

        chunk_word = (
            "chunk"
            if len(chunks) == 1
            else "chunks"
        )

        await update.message.reply_text(
            f"✅ Document ready — indexed into "
            f"{len(chunks)} {chunk_word}. "
            "Ask me anything about it."
        )

    except Exception:
        logger.exception(
            "Document replacement failed "
            "for chat_id=%s",
            chat_id,
        )

        await update.message.reply_text(
            "❌ I couldn't process this document. "
            "Your previous document is still available."
        )

async def cleanup_expired_sessions(
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    expired_sessions = get_expired_sessions(days=30)

    if not expired_sessions:
        return

    for session in expired_sessions:

        chat_id = session["chat_id"]
        collection_name = session["collection_name"]

        try:
            # Delete Chroma collection
            if collection_name:
                try:
                    delete_collection(collection_name)
                except Exception:
                    logger.exception(
                        "Failed to delete collection %s "
                        "for chat %s",
                        collection_name,
                        chat_id,
                    )

            # Delete SQLite session + history
            delete_session(chat_id)

            # Remove in-memory session if it exists
            user_sessions.pop(chat_id, None)

            logger.info(
                "Expired session cleaned: chat_id=%s",
                chat_id,
            )

        except Exception:
            logger.exception(
                "Cleanup failed for chat_id=%s",
                chat_id,
            )

def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")

    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not found — "
            "add it to your .env file."
        )

    init_db()

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("new", new_document))
    app.add_handler(CommandHandler("paste", paste_command))

    app.add_handler(
        MessageHandler(
            filters.Document.PDF,
            handle_document,
        )
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_text,
        )
    )

    # Run cleanup every 24 hours
    app.job_queue.run_repeating(
        cleanup_expired_sessions,
        interval=24 * 60 * 60,
        first=60,
    )

    logger.info("Bot starting (polling)...")

    app.run_polling()


if __name__ == "__main__":
    main()