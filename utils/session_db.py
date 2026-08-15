import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
DB_PATH = Path("./bot.db")

MAX_HISTORY_MESSAGES = 20


def init_db() -> None:
    """Create and migrate the database tables."""

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                chat_id INTEGER PRIMARY KEY,
                collection_name TEXT,
                chunk_count INTEGER DEFAULT 0,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Migration for existing databases
        try:
            conn.execute(
                """
                ALTER TABLE sessions
                ADD COLUMN last_activity TIMESTAMP
                """
            )

            # Set the current time for existing sessions
            conn.execute(
                """
                UPDATE sessions
                SET last_activity = CURRENT_TIMESTAMP
                WHERE last_activity IS NULL
                """
            )

        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_chat_id
            ON messages(chat_id)
            """
        )

        conn.commit()


# ============================================================
# Sessions
# ============================================================

def save_session(
    chat_id: int,
    collection_name: str,
    chunk_count: int,
) -> None:
    """Save or update the document collection for a chat."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
        """
        INSERT INTO sessions (
            chat_id,
            collection_name,
            chunk_count,
            last_activity
        )
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)

        ON CONFLICT(chat_id) DO UPDATE SET
            collection_name = excluded.collection_name,
            chunk_count = excluded.chunk_count,
            last_activity = CURRENT_TIMESTAMP
        """,
        (
            chat_id,
            collection_name,
            chunk_count,
        ),
    )

        conn.commit()


def get_session(chat_id: int) -> dict | None:
    """Return the saved session for a chat."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT
                chat_id,
                collection_name,
                chunk_count,
                last_activity
            FROM sessions
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchone()

        if row is None:
            return None

        return dict(row)


def delete_session(chat_id: int) -> None:
    """
    Delete the saved document session and
    its conversation history.
    """

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM sessions WHERE chat_id = ?",
            (chat_id,),
        )

        conn.execute(
            "DELETE FROM messages WHERE chat_id = ?",
            (chat_id,),
        )

        conn.commit()


# ============================================================
# Conversation History
# ============================================================

def save_message(
    chat_id: int,
    role: str,
    content: str,
) -> None:
    """Save one conversation message."""

    if role not in {"user", "assistant"}:
        raise ValueError(
            "role must be 'user' or 'assistant'"
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO messages (
                chat_id,
                role,
                content
            )
            VALUES (?, ?, ?)
            """,
            (
                chat_id,
                role,
                content,
            ),
        )

        # Keep only the latest MAX_HISTORY_MESSAGES
        conn.execute(
            """
            DELETE FROM messages
            WHERE chat_id = ?
              AND id NOT IN (
                  SELECT id
                  FROM messages
                  WHERE chat_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (
                chat_id,
                chat_id,
                MAX_HISTORY_MESSAGES,
            ),
        )

        conn.commit()


def get_history(chat_id: int) -> list[dict]:
    """Return the latest conversation messages for a chat."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT
                role,
                content
            FROM messages
            WHERE chat_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (
                chat_id,
                MAX_HISTORY_MESSAGES,
            ),
        ).fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
            }
            for row in rows
        ]


def clear_history(chat_id: int) -> None:
    """Delete conversation history without deleting the document."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM messages WHERE chat_id = ?",
            (chat_id,),
        )

        conn.commit()

def update_activity(chat_id: int) -> None:
    """Update the last activity timestamp for a chat."""

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE sessions
            SET last_activity = CURRENT_TIMESTAMP
            WHERE chat_id = ?
            """,
            (chat_id,),
        )

        conn.commit()

def get_expired_sessions(days: int = 30) -> list[dict]:
    """Return sessions that have been inactive for at least `days` days."""

    cutoff = datetime.utcnow() - timedelta(days=days)

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT
                chat_id,
                collection_name,
                chunk_count,
                last_activity
            FROM sessions
            WHERE last_activity < ?
            """,
            (cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
        ).fetchall()

        return [dict(row) for row in rows]