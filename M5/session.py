"""
M5 — Session Store
PS 26152 — AI-Powered Criminal Network Analysis System

In-memory, single-process session store. Sessions live for the lifetime of
the running process — no database, no cross-restart persistence (by design;
see backend.md §5 and PRD §3).

M5.4 — Session State
"""

import logging
import threading
import uuid

from M5.models import SessionState

logger = logging.getLogger(__name__)

# Thread-safe in-memory store: session_id -> SessionState
_sessions: dict[str, SessionState] = {}
_lock = threading.Lock()


def create_session() -> SessionState:
    """Create a new, empty session and register it in the store."""
    session_id = f"SESS_{uuid.uuid4().hex[:8].upper()}"
    state = SessionState(session_id=session_id)
    with _lock:
        _sessions[session_id] = state
    logger.info(f"[Session] Created session {session_id}")
    return state


def get_session_state(session_id: str) -> SessionState | None:
    """Return the SessionState for an existing session, or None if not found."""
    with _lock:
        return _sessions.get(session_id)


def save_session_state(state: SessionState) -> None:
    """Persist the (modified) state back to the store."""
    with _lock:
        _sessions[state.session_id] = state
    logger.debug(f"[Session] Saved state for {state.session_id}")


def clear_session(session_id: str) -> None:
    """Remove a session from the store (e.g., after expiry or explicit close)."""
    with _lock:
        _sessions.pop(session_id, None)
    logger.info(f"[Session] Cleared session {session_id}")


def get_or_create_session(session_id: str | None) -> SessionState:
    """
    Return an existing session if session_id is provided and found;
    create a fresh one otherwise (handles lost/expired session gracefully,
    per backend.md §8 failure handling rules).
    """
    if session_id:
        state = get_session_state(session_id)
        if state is not None:
            return state
        logger.warning(
            f"[Session] Session {session_id!r} not found — starting a fresh session"
        )
    return create_session()
