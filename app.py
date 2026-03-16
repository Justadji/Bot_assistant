from __future__ import annotations

import os
import uuid
import sqlite3
import hashlib
import shutil
import json
import time
from dataclasses import dataclass
from typing import Dict, Literal, List, Tuple

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

import re

def _log_event(event: str, **payload) -> None:
    line = {"event": event, **payload}
    try:
        print(json.dumps(line, ensure_ascii=False))
    except Exception:
        pass

def fix_latex(text: str) -> str:
    """
    Nettoie et normalise le LaTeX pour Streamlit.
    """
    if not text:
        return text

    # Convert \( ... \) to $...$
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text)

    # Convert \[ ... \] to $$...$$
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)

    return text

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY manquant dans .env")
    st.stop()

# -----------------------------
# Config & Types
# -----------------------------
Mode = Literal["prof", "quiz", "correcteur"]

BASE_DATA_DIR = os.getenv("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
os.makedirs(BASE_DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(BASE_DATA_DIR, "chat_history.sqlite")
SQL_CONN_STR = f"sqlite:///{DB_PATH}"

RAG_DIR = os.path.join(BASE_DATA_DIR, "rag_store")
UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")

os.makedirs(RAG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_MB = 25  # ajuste
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_FULL_MESSAGES = 400  # ~200 tours user/assistant

os.makedirs(RAG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

def trim_full_history(session_id: str, keep: int = MAX_FULL_MESSAGES) -> None:
    conn = _db_connect()
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM message_store_full WHERE session_id=?", (session_id,))
            n = cur.fetchone()[0]
            if n <= keep:
                return
            # on supprime les plus anciens en se basant sur rowid
            to_delete = n - keep
            cur.execute(
                """
                DELETE FROM message_store_full
                WHERE rowid IN (
                    SELECT rowid FROM message_store_full
                    WHERE session_id=?
                    ORDER BY rowid ASC
                    LIMIT ?
                )
                """,
                (session_id, to_delete),
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()

def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.execute("PRAGMA synchronous=NORMAL")
    conn.commit()
    return conn

def _db_checkpoint() -> None:
    """Force un checkpoint WAL pour compacter les ecritures SQLite."""
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    finally:
        conn.close()

SYSTEM_BY_MODE: Dict[Mode, str] = {
    "prof": (
        "Tu es un assistant pedagogique en base de données."
        "Explique clairement, avec un exemple concret quand utile. "
        "Resous les exercice de bases de données, créations d'une base et des requetes."
    ),
    "quiz": (
        "Tu es un examinateur bienveillant en statistiques/econometrie. "
        "Tu poses des questions progressives (1 a la fois), "
        "tu attends la reponse de l'etudiant, puis tu corriges et tu notes sur 20."
    ),
    "correcteur": (
        "Tu es un correcteur rigoureux. "
        "L'etudiant te donne une solution/raisonnement : "
        "tu detectes les erreurs, tu expliques pourquoi, puis tu proposes une correction propre."
    ),
}

@dataclass
class SessionMeta:
    mode: Mode
    summary: str
    rag_enabled: bool
    rag_k: int
    rag_collection: str  # collection name in Chroma (per session)
    rag_ready: bool      # indicates index exists

def _handle_disk_full(e: Exception, context: str = "") -> None:
    if "database or disk is full" in str(e).lower():
        st.error(f"SQLite plein ({context}). Libère de l'espace disque puis relance.")
        st.stop()

# -----------------------------
# Meta DB (mode + summary + rag settings)
# -----------------------------
def _init_meta_db() -> None:
    conn = _db_connect()
    try:
        cur = conn.cursor()

        # 1) creer la table si elle n'existe pas
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS session_meta (
                session_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                summary TEXT NOT NULL,
                rag_enabled INTEGER NOT NULL DEFAULT 1,
                rag_k INTEGER NOT NULL DEFAULT 4,
                rag_collection TEXT NOT NULL,
                rag_ready INTEGER NOT NULL DEFAULT 0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

        # 2) migration: ajouter colonnes si elles manquent (DB ancienne)
        cur.execute("PRAGMA table_info(session_meta)")
        cols = {row[1] for row in cur.fetchall()}  # row[1] = column name

        def add_col(sql: str) -> None:
            cur.execute(sql)
            conn.commit()

        if "rag_enabled" not in cols:
            add_col("ALTER TABLE session_meta ADD COLUMN rag_enabled INTEGER NOT NULL DEFAULT 1")
        if "rag_k" not in cols:
            add_col("ALTER TABLE session_meta ADD COLUMN rag_k INTEGER NOT NULL DEFAULT 4")
        if "rag_collection" not in cols:
            add_col("ALTER TABLE session_meta ADD COLUMN rag_collection TEXT NOT NULL DEFAULT ''")
        if "rag_ready" not in cols:
            add_col("ALTER TABLE session_meta ADD COLUMN rag_ready INTEGER NOT NULL DEFAULT 0")
        if "updated_at" not in cols:
            add_col("ALTER TABLE session_meta ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP")

        # 3) backfill rag_collection pour les anciennes lignes vides
        cur.execute("UPDATE session_meta SET rag_collection = '' WHERE rag_collection IS NULL")
        conn.commit()

    finally:
        conn.close()
def _init_conversations_db() -> None:
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute("PRAGMA table_info(conversations)")
        cols = {row[1] for row in cur.fetchall()}
        if "archived" not in cols:
            cur.execute("ALTER TABLE conversations ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                original_name TEXT NOT NULL,
                saved_path TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_rag_files_session_hash ON rag_files(session_id, file_hash)"
        )
        conn.commit()
    finally:
        conn.close()

def _touch_conversation(session_id: str, title: str | None = None) -> None:
    """Cree la conversation si absente, met a jour updated_at sinon."""
    conn = _db_connect()
    try:
        cur = conn.cursor()
        if title is None:
            title = f"Conversation {session_id[:8]}"
        cur.execute(
            """
            INSERT INTO conversations(session_id, title)
            VALUES(?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at=CURRENT_TIMESTAMP
            """,
            (session_id, title),
        )
        conn.commit()
    finally:
        conn.close()

def _ensure_conversation_exists(session_id: str) -> None:
    """Crée la conversation uniquement si elle n'existe pas (évite les writes à chaque rerun)."""
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM conversations WHERE session_id=? LIMIT 1", (session_id,))
        if cur.fetchone() is None:
            _touch_conversation(session_id)
    finally:
        conn.close()

def _list_conversations(limit: int = 50, include_archived: bool = False, query: str = "") -> list[tuple[str, str, str, int]]:
    """Retourne [(session_id, title, updated_at, archived)] trie du plus recent au plus ancien."""
    conn = _db_connect()
    try:
        cur = conn.cursor()
        where = ["1=1"]
        params: List = []
        if not include_archived:
            where.append("archived=0")
        if query.strip():
            where.append("LOWER(title) LIKE ?")
            params.append(f"%{query.strip().lower()}%")
        params.append(limit)
        cur.execute(
            f"""
            SELECT session_id, title, updated_at, archived
            FROM conversations
            WHERE {' AND '.join(where)}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return cur.fetchall()
    finally:
        conn.close()

def _rename_conversation(session_id: str, title: str) -> None:
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE conversations
            SET title=?, updated_at=CURRENT_TIMESTAMP
            WHERE session_id=?
            """,
            (title, session_id),
        )
        conn.commit()
    finally:
        conn.close()

def _archive_conversation(session_id: str, archived: bool) -> None:
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE conversations SET archived=?, updated_at=CURRENT_TIMESTAMP WHERE session_id=?",
            (1 if archived else 0, session_id),
        )
        conn.commit()
    finally:
        conn.close()

def _duplicate_conversation(session_id: str, new_title: str) -> str:
    new_sid = str(uuid.uuid4())
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversations(session_id, title, archived) VALUES(?, ?, 0)",
            (new_sid, new_title.strip() or f"Copie {session_id[:8]}"),
        )
        cur.execute(
            """
            INSERT INTO session_meta(session_id, mode, summary, rag_enabled, rag_k, rag_collection, rag_ready)
            SELECT ?, mode, summary, rag_enabled, rag_k, ?, rag_ready
            FROM session_meta WHERE session_id=?
            """,
            (new_sid, _default_meta(new_sid).rag_collection, session_id),
        )
        try:
            cur.execute(
                """
                INSERT INTO message_store_ctx(session_id, message)
                SELECT ?, message FROM message_store_ctx WHERE session_id=?
                """,
                (new_sid, session_id),
            )
        except sqlite3.OperationalError:
            pass
        try:
            cur.execute(
                """
                INSERT INTO message_store_full(session_id, message)
                SELECT ?, message FROM message_store_full WHERE session_id=?
                """,
                (new_sid, session_id),
            )
        except sqlite3.OperationalError:
            pass
        cur.execute("SELECT 1 FROM session_meta WHERE session_id=? LIMIT 1", (new_sid,))
        if cur.fetchone() is None:
            meta = _default_meta(new_sid)
            cur.execute(
                """
                INSERT INTO session_meta(session_id, mode, summary, rag_enabled, rag_k, rag_collection, rag_ready)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (new_sid, meta.mode, meta.summary, 1, meta.rag_k, meta.rag_collection, 0),
            )
        conn.commit()
    finally:
        conn.close()
    return new_sid

def _list_rag_files(session_id: str, include_inactive: bool = False) -> List[Tuple[int, str, str, int]]:
    conn = _db_connect()
    try:
        cur = conn.cursor()
        where = "" if include_inactive else "AND active=1"
        cur.execute(
            f"""
            SELECT id, original_name, saved_path, active
            FROM rag_files
            WHERE session_id=? {where}
            ORDER BY indexed_at DESC
            """,
            (session_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()

def _upsert_rag_file(session_id: str, file_hash: str, original_name: str, saved_path: str) -> bool:
    """Retourne True si nouveau fichier, False si deja indexe actif."""
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM rag_files WHERE session_id=? AND file_hash=? AND active=1 LIMIT 1",
            (session_id, file_hash),
        )
        existed_active = cur.fetchone() is not None
        cur.execute(
            """
            INSERT INTO rag_files(session_id, file_hash, original_name, saved_path, active)
            VALUES(?, ?, ?, ?, 1)
            ON CONFLICT(session_id, file_hash) DO UPDATE SET
                original_name=excluded.original_name,
                saved_path=excluded.saved_path,
                active=1,
                indexed_at=CURRENT_TIMESTAMP
            """,
            (session_id, file_hash, original_name, saved_path),
        )
        conn.commit()
        return not existed_active
    finally:
        conn.close()

def _deactivate_rag_files(file_ids: List[int]) -> None:
    if not file_ids:
        return
    conn = _db_connect()
    try:
        cur = conn.cursor()
        marks = ",".join("?" for _ in file_ids)
        cur.execute(f"UPDATE rag_files SET active=0 WHERE id IN ({marks})", tuple(file_ids))
        conn.commit()
    finally:
        conn.close()

def _delete_conversation(session_id: str) -> None:
    """Supprime une conversation (meta, messages, index, listing)."""
    meta = _get_meta(session_id)
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM conversations WHERE session_id=?", (session_id,))
        cur.execute("DELETE FROM session_meta WHERE session_id=?", (session_id,))
        cur.execute("DELETE FROM rag_files WHERE session_id=?", (session_id,))
        # Tables créées automatiquement par SQLChatMessageHistory.
        try:
            cur.execute("DELETE FROM message_store_ctx WHERE session_id=?", (session_id,))
        except sqlite3.OperationalError:
            pass
        try:
            cur.execute("DELETE FROM message_store_full WHERE session_id=?", (session_id,))
        except sqlite3.OperationalError:
            pass
        conn.commit()
    finally:
        conn.close()

    # Nettoie aussi le store RAG de la session.
    try:
        clear_rag_index(session_id, meta)
    except Exception:
        pass
    _db_checkpoint()

def _create_conversation(title: str) -> str:
    """Cree une nouvelle conversation et retourne son session_id."""
    new_id = str(uuid.uuid4())
    safe_title = title.strip() or "Nouvelle conversation"
    _touch_conversation(new_id, title=safe_title)
    return new_id

def _default_meta(session_id: str) -> SessionMeta:
    # Use a stable collection per session
    collection = f"rag_{session_id.replace('-', '')[:16]}"
    return SessionMeta(mode="prof", summary="", rag_enabled=True, rag_k=4, rag_collection=collection, rag_ready=False)

def _get_meta(session_id: str) -> SessionMeta:
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT mode, summary, rag_enabled, rag_k, rag_collection, rag_ready FROM session_meta WHERE session_id=?",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return _default_meta(session_id)

        mode, summary, rag_enabled, rag_k, rag_collection, rag_ready = row
        if mode not in ("prof", "quiz", "correcteur"):
            mode = "prof"
        rag_enabled = bool(rag_enabled)
        rag_ready = bool(rag_ready)
        if not rag_collection:
            rag_collection = _default_meta(session_id).rag_collection
        return SessionMeta(
            mode=mode,
            summary=summary or "",
            rag_enabled=rag_enabled,
            rag_k=int(rag_k),
            rag_collection=str(rag_collection),
            rag_ready=rag_ready,
        )
    finally:
        conn.close()


def _upsert_meta(session_id: str, meta: SessionMeta) -> None:
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO session_meta(session_id, mode, summary, rag_enabled, rag_k, rag_collection, rag_ready)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                mode=excluded.mode,
                summary=excluded.summary,
                rag_enabled=excluded.rag_enabled,
                rag_k=excluded.rag_k,
                rag_collection=excluded.rag_collection,
                rag_ready=excluded.rag_ready,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                session_id,
                meta.mode,
                meta.summary,
                1 if meta.rag_enabled else 0,
                int(meta.rag_k),
                meta.rag_collection,
                1 if meta.rag_ready else 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()

# -----------------------------
# Two histories: CTX (model) + FULL (transcript)
# -----------------------------
def get_ctx_history(session_id: str) -> SQLChatMessageHistory:
    # memoire courte utilisee par le modele (tronquable)
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=SQL_CONN_STR,
        table_name="message_store_ctx",
    )

def get_full_history(session_id: str) -> SQLChatMessageHistory:
    # transcript complet (NE JAMAIS CLEAR)
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=SQL_CONN_STR,
        table_name="message_store_full",
    )

# -----------------------------
# LangChain LLMs
# -----------------------------
DEFAULT_MODEL = "gpt-4o-mini"
llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.3, streaming=True)
summarizer_llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.0, streaming=False)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Prompt includes RAG context
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es un assistant. Tu dois etre fidele aux sources quand elles existent.\n\n"
            "CONTEXTE RESUME (memoire longue) :\n{summary}\n\n"
            "MODE ACTUEL : {mode}\n"
            "{mode_instructions}\n"
            "{extra_instructions}\n\n"
            "{strict_rule}\n\n"
            "CONTEXTE DOCUMENTAIRE (RAG) :\n{context}\n\n"
            "Regles RAG (IMPORTANT):\n"
            "- Le CONTEXTE DOCUMENTAIRE contient des extraits numerotes [1], [2], [3]...\n"
            "- Quand tu utilises une information provenant d'un extrait, ajoute la citation correspondante dans ta phrase, ex: ... [2]\n"
            "- Tu peux citer plusieurs extraits: [1][3]\n"
            "- Si le contexte est vide ou insuffisant, dis-le clairement et reponds sans inventer.\n"
            "- Ne fabrique pas de chiffres, de pages ou de sources.\n"
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

def build_chat_runtime(model: str, temperature: float) -> RunnableWithMessageHistory:
    runtime_llm = ChatOpenAI(model=model, temperature=temperature, streaming=True)
    runtime_chain = prompt | runtime_llm
    return RunnableWithMessageHistory(
        runtime_chain,
        get_ctx_history,
        input_messages_key="input",
        history_messages_key="history",
    )

@st.cache_resource
def get_chat_runtime_cached(model: str, temperature: float) -> RunnableWithMessageHistory:
    return build_chat_runtime(model, temperature)

# -----------------------------
# RAG Helpers (Chroma)
# -----------------------------
def _collection_dir(session_id: str, collection: str) -> str:
    # Persist dir per session+collection
    d = os.path.join(RAG_DIR, f"{collection}")
    os.makedirs(d, exist_ok=True)
    return d

def _is_chroma_schema_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "no such table: tenants" in msg or "chromadb.errors.internalerror" in msg

def _new_rag_collection_name(session_id: str) -> str:
    return f"rag_{session_id.replace('-', '')[:12]}_{uuid.uuid4().hex[:8]}"

def _rotate_rag_collection(session_id: str, meta: SessionMeta, reason: str = "") -> None:
    old_collection = meta.rag_collection
    meta.rag_collection = _new_rag_collection_name(session_id)
    meta.rag_ready = False
    _upsert_meta(session_id, meta)
    _log_event(
        "chroma_rotate_collection",
        session_id=session_id,
        old_collection=old_collection,
        new_collection=meta.rag_collection,
        reason=reason,
    )

def _reset_collection_store(session_id: str, collection: str) -> None:
    persist_dir = _collection_dir(session_id, collection)
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)

def get_vectorstore(session_id: str, collection: str, allow_repair: bool = True) -> Chroma:
    persist_dir = _collection_dir(session_id, collection)
    try:
        return Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    except Exception as e:
        if allow_repair and _is_chroma_schema_error(e):
            _log_event("chroma_repair_init", collection=collection, reason=str(e))
            _reset_collection_store(session_id, collection)
            return get_vectorstore(session_id, collection, allow_repair=False)
        raise

def format_sources(docs: List[Document]) -> List[str]:
    out = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "document")
        page = d.metadata.get("page")
        sheet = d.metadata.get("sheet")
        row_range = d.metadata.get("rows")

        extra = []
        if page is not None:
            extra.append(f"p.{page+1}")
        if sheet:
            extra.append(f"sheet:{sheet}")
        if row_range:
            extra.append(f"rows:{row_range}")

        suffix = f" ({', '.join(extra)})" if extra else ""
        out.append(f"[{i}] {os.path.basename(str(src))}{suffix}")
    return out

def _tokenize_for_rerank(text: str) -> set[str]:
    toks = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
    return {t for t in toks if len(t) >= 2}

def _doc_rerank_score(query: str, doc: Document) -> float:
    q_tokens = _tokenize_for_rerank(query)
    if not q_tokens:
        return 0.0
    content = doc.page_content or ""
    d_tokens = _tokenize_for_rerank(content)
    overlap = len(q_tokens & d_tokens)
    score = float(overlap)

    # boost sur motifs frequents en cours/exos
    q = query.lower()
    c = content.lower()
    ex_match = re.search(r"exercice\s*([0-9]+)", q)
    if ex_match:
        ex_num = ex_match.group(1)
        if re.search(rf"exercice\s*{re.escape(ex_num)}", c):
            score += 5.0
    return score

def build_context_and_docs(session_id: str, meta: SessionMeta, query: str) -> Tuple[str, List[Document]]:
    if not (meta.rag_enabled and meta.rag_ready):
        return "", []

    def _retrieve_once() -> List[Document]:
        vs_local = get_vectorstore(session_id, meta.rag_collection)
        target_k = max(int(meta.rag_k), 1)
        # Recupere large puis rerank lexical pour mieux capter les requetes courtes ("exercice 1", etc.)
        wide_k = max(20, target_k * 6)
        try:
            candidates = vs_local.similarity_search(query, k=wide_k)
        except Exception:
            retriever_local = vs_local.as_retriever(search_kwargs={"k": wide_k})
            candidates = retriever_local.invoke(query)
        if not candidates:
            return []
        ranked = sorted(candidates, key=lambda d: _doc_rerank_score(query, d), reverse=True)
        return ranked[:target_k]

    try:
        docs = _retrieve_once()
    except Exception as e:
        if _is_chroma_schema_error(e):
            _log_event("chroma_repair_retrieve", collection=meta.rag_collection, reason=str(e))
            _reset_collection_store(session_id, meta.rag_collection)
            _rotate_rag_collection(session_id, meta, reason=str(e))
            return "", []
        raise
    if not docs:
        return "", []

    # Contexte numerote pour citations inline
    parts = []
    for idx, d in enumerate(docs, start=1):
        src = os.path.basename(str(d.metadata.get("source", "document")))
        page = d.metadata.get("page")
        sheet = d.metadata.get("sheet")
        rows = d.metadata.get("rows")

        meta_bits = [src]
        if page is not None:
            meta_bits.append(f"p.{page+1}")
        if sheet:
            meta_bits.append(f"sheet:{sheet}")
        if rows:
            meta_bits.append(f"rows:{rows}")

        header = f"[{idx}] " + " | ".join(meta_bits)
        parts.append(f"{header}\n{d.page_content}")

    context = "\n\n---\n\n".join(parts)
    return context, docs

def safe_chunk_text(chunk) -> str:
    if chunk is None:
        return ""
    if hasattr(chunk, "content") and isinstance(chunk.content, str):
        return chunk.content
    if isinstance(chunk, dict):
        c = chunk.get("content")
        return c if isinstance(c, str) else ""
    return ""

# -----------------------------
# Build Index from uploaded files
# -----------------------------
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def save_upload(file) -> str:
    data = file.getvalue()
    h = _hash_bytes(data)
    safe_name = f"{h}_{file.name}"
    path = os.path.join(UPLOAD_DIR, safe_name)
    with open(path, "wb") as f:
        f.write(data)
    return path

def save_upload_with_hash(file) -> Tuple[str, str, str]:
    data = file.getvalue()
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Fichier trop gros: {len(data)/1024/1024:.1f} MB (max {MAX_UPLOAD_MB} MB)")
    data = file.getvalue()
    h = _hash_bytes(data)
    safe_name = f"{h}_{file.name}"
    path = os.path.join(UPLOAD_DIR, safe_name)
    with open(path, "wb") as f:
        f.write(data)
    return path, h, file.name

def load_documents_from_file(path: str) -> List[Document]:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()
        # ensure source metadata is the real file path
        for d in docs:
            d.metadata["source"] = path
        return docs

    if ext in (".csv",):
        df = pd.read_csv(path)
        # Represent rows as text blocks
        docs: List[Document] = []
        chunk_size = 200  # rows per doc block
        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            block = df.iloc[start:end]
            text = block.to_csv(index=False)
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path, "rows": f"{start+1}-{end}"},
                )
            )
        return docs

    if ext in (".xlsx", ".xls"):
        xl = pd.ExcelFile(path)
        docs = []
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            chunk_size = 200
            for start in range(0, len(df), chunk_size):
                end = min(start + chunk_size, len(df))
                block = df.iloc[start:end]
                text = block.to_csv(index=False)
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": path, "sheet": sheet, "rows": f"{start+1}-{end}"},
                    )
                )
        return docs

    # Fallback: treat as plain text if possible
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        return [Document(page_content=txt, metadata={"source": path})]
    except Exception:
        return []

def index_files(session_id: str, meta: SessionMeta, file_paths: List[str]) -> Tuple[int, int]:
    """
    Returns (num_docs_loaded, num_chunks_indexed)
    """
    # 1) Load docs
    docs: List[Document] = []
    for p in file_paths:
        docs.extend(load_documents_from_file(p))

    if not docs:
        return 0, 0

    # 2) Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    MAX_CHUNKS = 6000
    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]

    # 3) Vector store
    try:
        vs = get_vectorstore(session_id, meta.rag_collection)
    except Exception as e:
        if _is_chroma_schema_error(e):
            _log_event("chroma_repair_index_init", collection=meta.rag_collection, reason=str(e))
            _reset_collection_store(session_id, meta.rag_collection)
            _rotate_rag_collection(session_id, meta, reason=str(e))
            vs = get_vectorstore(session_id, meta.rag_collection, allow_repair=False)
        else:
            raise

    # 4) BATCHING
    BATCH_SIZE = 100  # valeur sure (tu peux mettre 200 max)

    try:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            vs.add_documents(batch)
    except Exception as e:
        if _is_chroma_schema_error(e):
            _log_event("chroma_repair_index", collection=meta.rag_collection, reason=str(e))
            _reset_collection_store(session_id, meta.rag_collection)
            _rotate_rag_collection(session_id, meta, reason=str(e))
            vs = get_vectorstore(session_id, meta.rag_collection, allow_repair=False)
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                vs.add_documents(batch)
        else:
            raise

    # persist
    try:
        vs.persist()
    except Exception:
        pass

    return len(docs), len(chunks)
def clear_rag_index(session_id: str, meta: SessionMeta) -> None:
    # Remove persisted vector store directory for this collection
    persist_dir = _collection_dir(session_id, meta.rag_collection)
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)
    # Also reset rag_ready
    meta.rag_ready = False
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE rag_files SET active=0 WHERE session_id=?", (session_id,))
        conn.commit()
    finally:
        conn.close()
    try:
        _upsert_meta(session_id, meta)
    except sqlite3.OperationalError as e:
        if "database or disk is full" in str(e).lower():
            st.error("Impossible d'ecrire dans SQLite (disque plein). Libere de l'espace puis relance.")
            st.stop()
        raise

def rebuild_rag_index_from_active_files(session_id: str, meta: SessionMeta) -> Tuple[int, int]:
    active_files = _list_rag_files(session_id, include_inactive=False)
    file_paths = [row[2] for row in active_files if os.path.exists(row[2])]
    clear_rag_index(session_id, meta)
    if not file_paths:
        return 0, 0
    docs, chunks = index_files(session_id, meta, file_paths)
    meta.rag_ready = chunks > 0
    _upsert_meta(session_id, meta)
    return docs, chunks

def cleanup_storage() -> Tuple[int, int]:
    """
    Supprime les uploads non référencés et compacte la DB.
    Retourne (fichiers_supprimes, bytes_libérés).
    """

    # --- récupérer les fichiers référencés ---
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT saved_path FROM rag_files")
        referenced_paths = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    # --- suppression des fichiers orphelins ---
    removed = 0
    removed_bytes = 0

    for name in os.listdir(UPLOAD_DIR):
        p = os.path.join(UPLOAD_DIR, name)

        if os.path.isfile(p) and p not in referenced_paths:
            try:
                removed_bytes += os.path.getsize(p)
                os.remove(p)
                removed += 1
            except Exception:
                pass

    # --- nettoyage SQLite (important avec WAL) ---
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.commit()
        conn.execute("VACUUM;")
        conn.commit()
    finally:
        conn.close()

    return removed, removed_bytes


# -----------------------------
# Memory long: summarize + trim
# -----------------------------
def update_summary_if_needed(session_id: str, max_messages: int = 16) -> None:
    meta = _get_meta(session_id)
    hist = get_ctx_history(session_id)
    messages = hist.messages

    if len(messages) <= max_messages:
        return

    keep_last = 8
    old = messages[:-keep_last]
    recent = messages[-keep_last:]

    old_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in old])

    summary_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu compresses une conversation en un resume utile et fidele.\n"
                "Objectif: garder faits, objectifs, definitions, decisions, notations, exemples cles.\n"
                "Reponse en francais, style bullet points courts.",
            ),
            (
                "human",
                "Resume existant (peut etre vide):\n{prev}\n\n"
                "Nouveaux messages a integrer:\n{chunk}\n\n"
                "Donne le nouveau resume consolide.",
            ),
        ]
    )
    summary_chain = summary_prompt | summarizer_llm
    new_summary = summary_chain.invoke({"prev": meta.summary, "chunk": old_text}).content

    meta.summary = new_summary
    _upsert_meta(session_id, meta)

    # Trim SQL history
    hist.clear()
    for m in recent:
        hist.add_message(m)

def reset_session(session_id: str) -> None:
    get_ctx_history(session_id).clear()
    get_full_history(session_id).clear()
    meta = _default_meta(session_id)
    _upsert_meta(session_id, meta)
    _touch_conversation(session_id, title=f"Conversation {session_id[:8]}")

def to_markdown_conversation(messages) -> str:
    md_parts = []
    for m in messages:
        role = "User" if m.type == "human" else "Assistant" if m.type == "ai" else m.type
        md_parts.append(f"### {role}\n{m.content}\n")
    return "\n".join(md_parts).strip()


# -----------------------------
# Streamlit UI
# -----------------------------
_init_meta_db()
_init_conversations_db()

st.set_page_config(page_title="Chatbot NJAB", page_icon="📚", layout="centered")
st.title("Chatbot NJAB")
st.caption("Upload PDF/Excel/CSV -> Index -> Chat (modes + memoire longue + streaming + sources)")

# session_id persistent in UI
# -----------------------------
# Session id persistant (URL) -> survit au refresh navigateur
# -----------------------------
qp = st.query_params  # Streamlit >= 1.30

if "sid" in qp and qp["sid"]:
    sid = qp["sid"]
    # query_params peut renvoyer une liste selon versions
    if isinstance(sid, list):
        sid = sid[0]
    st.session_state.session_id = str(sid)
else:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    # Ecrire l'id dans l'URL
    st.query_params["sid"] = st.session_state.session_id

session_id: str = st.session_state.session_id
try:
    _ensure_conversation_exists(session_id)  # assure l'existence sans maj permanente
except sqlite3.OperationalError as e:
    if "database or disk is full" in str(e).lower():
        st.error("SQLite est plein (database or disk is full). Libère de l'espace disque puis relance.")
        st.stop()
    raise

# Sidebar
with st.sidebar:
    st.subheader("Reglages")
    meta = _get_meta(session_id)

    model_name = st.selectbox("Modele", ["gpt-4o-mini", "gpt-4.1-mini"], index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    strict_rag = st.toggle("Mode strict sources", value=False, help="Ne repondre que si les sources suffisent.")

    # Mode
    mode = st.selectbox(
        "Mode",
        ["prof", "quiz", "correcteur"],
        index=["prof", "quiz", "correcteur"].index(meta.mode),
    )
    meta.mode = mode  # type: ignore

    extra = st.text_area(
        "Instructions supplementaires (optionnel)",
        value="",
        height=90,
        placeholder="Ex: Sois tres concis / Utilise des formules / Donne un plan avant de repondre...",
    )

    st.divider()
    st.subheader("Memoire longue (resume)")
    st.text_area("Resume", value=meta.summary or "", height=160, disabled=True)

    st.divider()
    st.subheader("RAG (documents)")
    meta.rag_enabled = st.toggle("Activer RAG", value=meta.rag_enabled)
    meta.rag_k = st.slider("Top-k passages", min_value=2, max_value=10, value=int(meta.rag_k), step=1)

    uploaded = st.file_uploader(
        "Ajoute des fichiers (PDF/CSV/XLSX)",
        type=["pdf", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    selected_upload_indices: List[int] = []
    if uploaded:
        upload_labels = [f.name for f in uploaded]
        selected_labels = st.multiselect(
            "Fichiers a indexer",
            options=upload_labels,
            default=upload_labels,
            help="Selectionne les fichiers a inclure dans l'index.",
        )
        selected_upload_indices = [i for i, f in enumerate(uploaded) if f.name in selected_labels]

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Indexer", use_container_width=True):
            if not uploaded:
                st.warning("Ajoute au moins un fichier avant d'indexer.")
            elif not selected_upload_indices:
                st.warning("Selectionne au moins un fichier a indexer.")
            else:
                files_to_index = [uploaded[i] for i in selected_upload_indices]
                paths: List[str] = []
                skipped: List[str] = []
                for f in files_to_index:
                    try:
                        path, file_hash, original_name = save_upload_with_hash(f)
                    except ValueError as e:
                        st.error(str(e))
                        continue
                    is_new = _upsert_rag_file(session_id, file_hash, original_name, path)
                    if is_new:
                        paths.append(path)
                    else:
                        skipped.append(original_name)
                with st.spinner("Indexation en cours..."):
                    try:
                        n_docs, n_chunks = index_files(session_id, meta, paths) if paths else (0, 0)
                    except Exception as e:
                        if _is_chroma_schema_error(e):
                            _reset_collection_store(session_id, meta.rag_collection)
                            n_docs, n_chunks = index_files(session_id, meta, paths) if paths else (0, 0)
                        else:
                            raise
                if n_chunks > 0:
                    meta.rag_ready = True
                    _upsert_meta(session_id, meta)
                    _db_checkpoint()
                    st.success(f"Index OK. Docs: {n_docs} | Chunks: {n_chunks}")
                elif skipped:
                    st.info("Aucun nouveau fichier a indexer (doublons ignores).")
                else:
                    st.error("Aucun document lisible trouve.")
                if skipped:
                    st.caption("Doublons ignores: " + ", ".join(skipped[:5]))
            

    with col2:
        if st.button("Clear index", use_container_width=True):
            clear_rag_index(session_id, meta)
            st.success("Index RAG efface.")
            st.rerun()

    st.caption(f"Index pret: {'OK' if meta.rag_ready else 'NON'} | Collection: {meta.rag_collection}")
    indexed_files = _list_rag_files(session_id, include_inactive=False)
    if indexed_files:
        indexed_labels = {str(fid): f"{name} (id:{fid})" for fid, name, _, _ in indexed_files}
        to_remove = st.multiselect(
            "Documents indexes",
            options=list(indexed_labels.keys()),
            format_func=lambda x: indexed_labels[x],
            help="Supprime les documents selectionnes de l'index puis reconstruit.",
        )
        if st.button("Supprimer documents selectionnes", use_container_width=True):
            ids = [int(x) for x in to_remove]
            _deactivate_rag_files(ids)
            with st.spinner("Reconstruction de l'index..."):
                docs, chunks = rebuild_rag_index_from_active_files(session_id, meta)
            st.success(f"Index reconstruit. Docs: {docs} | Chunks: {chunks}")
            st.rerun()
    if st.button("Nettoyage stockage", use_container_width=True):
        removed, bytes_removed = cleanup_storage()
        _db_checkpoint()
        st.success(f"Nettoyage termine. Fichiers supprimes: {removed}, espace libere: {bytes_removed} bytes")
    
    if st.button("Optimiser index (rebuild)", use_container_width=True):
        with st.spinner("Optimisation..."):
            docs, chunks = rebuild_rag_index_from_active_files(session_id, meta)
        st.success(f"Optimisation OK. Docs: {docs} | Chunks: {chunks}")
        _db_checkpoint()
        st.rerun()

    st.divider()
    st.subheader("Export")
    hist_msgs = get_full_history(session_id).messages
    md_text = to_markdown_conversation(hist_msgs)
    st.download_button(
        "Telecharger resume (.txt)",
        data=(meta.summary or ""),
        file_name="resume_chat.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.download_button(
        "Telecharger conversation (.md)",
        data=md_text,
        file_name="conversation.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Conversations")
    query_conv = st.text_input("Rechercher", value="", placeholder="Titre...")
    show_archived = st.toggle("Afficher archivees", value=False)

    convs = _list_conversations(limit=100, include_archived=show_archived, query=query_conv)
    sid_options = [sid for (sid, _, _, _) in convs]
    sid_to_label = {
        sid: (f"{title} (archivee)" if archived else title)
        for (sid, title, updated_at, archived) in convs
    }
    sid_to_title = {sid: title for (sid, title, _, _) in convs}
    sid_to_archived = {sid: bool(archived) for (sid, _, _, archived) in convs}

    if not sid_options:
        sid_options = [session_id]
        sid_to_label = {session_id: f"Conversation {session_id[:8]}"}
        sid_to_title = {session_id: f"Conversation {session_id[:8]}"}

    current_idx = sid_options.index(session_id) if session_id in sid_options else 0
    selected_sid = st.selectbox(
        "Reprendre une conversation",
        options=sid_options,
        index=current_idx,
        format_func=lambda sid: sid_to_label.get(sid, sid),
        key="conversation_selector",
    )

    if selected_sid != session_id:
        new_sid = selected_sid
        st.session_state.session_id = new_sid
        st.query_params["sid"] = new_sid
        st.rerun()

    cconv1, cconv2 = st.columns(2)
    with cconv1:
        if st.button("Dupliquer", use_container_width=True):
            dup_sid = _duplicate_conversation(selected_sid, f"Copie - {sid_to_title.get(selected_sid, 'Conversation')}")
            st.session_state.session_id = dup_sid
            st.query_params["sid"] = dup_sid
            st.success("Conversation dupliquee.")
            st.rerun()
    with cconv2:
        arch_label = "Desarchiver" if sid_to_archived.get(selected_sid, False) else "Archiver"
        if st.button(arch_label, use_container_width=True):
            _archive_conversation(selected_sid, not sid_to_archived.get(selected_sid, False))
            if selected_sid == session_id and not show_archived and not sid_to_archived.get(selected_sid, False):
                next_convs = _list_conversations(limit=100, include_archived=False)
                if next_convs:
                    st.session_state.session_id = next_convs[0][0]
                    st.query_params["sid"] = next_convs[0][0]
            st.rerun()

    confirm_key = "confirm_delete_sid"
    if st.button("Supprimer la conversation", use_container_width=True):
        st.session_state[confirm_key] = selected_sid

    if st.session_state.get(confirm_key) == selected_sid:
        st.warning("Confirmer la suppression definitive de cette conversation ?")
        cdel1, cdel2 = st.columns(2)
        with cdel1:
            if st.button("Confirmer suppression", use_container_width=True):
                sid_to_delete = selected_sid
                _delete_conversation(sid_to_delete)
                st.session_state.pop(confirm_key, None)

                if sid_to_delete == session_id:
                    remaining = [sid for sid in sid_options if sid != sid_to_delete]
                    if remaining:
                        next_sid = remaining[0]
                    else:
                        next_sid = str(uuid.uuid4())
                        _ensure_conversation_exists(next_sid)
                    st.session_state.session_id = next_sid
                    st.query_params["sid"] = next_sid
                st.rerun()
        with cdel2:
            if st.button("Annuler", use_container_width=True):
                st.session_state.pop(confirm_key, None)
                st.rerun()

    if st.button("Nouvelle conversation", use_container_width=True):
        st.session_state["show_new_conv_dialog"] = True
        st.session_state["new_conv_title"] = ""
        st.rerun()

    st.caption(f"Session: {session_id[:8]}...")
    meta_fingerprint = (meta.mode, meta.rag_enabled, int(meta.rag_k), meta.rag_collection, meta.rag_ready, meta.summary)
    prev = st.session_state.get("meta_fp")
    if prev != meta_fingerprint:
        st.session_state["meta_fp"] = meta_fingerprint
        
    _upsert_meta(session_id, meta)

# Popup: creation d'une nouvelle conversation avec titre
if st.session_state.get("show_new_conv_dialog"):
    if hasattr(st, "dialog"):
        @st.dialog("Nouvelle conversation")
        def _new_conversation_dialog() -> None:
            title = st.text_input(
                "Titre de la conversation",
                value=st.session_state.get("new_conv_title", ""),
                key="new_conv_title_input",
                placeholder="Ex: Revision chapitre 3",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Creer", use_container_width=True, key="create_new_conv_btn"):
                    new_id = _create_conversation(title)
                    st.session_state.session_id = new_id
                    st.query_params["sid"] = new_id
                    st.session_state.pop("show_new_conv_dialog", None)
                    st.session_state.pop("new_conv_title", None)
                    st.session_state.pop("new_conv_title_input", None)
                    st.rerun()
            with c2:
                if st.button("Annuler", use_container_width=True, key="cancel_new_conv_btn"):
                    st.session_state.pop("show_new_conv_dialog", None)
                    st.session_state.pop("new_conv_title", None)
                    st.session_state.pop("new_conv_title_input", None)
                    st.rerun()

        _new_conversation_dialog()
    else:
        # Fallback si st.dialog n'est pas disponible.
        with st.sidebar:
            st.warning("Popup non disponible sur cette version Streamlit.")
            title = st.text_input(
                "Titre de la nouvelle conversation",
                value=st.session_state.get("new_conv_title", ""),
                key="new_conv_title_fallback",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Creer", use_container_width=True, key="create_new_conv_btn_fallback"):
                    new_id = _create_conversation(title)
                    st.session_state.session_id = new_id
                    st.query_params["sid"] = new_id
                    st.session_state.pop("show_new_conv_dialog", None)
                    st.session_state.pop("new_conv_title", None)
                    st.session_state.pop("new_conv_title_fallback", None)
                    st.rerun()
            with c2:
                if st.button("Annuler", use_container_width=True, key="cancel_new_conv_btn_fallback"):
                    st.session_state.pop("show_new_conv_dialog", None)
                    st.session_state.pop("new_conv_title", None)
                    st.session_state.pop("new_conv_title_fallback", None)
                    st.rerun()

# Display transcript from SQLite (source of truth)
history_messages = get_full_history(session_id).messages
for m in history_messages:
    role = "user" if m.type == "human" else "assistant" if m.type == "ai" else "assistant"
    with st.chat_message(role):
        st.markdown(fix_latex(m.content))

# Chat input
user_text = st.chat_input("Pose ta question... (RAG utilisera tes documents si indexes)")
if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)

    # summarize if needed
    update_summary_if_needed(session_id)

    meta = _get_meta(session_id)
    mode_instructions = SYSTEM_BY_MODE[meta.mode]
    extra_instructions = ""
    if extra.strip():
        extra_instructions = "\nINSTRUCTIONS SUPPLEMENTAIRES:\n" + extra.strip()
    strict_rule = ""
    if strict_rag:
        strict_rule = (
            "MODE STRICT SOURCE ACTIVE:\n"
            "- Si le CONTEXTE DOCUMENTAIRE ne suffit pas, reponds uniquement: "
            "\"Je ne peux pas repondre de facon fiable avec les sources disponibles.\""
        )

    # RAG retrieve
    context, docs = build_context_and_docs(session_id, meta, user_text)
    sources = format_sources(docs) if docs else []
    chat = get_chat_runtime_cached(model_name, float(temperature))
    started = time.time()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        acc = ""

        with st.spinner("Reflexion..."):
            for chunk in chat.stream(
                {
                    "input": user_text,
                    "summary": meta.summary,
                    "mode": meta.mode,
                    "mode_instructions": mode_instructions,
                    "extra_instructions": extra_instructions,
                    "strict_rule": strict_rule,
                    "context": context,
                },
                config={"configurable": {"session_id": session_id}},
            ):
                text = safe_chunk_text(chunk)
                if text:
                    acc += text
                    placeholder.markdown(fix_latex(acc))

        if not acc.strip():
            placeholder.markdown("(Reponse vide)")

        if sources:
            with st.expander("Sources utilisees"):
                for s in sources:
                    st.write(s)
        elif meta.rag_enabled and meta.rag_ready:
            st.caption("RAG: aucun passage pertinent trouve. Reformule en citant un mot cle (ex: 'exercice 1').")
        elif strict_rag:
            st.warning("Aucune source disponible en mode strict.")

    update_summary_if_needed(session_id)

    # Sauvegarde transcript complet
    try:
        full = get_full_history(session_id)
        full.add_user_message(user_text)
        full.add_ai_message(acc)
        trim_full_history(session_id)
    except sqlite3.OperationalError as e:
        if "database or disk is full" in str(e).lower():
            st.error("Historique non enregistre: SQLite plein. Libere de l'espace disque.")
        else:
            raise

    # met a jour la liste des conversations
    try:
        _touch_conversation(session_id)
    except sqlite3.OperationalError:
        pass

    _log_event(
        "chat_turn",
        session_id=session_id,
        mode=meta.mode,
        model=model_name,
        strict=strict_rag,
        rag_sources=len(sources),
        latency_ms=int((time.time() - started) * 1000),
    )

    st.rerun()
