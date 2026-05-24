"""RAG: загрузка базы знаний, нарезка на чанки, эмбеддинги, поиск top-k."""
from __future__ import annotations

import os
import re
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_TARGET_SIZE = 1000
CHUNK_MAX_SIZE = 1400
CHUNK_OVERLAP = 150
CACHE_FILE = Path("embeddings_cache.npz")


@dataclass
class Chunk:
    idx: int
    title: str
    text: str

    def to_dict(self, similarity: float, rank: int) -> dict:
        return {
            "rank": rank,
            "similarity": round(float(similarity), 4),
            "title": self.title,
            "text": self.text,
        }


def _read_kb(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл базы знаний не найден: {path}")
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="cp1251")


def _split_into_sections(text: str) -> List[tuple[str, str]]:
    """Делит markdown по заголовкам уровня # и ##. Возвращает список (title_path, body)."""
    lines = text.splitlines()
    sections: List[tuple[str, str]] = []
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    buffer: List[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        if not body:
            return
        title_parts = [p for p in (current_h1, current_h2, current_h3) if p]
        title = " / ".join(title_parts) if title_parts else "Без заголовка"
        sections.append((title, body))

    for raw in lines:
        line = raw.rstrip()
        # игнорируем декоративные строки из ###...###
        if re.match(r"^#{4,}\s*$", line) or re.match(r"^#{4,}.*#{4,}\s*$", line):
            continue

        m1 = re.match(r"^#\s+(.+)$", line)
        m2 = re.match(r"^##\s+(.+)$", line)
        m3 = re.match(r"^###\s+(.+)$", line)
        if m1:
            flush()
            buffer = []
            current_h1 = m1.group(1).strip()
            current_h2 = ""
            current_h3 = ""
            continue
        if m2:
            flush()
            buffer = []
            current_h2 = m2.group(1).strip()
            current_h3 = ""
            continue
        if m3:
            flush()
            buffer = []
            current_h3 = m3.group(1).strip()
            continue
        buffer.append(raw)

    flush()
    return sections


def _split_long_text(text: str, target: int, max_size: int, overlap: int) -> List[str]:
    """Делит длинный текст на куски по абзацам с overlap."""
    if len(text) <= max_size:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= target:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
                # overlap: захватываем хвост предыдущего чанка
                tail = current[-overlap:] if overlap > 0 else ""
                current = f"{tail}\n\n{para}" if tail else para
            else:
                current = para
            # если параграф очень длинный — режем грубо
            while len(current) > max_size:
                chunks.append(current[:max_size])
                current = current[max_size - overlap:]
    if current:
        chunks.append(current)
    return chunks


def build_chunks(kb_text: str) -> List[Chunk]:
    sections = _split_into_sections(kb_text)
    chunks: List[Chunk] = []
    idx = 0
    for title, body in sections:
        parts = _split_long_text(body, CHUNK_TARGET_SIZE, CHUNK_MAX_SIZE, CHUNK_OVERLAP)
        for part in parts:
            text_with_title = f"[{title}]\n\n{part.strip()}"
            chunks.append(Chunk(idx=idx, title=title, text=text_with_title))
            idx += 1
    return chunks


def _kb_fingerprint(kb_text: str, n_chunks: int) -> str:
    h = hashlib.sha1()
    h.update(kb_text.encode("utf-8", errors="ignore"))
    h.update(str(n_chunks).encode())
    h.update(EMBEDDING_MODEL_NAME.encode())
    return h.hexdigest()


class RAGIndex:
    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.model: SentenceTransformer | None = None
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.fingerprint: str = ""

    def build(self) -> None:
        logger.info("Чтение базы знаний: %s", self.kb_path)
        kb_text = _read_kb(self.kb_path)

        logger.info("Нарезка на чанки...")
        self.chunks = build_chunks(kb_text)
        logger.info("Получено чанков: %d", len(self.chunks))

        self.fingerprint = _kb_fingerprint(kb_text, len(self.chunks))

        logger.info("Загрузка модели эмбеддингов: %s", EMBEDDING_MODEL_NAME)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

        if self._load_cache():
            logger.info("Эмбеддинги загружены из кэша.")
            return

        logger.info("Вычисление эмбеддингов...")
        texts = [c.text for c in self.chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self.embeddings = embeddings
        self._save_cache()
        logger.info("Эмбеддинги готовы. Shape: %s", embeddings.shape)

    def _load_cache(self) -> bool:
        if not CACHE_FILE.exists():
            return False
        try:
            data = np.load(CACHE_FILE, allow_pickle=False)
            if str(data.get("fingerprint", "")) != self.fingerprint:
                return False
            embs = data["embeddings"]
            if embs.shape[0] != len(self.chunks):
                return False
            self.embeddings = embs.astype("float32")
            return True
        except Exception as e:
            logger.warning("Не удалось загрузить кэш эмбеддингов: %s", e)
            return False

    def _save_cache(self) -> None:
        try:
            np.savez(
                CACHE_FILE,
                embeddings=self.embeddings,
                fingerprint=np.array(self.fingerprint),
            )
        except Exception as e:
            logger.warning("Не удалось сохранить кэш эмбеддингов: %s", e)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if self.model is None or self.embeddings is None:
            raise RuntimeError("Индекс не построен.")
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")[0]
        # эмбеддинги уже нормированы — cosine = dot
        scores = self.embeddings @ q_emb
        top_idx = np.argsort(-scores)[:top_k]
        return [
            self.chunks[i].to_dict(similarity=scores[i], rank=rank + 1)
            for rank, i in enumerate(top_idx)
        ]
