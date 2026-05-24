"""FastAPI приложение: RAG-чат AI-продажника ЭкоДомСтрой."""
from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from llm import generate_answer
from rag import RAGIndex

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ecodomstroy-rag")

KB_FILE = os.getenv("KB_FILE", "rag_knowledge_base_20260519_185732.txt")
LEADS_FILE = Path("leads.json")
WELCOME_MESSAGE = (
    "Здравствуйте! Я AI-консультант компании по строительству быстровозводимых "
    "домов. Чем могу помочь? Какой дом вы рассматриваете: дачный, модульный, "
    "SIP, каркасный или для постоянного проживания?"
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[ChatMessage] = Field(default_factory=list)


class LeadRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    email: str = Field(..., min_length=3, max_length=200)
    phone: str = Field(..., min_length=3, max_length=40)


index: RAGIndex | None = None
index_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, index_error
    try:
        idx = RAGIndex(KB_FILE)
        idx.build()
        index = idx
        logger.info("RAG-индекс готов. Чанков: %d", len(idx.chunks))
    except FileNotFoundError as e:
        index_error = str(e)
        logger.error("База знаний не найдена: %s", e)
    except Exception as e:
        index_error = f"Ошибка инициализации RAG: {type(e).__name__}: {e}"
        logger.exception("Ошибка инициализации RAG")
    yield


app = FastAPI(title="EcoDomStroy RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "welcome_message": WELCOME_MESSAGE,
            "index_error": index_error,
        },
    )


@app.get("/api/health")
async def health():
    return {
        "ok": index is not None,
        "chunks": len(index.chunks) if index else 0,
        "error": index_error,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    if index is None:
        return JSONResponse(
            status_code=503,
            content={
                "chunks": [],
                "answer": "",
                "llm_model_used": "",
                "llm_error": index_error or "RAG-индекс не готов.",
            },
        )

    try:
        chunks = index.search(req.message, top_k=5)
    except Exception as e:
        logger.exception("Ошибка поиска")
        return JSONResponse(
            status_code=500,
            content={
                "chunks": [],
                "answer": "",
                "llm_model_used": "",
                "llm_error": f"Ошибка поиска: {e}",
            },
        )

    history_dicts = [m.model_dump() for m in req.history]
    result = await generate_answer(req.message, chunks, history_dicts)

    return {
        "chunks": chunks,
        "answer": result.answer,
        "llm_model_used": result.model_used,
        "llm_error": result.error,
    }


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@app.post("/lead")
async def lead(req: LeadRequest):
    if not _EMAIL_RE.match(req.email):
        raise HTTPException(status_code=400, detail="Некорректный e-mail.")
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Имя не должно быть пустым.")
    if not req.phone.strip():
        raise HTTPException(status_code=400, detail="Телефон не должен быть пустым.")

    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "name": req.name.strip(),
        "email": req.email.strip(),
        "phone": req.phone.strip(),
    }
    leads: list = []
    if LEADS_FILE.exists():
        try:
            leads = json.loads(LEADS_FILE.read_text(encoding="utf-8"))
            if not isinstance(leads, list):
                leads = []
        except Exception:
            leads = []
    leads.append(entry)
    LEADS_FILE.write_text(
        json.dumps(leads, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Сохранена заявка: %s", entry["email"])
    lead_id = f"LEAD-{len(leads):05d}"
    return {
        "success": True,
        "message": "Спасибо! Заявка сохранена. Менеджер свяжется с вами.",
        "lead_id": lead_id,
        "ts": entry["ts"],
        "recipient": "Отдел продаж ЭкоДомСтрой",
        "name": entry["name"],
        "email": entry["email"],
        "phone": entry["phone"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=False,
    )
