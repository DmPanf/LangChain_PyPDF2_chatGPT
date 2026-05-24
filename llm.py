"""LLM-клиент с перебором нескольких провайдеров: OpenAI -> Qwen -> OpenRouter (free)."""
from __future__ import annotations

import asyncio
import os
import logging
from dataclasses import dataclass
from typing import List

import httpx

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 25.0       # таймаут на один запрос к модели
PAUSE_BETWEEN_MODELS = 0.8   # пауза между попытками разных моделей

OPENROUTER_FREE_MODELS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1:free",
    "qwen/qwen3-235b-a22b:free",
    "qwen/qwen3-30b-a3b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
]

SYSTEM_PROMPT = """Ты — AI-продажник компании «ЭкоДомСтрой», которая строит быстровозводимые загородные дома: каркасные, модульные, SIP-дома, дачные дома и дома для постоянного проживания.

Твоя задача:
- помочь клиенту разобраться;
- подобрать подходящее решение;
- отвечать ТОЛЬКО на основе предоставленного контекста из базы знаний;
- не выдумывать факты, цены, сроки, которых нет в контексте;
- если точной информации в контексте нет — честно сказать об этом;
- отвечать простым русским языком, кратко и по делу;
- быть дружелюбным и уверенным;
- после каждого ответа задавать один проактивный следующий вопрос, который ведёт к продаже;
- если клиент явно заинтересован — мягко предложить оставить имя, e-mail и телефон в форме справа."""


@dataclass
class LLMResult:
    answer: str
    model_used: str
    error: str | None = None


class LLMConfigError(Exception):
    """4xx ошибка — проблема в конфиге (ключ, модель), повторять бесполезно."""


class LLMTransientError(Exception):
    """429/5xx/таймаут — можно перейти к следующей модели/провайдеру."""


def _build_messages(question: str, chunks: List[dict], history: List[dict]) -> List[dict]:
    context = "\n\n---\n\n".join(
        f"Чанк {c['rank']} (similarity {c['similarity']}):\n{c['text']}" for c in chunks
    )
    user_content = (
        f"Контекст из базы знаний:\n{context}\n\n"
        f"Вопрос клиента: {question}\n\n"
        "Ответь полезно, кратко, по делу и задай один следующий проактивный вопрос."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[-8:]:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})
    return messages


async def _call_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[dict],
    extra_headers: dict | None = None,
    timeout: float = REQUEST_TIMEOUT,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 800,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as e:
        raise LLMTransientError(f"timeout {timeout}s: {e}") from e
    except httpx.HTTPError as e:
        raise LLMTransientError(f"network: {type(e).__name__}: {e}") from e

    if r.status_code == 429 or r.status_code >= 500:
        raise LLMTransientError(f"HTTP {r.status_code}: {r.text[:200]}")
    if 400 <= r.status_code < 500:
        # 401 (bad key), 403, 404 (model not found) — это конфиг, дальше пробовать ту же модель бессмысленно
        raise LLMConfigError(f"HTTP {r.status_code}: {r.text[:200]}")

    try:
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, ValueError, TypeError) as e:
        raise LLMTransientError(f"bad response: {e}") from e


async def _try_model(
    label: str,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[dict],
    extra_headers: dict | None = None,
    errors: List[str] | None = None,
) -> str | None:
    """Возвращает ответ или None. При config-ошибке (4xx) — логирует и идёт дальше БЕЗ паузы."""
    try:
        return await _call_openai_compatible(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=messages,
            extra_headers=extra_headers,
        )
    except LLMConfigError as e:
        msg = f"{label} ({model}) config-error: {e}"
        logger.warning(msg)
        if errors is not None:
            errors.append(msg)
    except LLMTransientError as e:
        msg = f"{label} ({model}) transient: {e}"
        logger.warning(msg)
        if errors is not None:
            errors.append(msg)
    return None


async def generate_answer(
    question: str, chunks: List[dict], history: List[dict]
) -> LLMResult:
    messages = _build_messages(question, chunks, history)
    errors: List[str] = []
    first_attempt = True

    async def pause_if_needed():
        nonlocal first_attempt
        if first_attempt:
            first_attempt = False
        else:
            await asyncio.sleep(PAUSE_BETWEEN_MODELS)

    # 1) OpenAI
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        await pause_if_needed()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        answer = await _try_model(
            "OpenAI", "https://api.openai.com/v1",
            openai_key, model, messages, errors=errors,
        )
        if answer:
            return LLMResult(answer=answer, model_used=f"openai/{model}")

    # 2) Qwen (DashScope, OpenAI-совместимый)
    qwen_key = os.getenv("QWEN_API_KEY", "").strip()
    if qwen_key:
        await pause_if_needed()
        base = os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ).strip()
        model = os.getenv("QWEN_MODEL", "qwen-plus").strip()
        answer = await _try_model(
            "Qwen", base, qwen_key, model, messages, errors=errors,
        )
        if answer:
            return LLMResult(answer=answer, model_used=f"qwen/{model}")

    # 3) OpenRouter free-модели (перебор)
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if or_key:
        for model in OPENROUTER_FREE_MODELS:
            await pause_if_needed()
            answer = await _try_model(
                "OpenRouter", "https://openrouter.ai/api/v1",
                or_key, model, messages,
                extra_headers={
                    "HTTP-Referer": "https://replit.com",
                    "X-Title": "EcoDomStroy RAG",
                },
                errors=errors,
            )
            if answer:
                return LLMResult(answer=answer, model_used=f"openrouter/{model}")

    if not errors:
        return LLMResult(
            answer="",
            model_used="",
            error="Не настроен ни один LLM-провайдер. Заполните ключи в .env / Secrets.",
        )
    return LLMResult(
        answer="",
        model_used="",
        error="Не удалось получить ответ LLM. Ниже показаны найденные чанки, "
        "чтобы можно было проверить качество поиска.\n\nДетали: "
        + " | ".join(errors[-3:]),
    )
