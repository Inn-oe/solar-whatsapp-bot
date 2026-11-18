"""Large-language-model fallback for messages the quick model is unsure about.

The intent classifier (nlu.py) is fast and runs on the bot's small server, but it only knows phrasings close to
its examples. When it is unsure, this asks an instruction-tuned LLM (Mistral, Qwen, Llama, ...) through any
OpenAI-compatible chat API to pick one of the bot's own intents. The LLM never writes the reply to the customer:
it only chooses the intent, so the answers stay the ones staff wrote on the Train page.

Environment variables (all optional; without LLM_API_KEY the fallback is off):
  LLM_API_KEY    API key of the provider
  LLM_API_URL    OpenAI-compatible base URL (default https://api.mistral.ai/v1)
                   Groq:        https://api.groq.com/openai/v1
                   OpenRouter:  https://openrouter.ai/api/v1
  LLM_MODEL      model name (default mistral-small-latest)
                   e.g. open-mistral-nemo, ministral-8b-latest, qwen/qwen-2.5-7b-instruct (OpenRouter),
                   qwen/qwen3-32b or llama-3.1-8b-instant (Groq)
  LLM_TIMEOUT    seconds to wait (default 8)
"""
import json
import logging
import os
import re
import threading

log = logging.getLogger("solar-bot.llm")

_cache = {}
_cache_lock = threading.Lock()
_menu = []          # [(intent, category, [example phrases])], refreshed after every training run
_status = {"calls": 0, "errors": 0, "last_error": ""}


def enabled():
    return bool(os.getenv("LLM_API_KEY"))


def model_name():
    return os.getenv("LLM_MODEL", "mistral-small-latest")


def status():
    return dict(_status, enabled=enabled(), model=model_name() if enabled() else None)


def set_menu(intents, examples, per_intent=4):
    """intents: name -> row (category, enabled); examples: [(intent, text)]. Keeps a few phrases per intent."""
    by = {}
    for intent, text in examples:
        lst = by.setdefault(intent, [])
        if len(lst) < per_intent and text not in lst:
            lst.append(text)
    global _menu
    _menu = [(n, (row or {}).get("category", "enquiry"), by.get(n, []))
             for n, row in sorted(intents.items()) if (row or {}).get("enabled", True) and by.get(n)]
    with _cache_lock:
        _cache.clear()


SYSTEM = (
    "You sort WhatsApp messages sent to a solar energy company (Zimbabwe; customers may mix English and Shona, "
    "write in short form, or make spelling mistakes). Pick the ONE intent from the list that best matches what the "
    "customer wants. Use the assistant's previous message to understand short answers: for example, after "
    "'are you looking for solar for your home, or help with a system you already have?', the reply 'no I need new "
    "system' means they want to buy a new system. If the message matches none of the intents, or is gibberish, "
    "answer none. Reply with JSON only: {\"intent\": \"<intent name or none>\", \"confidence\": <0 to 1>}"
)


def _prompt(text, context):
    lines = [f"- {name} ({cat}): e.g. " + " | ".join(f'"{e}"' for e in ex) for name, cat, ex in _menu]
    parts = ["Intents:", *lines, ""]
    if context:
        parts.append(f"Assistant's previous message: \"{context[:400]}\"")
    parts.append(f"Customer's message: \"{text[:500]}\"")
    return "\n".join(parts)


def classify(text, context=""):
    """-> (intent or None, confidence). Never raises; returns (None, 0.0) when disabled or on any error."""
    if not enabled() or not _menu or not (text or "").strip():
        return None, 0.0
    key = (text.strip().lower(), (context or "")[:200])
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    import requests
    url = os.getenv("LLM_API_URL", "https://api.mistral.ai/v1").rstrip("/") + "/chat/completions"
    body = {"model": model_name(), "temperature": 0, "max_tokens": 60,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _prompt(text, context)}]}
    _status["calls"] += 1
    try:
        r = requests.post(url, json=body, timeout=float(os.getenv("LLM_TIMEOUT", "8")),
                          headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY')}"})
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"] or ""
        result = _parse(content)
    except Exception as exc:
        _status["errors"] += 1
        _status["last_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        log.warning("LLM fallback failed: %s", _status["last_error"])
        return None, 0.0
    with _cache_lock:
        if len(_cache) > 2000:
            _cache.clear()
        _cache[key] = result
    return result


def _parse(content):
    """The model's JSON answer -> (known intent or None, confidence)."""
    m = re.search(r"\{.*\}", content, re.S)
    try:
        data = json.loads(m.group(0) if m else content)
    except ValueError:
        return None, 0.0
    intent = str(data.get("intent") or "").strip()
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    known = {n for n, _, _ in _menu}
    return (intent, conf) if intent in known else (None, 0.0)
