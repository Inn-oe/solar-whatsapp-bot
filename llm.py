"""Large-language-model fallback for messages the quick model is unsure about.

The intent classifier (nlu.py) is fast and runs on the bot's small server, but it only knows phrasings close to
its examples. When it is unsure, this asks an instruction-tuned LLM (Mistral, Qwen, Llama, ...) through any
OpenAI-compatible chat API to pick one of the bot's own intents. The LLM never writes the reply to the customer:
it only chooses the intent. With LLM_REWORD=on (default off) it also rewords the staff-written answer naturally for the
conversation (natural_reply) and answers unmatched questions from the company facts only (answer_from_facts); every
number must come from the approved text, otherwise the approved text is sent unchanged.

Environment variables (all optional; without LLM_API_KEY the fallback is off):
  LLM_API_KEY    API key of the provider. A key starting "sk-ant-" is a Claude key: the bot then uses Anthropic's
                 own Messages API with the model claude-haiku-5-5 (set LLM_MODEL to change it, e.g. a Sonnet).
  LLM_PROVIDER   "anthropic" or "openai" (any OpenAI-compatible API); normally detected from the key
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


def provider():
    """'anthropic' for Claude (a key starting sk-ant-, an Anthropic URL, or LLM_PROVIDER=anthropic), else 'openai'
    (any OpenAI-compatible API such as Mistral, Groq or OpenRouter)."""
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    if forced in ("anthropic", "claude"):
        return "anthropic"
    if forced in ("openai", "compatible"):
        return "openai"
    if os.getenv("LLM_API_KEY", "").startswith("sk-ant-") or "anthropic.com" in os.getenv("LLM_API_URL", ""):
        return "anthropic"
    return "openai"


def model_name():
    return os.getenv("LLM_MODEL", "claude-haiku-5-5" if provider() == "anthropic" else "mistral-small-latest")


def status():
    return dict(_status, enabled=enabled(), model=model_name() if enabled() else None,
                provider=provider() if enabled() else None, replies=dict(_compose_status))


ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _anthropic(system, user, max_tokens, temperature, cache_system=False):
    """One call to Claude's Messages API; returns the reply text. cache_system lets Anthropic keep the (long,
    unchanging) system prompt warm between calls, which makes repeat calls cheaper and faster."""
    import requests
    block = {"type": "text", "text": system}
    if cache_system:
        block["cache_control"] = {"type": "ephemeral"}
    body = {"model": model_name(), "max_tokens": max_tokens, "temperature": temperature, "system": [block],
            "messages": [{"role": "user", "content": user}]}
    r = requests.post(ANTHROPIC_URL, json=body, timeout=float(os.getenv("LLM_TIMEOUT", "8")),
                      headers={"x-api-key": os.getenv("LLM_API_KEY", ""), "anthropic-version": "2023-06-01",
                               "content-type": "application/json"})
    if r.status_code >= 400:      # say what Anthropic said (wrong key, unknown model, no credit...)
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:140]}")
    return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()


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


def _menu_text():
    lines = [f"- {name} ({cat}): e.g. " + " | ".join(f'"{e}"' for e in ex) for name, cat, ex in _menu]
    return "\n".join(["Intents:", *lines, ""])


def _message_part(text, context):
    parts = []
    if context:
        parts.append(f"Assistant's previous message: \"{context[:400]}\"")
    parts.append(f"Customer's message: \"{text[:500]}\"")
    return "\n".join(parts)


def _prompt(text, context):
    return _menu_text() + "\n" + _message_part(text, context)


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
        if provider() == "anthropic":
            # the long list of intents is the same every time, so it goes in the cached system prompt
            content = _anthropic(SYSTEM + "\n\n" + _menu_text(), _message_part(text, context), 60, 0.0,
                                 cache_system=True)
        else:
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


# ---------------------------------------------------------------- natural replies
# The training answers say WHAT to tell the customer; the LLM says it the way a person would in this chat. Every
# number in its reply must come from the approved text, so prices and facts cannot drift. Any problem (no key,
# slow, a changed number, too long) and the bot simply sends the approved text.
STYLE = (
    "You are the WhatsApp assistant of {company}, a solar company in Zimbabwe, chatting with a customer. Respond "
    "naturally to the customer's latest message, like a friendly, professional person who remembers the conversation. "
    "Use the APPROVED ANSWER as the source for facts and the suggested next step, not as wording to copy. Be flexible: "
    "acknowledge what the customer said, answer their actual question first, don't repeat details they already gave, "
    "and ask at most one useful follow-up question when needed. Avoid sounding like a form or repeating a greeting. "
    "Keep every fact, price and number exactly as given; add no facts, prices, promises or products. Never weaken or "
    "omit any safety advice or technician handoff in the approved answer. Keep it concise and WhatsApp-friendly, use "
    "the customer's first name at most once, and if they wrote in Shona you may reply in simple Shona mixed with "
    "English. Reply with the message text only."
)
ANSWER = (
    "You are the WhatsApp assistant of {company}, a solar company in Zimbabwe. Answer the customer's question using "
    "ONLY the COMPANY FACTS below, in a short, friendly WhatsApp reply (2-4 sentences). Keep every number exactly as in "
    "the facts and add nothing that is not there. If the facts do not answer the question, reply with exactly: UNKNOWN"
)
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_compose_status = {"calls": 0, "used": 0, "rejected": 0}


def _chat(system, user, max_tokens=350, temperature=0.6):
    if provider() == "anthropic":
        return _anthropic(system, user, max_tokens, temperature)
    import requests
    url = os.getenv("LLM_API_URL", "https://api.mistral.ai/v1").rstrip("/") + "/chat/completions"
    body = {"model": model_name(), "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    r = requests.post(url, json=body, timeout=float(os.getenv("LLM_TIMEOUT", "8")),
                      headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY')}"})
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def _numbers(text):
    return {n.replace(",", "") for n in _NUM.findall(text or "")}


def _safe(reply, source, limit):
    """The reply only uses numbers found in the source, is not empty and not much longer than allowed."""
    reply = (reply or "").strip().strip('"').strip()
    if not reply or reply.upper().startswith("UNKNOWN") or len(reply) > limit:
        return None
    if not _numbers(reply) <= _numbers(source):
        return None
    return reply


def _history(turns):
    return "\n".join(f"{'Customer' if who == 'in' else 'You'}: {text[:300]}" for who, text in (turns or [])[-6:])


def _flag(name, default):
    return os.getenv(name, default).strip().lower() in ("on", "1", "true", "yes")


def natural_reply(approved, customer_text, company, turns=None, first_name=""):
    """The approved answer reworded for this conversation, or None to send the approved text as it is."""
    if not enabled() or not _flag("LLM_REWORD", "on") or not approved:
        return None
    _compose_status["calls"] += 1
    user = (f"Conversation so far:\n{_history(turns)}\n\nCustomer's latest message: \"{customer_text[:500]}\"\n"
            f"Customer's first name: {first_name or 'unknown'}\n\nAPPROVED ANSWER:\n{approved}")
    try:
        out = _safe(_chat(STYLE.format(company=company), user), approved, max(400, int(len(approved) * 1.6)))
    except Exception as exc:
        _status["errors"] += 1
        _status["last_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        log.warning("LLM reply failed: %s", _status["last_error"])
        return None
    _compose_status["used" if out else "rejected"] += 1
    return out


def answer_from_facts(question, facts, company, turns=None):
    """A short answer built only from the company's facts, or None when they do not cover the question."""
    if not enabled() or not facts or not _flag("LLM_ANSWERS", "on"):
        return None
    _compose_status["calls"] += 1
    user = (f"COMPANY FACTS:\n{facts}\n\nConversation so far:\n{_history(turns)}\n\n"
            f"Customer's question: \"{question[:500]}\"")
    try:
        out = _safe(_chat(ANSWER.format(company=company), user, temperature=0.3), facts, 700)
    except Exception as exc:
        _status["errors"] += 1
        _status["last_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        log.warning("LLM answer failed: %s", _status["last_error"])
        return None
    _compose_status["used" if out else "rejected"] += 1
    return out
