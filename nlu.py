"""Natural-language understanding for the solar sales assistant.

What it does with a message such as "hi, my inverter keeps beeping and the lights went off":
  1. Intent: which kind of message it is (buy a system, a fault, a price question, ...), predicted by a
     classifier that is trained on example phrases stored in the database (see `nlu_examples`).
  2. Entities: the details inside the text (town, phone number, kWh, system size, appliances, budget, name).
  3. Category and priority: purchase / enquiry / maintenance, and urgent / high / medium / low.

Models (open source):
  * "tfidf"       scikit-learn TF-IDF word + character n-grams -> logistic regression. Small and fast;
                  runs on Render's free plan. This is the default.
  * "transformer" a sentence-embedding transformer (fastembed, ONNX, no PyTorch) -> logistic regression.
                  Understands paraphrases and typos much better; needs about 300 MB RAM extra.
                  Switch on with NLU_BACKEND=transformer (and `pip install fastembed`).
Training means adding example phrases (the ERP "Train the bot" page does this) and calling train().
There is no model file: the model is rebuilt from the examples at start-up and after every edit, so the
database is the only thing that has to be kept.
"""
import difflib
import logging
import os
import re
import threading
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

log = logging.getLogger("solar-bot.nlu")

PRIORITIES = ("low", "medium", "high", "urgent")
CATEGORIES = ("purchase", "enquiry", "maintenance", "other")

# ---------------------------------------------------------------- text clean-up
SHORTHAND = {
    "pls": "please", "plz": "please", "plss": "please", "u": "you", "ur": "your", "r": "are", "wat": "what",
    "wht": "what", "hw": "how", "hv": "have", "abt": "about", "bt": "but", "cn": "can", "cant": "cannot",
    "dont": "do not", "doesnt": "does not", "isnt": "is not", "wont": "will not", "im": "i am", "ive": "i have",
    "thx": "thanks", "thnx": "thanks", "tnx": "thanks", "msg": "message", "tel": "tell", "info": "information",
    "sys": "system", "sytem": "system", "sollar": "solar", "solor": "solar", "invertor": "inverter",
    "inverta": "inverter", "batt": "battery", "batery": "battery", "bateries": "batteries", "pannels": "panels",
    "panel": "panels", "quation": "quotation", "qoute": "quote", "quatation": "quotation", "wanna": "want to",
    "gonna": "going to", "gotta": "got to", "kva": "kva", "kw": "kw", "zesa": "zesa", "hie": "hi", "hey": "hi",
    "hello": "hi", "heyy": "hi", "hii": "hi", "morning": "morning", "nd": "and", "n": "and", "wen": "when",
    "ryt": "right", "asap": "urgent", "tomorow": "tomorrow", "tommorow": "tomorrow", "wen": "when",
    "installed": "installed", "instal": "install", "intall": "install", "instalation": "installation",
}


def clean(text):
    """Lower-case, strip accents, expand SMS shorthand and squeeze repeated letters ('pleaseee' -> 'please')."""
    t = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()
    t = t.replace("'", "").replace("’", "")
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"[^a-z0-9$%+./ ]", " ", t)
    words = [SHORTHAND.get(w, w) for w in t.split()]
    return " ".join(words)


# ---------------------------------------------------------------- entities
APPLIANCES = {  # keyword -> (label, typical kWh per day)
    "fridge": ("fridge", 1.5), "refrigerator": ("fridge", 1.5), "freezer": ("freezer", 2.0),
    "deep freezer": ("freezer", 2.0), "tv": ("TV", 0.5), "television": ("TV", 0.5), "decoder": ("decoder", 0.1),
    "lights": ("lights", 0.5), "light": ("lights", 0.5), "lighting": ("lights", 0.5), "bulbs": ("lights", 0.5),
    "laptop": ("laptop", 0.3), "computer": ("computer", 0.6), "wifi": ("WiFi", 0.2), "router": ("WiFi", 0.2),
    "phone": ("phone charging", 0.1), "phones": ("phone charging", 0.1), "washing machine": ("washing machine", 0.6),
    "microwave": ("microwave", 0.6), "geyser": ("geyser", 3.5), "borehole": ("borehole pump", 2.5),
    "pump": ("pump", 2.5), "stove": ("stove", 3.0), "heater": ("heater", 2.5), "aircon": ("aircon", 4.0),
    "air conditioner": ("aircon", 4.0), "iron": ("iron", 0.3), "fan": ("fan", 0.3), "kettle": ("kettle", 0.3),
    "cctv": ("CCTV", 0.3), "security": ("security system", 0.3), "gate": ("gate motor", 0.2),
    "welding": ("welding machine", 5.0), "welder": ("welding machine", 5.0), "radio": ("radio", 0.1),
}
APPLIANCE_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in APPLIANCES), key=len, reverse=True)) + r")\b")
PRODUCTS = ("inverter", "battery", "batteries", "panels", "controller", "geyser", "pump", "charger")
TOWNS_FALLBACK = ["Harare", "Bulawayo", "Chitungwiza", "Mutare", "Gweru", "Masvingo", "Kwekwe", "Kadoma", "Marondera"]


def extract_phone(text):
    for m in re.finditer(r"(?<![\w.])\+?\d[\d\s().-]{7,17}\d", str(text or "")):
        digits = re.sub(r"\D", "", m.group())
        if 9 <= len(digits) <= 13:
            return m.group().strip()
    return ""


def extract_kwh(text):
    """Daily kWh from '10 kwh', '300 kwh a month' ... Returns None when no figure is given."""
    t = clean(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*kwh", t)
    if not m:
        return None
    value = float(m.group(1))
    tail = t[m.end():m.end() + 25]
    if "month" in tail or "/m" in tail or value > 100:
        value = value / 30
    return round(value, 1)


def extract_size(text):
    """System size in kVA/kW, e.g. '5kva' -> 5.0."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:kva|kw)\b", clean(text))
    return float(m.group(1)) if m else None


def extract_budget(text):
    t = str(text or "").lower().replace(",", "")
    m = re.search(r"(?:\$|usd\s*)\s?(\d{2,6})|(\d{2,6})\s*(?:usd|dollars|bucks)\b", t)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def extract_appliances(text):
    seen, total = [], 0.0
    for k in APPLIANCE_RE.findall(clean(text)):
        label, kwh = APPLIANCES[k]
        if label not in seen:
            seen.append(label)
            total += kwh
    return seen, round(total, 1)


def extract_products(text):
    t = clean(text)
    return [p for p in PRODUCTS if re.search(rf"\b{p}\b", t)]


NOT_NAMES = {"from", "in", "at", "and", "the", "a", "looking", "interested", "calling", "writing", "not", "so",
             "very", "of", "to", "here", "just", "also", "still", "now", "tired", "sorry", "happy", "new", "an"}


def extract_name(text):
    """A name the person introduced themselves with. 'my name is' works in any case; the looser
    'I am / this is' only counts when the name is capitalised, so 'I am tired of ZESA' is not a name."""
    text = str(text or "")
    m = re.search(r"\b(?i:my name is|name is|ndinonzi)\s+([A-Za-z][A-Za-z'-]+(?:\s+[A-Za-z][A-Za-z'-]+){0,2})", text)
    if not m:
        m = re.search(r"\b(?i:i am|im|i'm|this is|ndiri)\s+([A-Z][a-z'-]+(?:\s+[A-Z][a-z'-]+){0,2})", text)
    if not m:
        return ""
    words = []
    for w in m.group(1).split():
        if w.lower() in NOT_NAMES:
            break
        words.append(w)
    return " ".join(words[:3]).title()


NOT_PLACES = {"i", "the", "my", "zimbabwe", "home", "work", "night", "day", "winter", "summer", "future", "monday",
              "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march",
              "april", "may", "june", "july", "august", "september", "october", "november", "december", "this",
              "that", "our", "your", "a", "an", "it", "case", "fact", "general", "addition", "stock", "total"}


def extract_location(text, towns=None):
    """Town/suburb from the text: a known town (fuzzy for typos), or 'in/at/from <Capitalised words>'."""
    towns = towns or TOWNS_FALLBACK
    t = str(text or "")
    low = clean(t)
    for town in sorted(towns, key=len, reverse=True):
        if re.search(rf"\b{re.escape(town.lower())}\b", low):
            return town
    words = re.findall(r"[a-z]{4,}", low)
    for w in words:
        hit = difflib.get_close_matches(w, [x.lower() for x in towns], n=1, cutoff=0.82)
        if hit:
            return next(x for x in towns if x.lower() == hit[0])
    m = re.search(r"\b(?:in|at|from|around|near|stay in|live in|based in|located in)\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,2})", t)
    if m and m.group(1).split()[0].lower() not in NOT_PLACES:
        return m.group(1).strip()
    return ""


def extract_entities(text, towns=None):
    appliances, appliance_kwh = extract_appliances(text)
    return {k: v for k, v in {
        "phone": extract_phone(text), "location": extract_location(text, towns), "name": extract_name(text),
        "kwh": extract_kwh(text), "size_kva": extract_size(text), "budget": extract_budget(text),
        "appliances": appliances, "appliance_kwh": appliance_kwh, "products": extract_products(text),
    }.items() if v not in (None, "", [], 0, 0.0)}


# ---------------------------------------------------------------- priority rules
SAFETY = re.compile(r"\b(smoke|smoking|fire|burn(?:t|ing)?|spark(?:s|ing)?|shock(?:ed|s)?|electrocut\w*|melt(?:ed|ing)?|"
                    r"explod\w*|exploded|swollen|bulging|hot battery|smells? of|smells? like|short circuit|caught fire|"
                    r"leaking onto|flooded|got wet|water (?:got|came|entered|leaking))\b")
URGENT_WORDS = re.compile(r"\b(urgent\w*|emergency|immediately|right now|asap|as soon as possible|today|"
                          r"this morning|tonight)\b")
DOWN_WORDS = re.compile(r"\b(no power|power off|not working|stopped working|dead|completely off|totally off|"
                        r"no electricity|blackout|keeps tripping|tripping|not charging|wont charge|will not (?:turn|switch) on)\b")


def bump(priority, steps=1):
    return PRIORITIES[min(PRIORITIES.index(priority) + steps, len(PRIORITIES) - 1)]


def decide_priority(base, category, text, entities, existing_customer=False):
    """Priority from the intent's base level, then raised by what the person actually wrote."""
    t = clean(text)
    p = base if base in PRIORITIES else "medium"
    reasons = []
    if category == "maintenance":
        if SAFETY.search(t):
            return "urgent", ["Safety risk: smoke, fire, sparks, burning or shock mentioned"]
        if DOWN_WORDS.search(t):
            p = max(p, "high", key=PRIORITIES.index)
            reasons.append("System is down or not working")
        if existing_customer:
            p = bump(p)
            reasons.append("Existing customer")
    if category == "purchase":
        if entities.get("budget") or entities.get("size_kva") or entities.get("kwh"):
            p = max(p, "high", key=PRIORITIES.index)
            reasons.append("Gave a budget or system size: ready to buy")
        if existing_customer:
            reasons.append("Existing customer")
    if URGENT_WORDS.search(t) and p != "low":
        p = bump(p)
        reasons.append("Wants it done urgently")
    return p, reasons


# ---------------------------------------------------------------- the classifier
class _Tfidf:
    """TF-IDF word + character n-grams: robust to typos and tiny training sets."""
    name = "tfidf"

    def fit(self, texts):
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1)
        return normalize(hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr())

    def transform(self, texts):
        return normalize(hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr())

    similarity_floor = 0.25  # below this cosine similarity to every example the message is "not understood"


class _Transformer:
    """Sentence-embedding transformer through ONNX Runtime (fastembed). Understands paraphrases."""
    name = "transformer"
    similarity_floor = 0.6   # sentence embeddings of unrelated text still score about 0.4-0.55

    def __init__(self, model=None):
        from fastembed import TextEmbedding  # imported lazily: only needed for this backend
        self.model_name = model or os.getenv("NLU_MODEL", "BAAI/bge-small-en-v1.5")
        self.model = TextEmbedding(model_name=self.model_name)

    def _embed(self, texts):
        return normalize(np.asarray(list(self.model.embed(list(texts))), dtype=np.float32))

    def fit(self, texts):
        return self._embed(texts)

    def transform(self, texts):
        return self._embed(texts)


def _cosine_max(a, b):
    """Highest cosine similarity of each row of a against the rows of b (all rows L2-normalised)."""
    sims = a @ b.T
    sims = sims.toarray() if hasattr(sims, "toarray") else np.asarray(sims)
    return sims.max(axis=1), sims.argmax(axis=1)


class Engine:
    """Holds the trained model. Thread-safe: predictions can run while a new model is being trained."""

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self.info = {"trained": False}
        self._transformer = None

    # ---- training
    def _featurizer(self, backend):
        if backend == "transformer":
            if self._transformer is None:
                self._transformer = _Transformer()
            return self._transformer
        return _Tfidf()

    def train(self, examples, backend=None):
        """examples: list of (intent, text). Returns a summary dict (also kept in self.info)."""
        backend = (backend or os.getenv("NLU_BACKEND", "tfidf")).lower()
        data = [(i, clean(t)) for i, t in examples if i and clean(t)]
        labels = sorted({i for i, _ in data})
        if len(labels) < 2:
            self.info = {"trained": False, "reason": "Need examples for at least two intents."}
            return self.info
        note = ""
        try:
            feat = self._featurizer(backend)
        except Exception as exc:  # transformer not installed / model download failed
            log.warning("transformer backend unavailable (%s); using tfidf", exc)
            note, backend = f"Transformer unavailable ({type(exc).__name__}); using tfidf.", "tfidf"
            feat = _Tfidf()
        texts = [t for _, t in data]
        y = np.array([i for i, _ in data])
        x = feat.fit(texts)
        clf = LogisticRegression(C=20.0, max_iter=300, tol=1e-3, class_weight="balanced")
        clf.fit(x, y)
        model = {"feat": feat, "clf": clf, "x": x, "y": y, "texts": texts}
        with self._lock:
            self._model = model
            self.info = {"trained": True, "backend": feat.name, "examples": len(data), "intents": len(labels),
                         "accuracy": None, "note": note,
                         "model": getattr(feat, "model_name", "scikit-learn TF-IDF + logistic regression")}
            info = self.info
        # the accuracy check refits the model several times, so it runs in the background and
        # appears in `info` (and on the platform's Train page) when it is done
        self._accuracy_thread = threading.Thread(target=self._measure_accuracy, args=(info, x, y), daemon=True)
        self._accuracy_thread.start()
        log.info("NLU trained: %s", {k: v for k, v in info.items() if k != "accuracy"})
        return dict(info)

    @staticmethod
    def _cross_validate(x, y):
        """Cross-validated accuracy: each phrase is predicted by a model that never saw it."""
        _, counts = np.unique(y, return_counts=True)
        folds = int(min(4, counts.min()))
        if folds < 2:
            return None
        pred = cross_val_predict(LogisticRegression(C=20.0, max_iter=300, tol=1e-3, class_weight="balanced"), x, y,
                                 cv=StratifiedKFold(folds, shuffle=True, random_state=1))
        return round(float((pred == y).mean()), 3)

    def _measure_accuracy(self, info, x, y):
        try:
            accuracy = self._cross_validate(x, y)
        except Exception:
            log.exception("accuracy check failed")
            return
        with self._lock:
            if self.info is info:  # ignore if a newer model replaced this one meanwhile
                info["accuracy"] = accuracy

    def wait_for_accuracy(self, timeout=120):
        t = getattr(self, "_accuracy_thread", None)
        if t is not None:
            t.join(timeout)

    # ---- prediction
    def predict(self, text, top=3):
        """-> {'intent', 'confidence', 'similarity', 'ranked': [(intent, p)], 'matched': nearest example}"""
        with self._lock:
            m = self._model
        t = clean(text)
        if m is None or not t:
            return {"intent": None, "confidence": 0.0, "similarity": 0.0, "ranked": [], "matched": ""}
        v = m["feat"].transform([t])
        proba = m["clf"].predict_proba(v)[0]
        order = np.argsort(proba)[::-1][:top]
        classes = m["clf"].classes_
        sim, idx = _cosine_max(v, m["x"])
        return {"intent": str(classes[order[0]]), "confidence": float(proba[order[0]]),
                "similarity": float(sim[0]), "ranked": [(str(classes[i]), float(proba[i])) for i in order],
                "matched": m["texts"][int(idx[0])]}

    def understood(self, result, min_confidence=0.12, min_margin=1.5):
        """True when the model is sure enough to act on its guess.

        With ~90 intents the probabilities are spread thin, so a correct guess can be only 15-30%. What matters
        is that the best intent clearly beats the runner-up (margin) and that the message is close to something
        it was taught (similarity); gibberish fails both."""
        if not result["intent"]:
            return False
        floor = _Transformer.similarity_floor if self.info.get("backend") == "transformer" else _Tfidf.similarity_floor
        ranked = result.get("ranked") or []
        margin = ranked[0][1] / max(ranked[1][1], 1e-9) if len(ranked) > 1 else 99
        return result["confidence"] >= min_confidence and result["similarity"] >= floor and margin >= min_margin


engine = Engine()
