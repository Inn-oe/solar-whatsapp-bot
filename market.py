"""Market prices: reads advertised solar-system prices from public web pages and summarises them.

How it works
  * A list of sources (seller price pages, classifieds per town) is read at most once per refresh
    period (24 hours by default), politely: robots.txt is obeyed, the bot identifies itself, one
    request per page with a pause between sites.
  * Each page's text is scanned for a system size (e.g. "5kVA") followed by a US-dollar price.
    Prices outside a sane cost-per-kVA band, and inverter-only lines, are dropped.
  * Prices are grouped by standard size and by town, giving a typical range (middle half of the
    adverts) that the bot shows to clients and staff.

Facebook is not a source: it refuses automated requests to public pages, and its terms forbid
scraping. Its official route (Meta's "Page Public Content Access") needs app review.
"""
import datetime as dt
import html
import json
import logging
import re
import statistics
import threading
import time
import urllib.parse
import urllib.robotparser

import requests

import db

log = logging.getLogger("solar-bot.market")

USER_AGENT = "SolarMarketBot/1.0 (solar price survey; obeys robots.txt)"
MAX_BYTES = 2_000_000
STANDARD_SIZES = (1, 1.5, 2, 3, 5, 6, 8, 10, 12, 15, 20)
PER_KVA = (180, 2000)       # USD per kVA; outside this a "price" is a part, a deposit or a typo
MIN_PRICES, MIN_SOURCES = 3, 2   # what a size needs before a range is shown to clients

DEFAULT_SOURCES = [
    {"name": "Solarpro", "url": "https://solarpro.co.zw/priceguide", "location": ""},
    {"name": "Sona Solar packages", "url": "https://www.sonasolar.co.zw/p/solar-system-packages.html", "location": "Harare"},
    {"name": "Sona Solar price list", "url": "https://www.sonasolar.co.zw/2026/03/5kva-solar-system-prices.html", "location": "Harare"},
    {"name": "Solar Power Shop", "url": "https://solarpowershop.co.zw/solar-system-prices-in-zimbabwe-buyers-guide/", "location": ""},
    {"name": "Solar Options", "url": "https://www.solaroptions.co.zw/p/prices.html", "location": "Harare"},
    {"name": "Solar Reviews", "url": "https://www.solarreviews.co.zw/2025/10/solar-system-packages-and-prices.html", "location": ""},
    {"name": "Solar Reviews installations", "url": "https://www.solarreviews.co.zw/p/solar-installations.html", "location": ""},
    {"name": "Solar Quotes", "url": "https://www.solarquotes.co.zw/2026/05/solar-power-system-installations.html", "location": ""},
    {"name": "ProExpert Solar", "url": "https://solarinstallation.co.zw/", "location": "Bulawayo"},
    {"name": "Construction Market Book", "url": "https://constructionmarketbook.com/articles/cost-of-solar-installation-in-zimbabwe-2026-guide", "location": ""},
    {"name": "Classifieds Harare", "url": "https://www.classifieds.co.zw/zimbabwe-solar-panels/harare", "location": "Harare"},
    {"name": "Classifieds Bulawayo", "url": "https://www.classifieds.co.zw/zimbabwe-solar-panels/bulawayo", "location": "Bulawayo"},
    {"name": "Classifieds Mutare", "url": "https://www.classifieds.co.zw/zimbabwe-solar-panels/mutare", "location": "Mutare"},
    {"name": "Classifieds Gweru", "url": "https://www.classifieds.co.zw/zimbabwe-solar-panels/gweru", "location": "Gweru"},
    {"name": "Classifieds Masvingo", "url": "https://www.classifieds.co.zw/zimbabwe-solar-panels/masvingo", "location": "Masvingo"},
]


# ---------------------------------------------------------------- sources
def validate_sources(sources):
    if not isinstance(sources, list) or len(sources) > 60:
        return "Sources must be a list (at most 60)."
    names = set()
    for s in sources:
        if not isinstance(s, dict) or not all(isinstance(s.get(k), str) for k in ("name", "url")):
            return 'Every source needs a "name" and a "url".'
        if not s["name"].strip() or s["name"] in names:
            return f'Source names must be unique and not empty ("{s.get("name")}").'
        names.add(s["name"])
        parts = urllib.parse.urlsplit(s["url"])
        if parts.scheme != "https" or not parts.netloc:
            return f'"{s["url"]}" must be an https:// address.'
        if parts.netloc.lower().endswith(("facebook.com", "fb.com", "instagram.com")):
            return "Facebook and Instagram cannot be read automatically (they block it and forbid it)."
        if not isinstance(s.get("location", ""), str):
            return '"location" must be a town name, or empty for national.'
    return None


def get_sources():
    saved = db.get_setting("market_sources")
    if saved:
        try:
            data = json.loads(saved)
            if validate_sources(data) is None:
                return data
        except ValueError:
            pass
    return [dict(s) for s in DEFAULT_SOURCES]


def set_sources(sources):
    old = {s["name"] for s in get_sources()}
    db.set_setting("market_sources", json.dumps(sources))
    for gone in old - {s["name"] for s in sources}:
        db.drop_market_source(gone)


# ---------------------------------------------------------------- reading a page
def text_of(raw):
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<(br|/p|/div|/li|/tr|/h[1-6]|/td|/th|/a|/span)\b[^>]*>", "\n", raw)
    raw = html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))
    return re.sub(r"[ \t\xa0]+", " ", raw)


SIZE_RE = re.compile(r"(?i)(?<![\d.,])(\d{1,2}(?:\.\d)?)\s?-?\s?(?:kva|kw)\b")
PRICE_RE = re.compile(r"(?i)(?:US\s?\$|USD|\$)\s?(\d{1,3}(?:,\d{3})+|\d{3,6})(?:\.\d{1,2})?(?!\d|\s?(?:w|wh|ah|v|kw|kva)\b)")
PART_RE = re.compile(r"(?i)^[\s\-–—:|(]*(?:(?:hybrid|pure sine(?: wave)?|off[- ]grid|solar|must|deye|sunsynk|growatt|kodak)\s+){0,3}"
                     r"inverter(?!\s*(?:\+|and|with|,))")
PRE_PART_RE = re.compile(r"(?i)(?:batter|lithium|panel|inverter|charger|controller|cable)[^.|]{0,45}\bfor\s+(?:a\s+|an\s+|the\s+)?$")
PRICE_FOR_SIZE_RE = re.compile(r"(?i)^(?:\.\d{1,2})?\s*for\s+(?:a\s+|an\s+|the\s+)?(\d{1,2}(?:\.\d)?)\s?-?\s?(?:kva|kw)\b")
RANGE_GAP_RE = re.compile(r"(?i)^[\s+]*(?:-|–|—|to|and)?\s*$")
SYSTEM_WORDS = re.compile(r"(?i)system|package|kit|bundle|combo|complete|installed|installation")
KIT_WORDS = re.compile(r"(?i)\bkit\b|equipment (?:only|package)|supply only|excluding install|without install|excl\.? install")
INSTALLED_WORDS = re.compile(r"(?i)install")


def _kind(line):
    if KIT_WORDS.search(line):
        return "kit"
    return "installed" if INSTALLED_WORDS.search(line) else ""


def _plausible(size, price):
    return 0.5 <= size <= 30 and PER_KVA[0] <= price / size <= PER_KVA[1]


def _pairs_in(line):
    """(size, price) pairs where a price follows a size mention in the same line."""
    sizes = [(m.start(), m.end(), float(m.group(1))) for m in SIZE_RE.finditer(line)]
    prices = [(m.start(), m.end(), int(m.group(1).replace(",", ""))) for m in PRICE_RE.finditer(line)]
    out = []
    for n, (pos, end, price) in enumerate(prices):
        after = PRICE_FOR_SIZE_RE.match(line[end:end + 40])
        if after:   # "US$520 for a 1kVA kit": the size comes after the price
            out.append((float(after.group(1)), price))
            continue
        before = [s for s in sizes if s[1] <= pos]
        if not before:
            continue
        s_start, s_end, size = before[-1]
        gap = line[s_end:pos]
        between = [q for q in prices[:n] if q[0] >= s_end]
        if len(gap) > 220 or len(between) > 1:
            continue
        if between and not (RANGE_GAP_RE.match(line[between[0][1]:pos]) and pos - between[0][1] <= 12):
            continue   # a second price counts only as the top of a range: "$600 - $900"
        if PART_RE.search(gap) and not SYSTEM_WORDS.search(line):
            continue   # "5kVA inverter - $450" is a part, not a system
        if PRE_PART_RE.search(line[max(0, s_start - 80):s_start]):
            continue   # "lithium battery for 3kVA systems - $1,200" is a part too
        out.append((size, price))
    return out


def extract(text):
    """All plausible system prices in a page's text: [{size_kva, price, kind, snippet}]."""
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    found, seen = [], set()

    def add(size, price, line):
        if _plausible(size, price) and (size, price) not in seen:
            seen.add((size, price))
            found.append({"size_kva": size, "price": price, "kind": _kind(line),
                          "snippet": re.sub(r"\s+", " ", line)[:300]})

    for i, line in enumerate(lines):
        if len(line) > 4000:
            continue
        pairs = _pairs_in(line)
        for size, price in pairs:
            add(size, price, line)
        if pairs or not PRICE_RE.search(line):
            continue
        # card layouts: a short heading with the size, then the price on one of the next lines
        for back in range(1, 4):
            if i - back < 0:
                break
            head = lines[i - back]
            if PRICE_RE.search(head):
                break
            if len(head) <= 90 and len(set(SIZE_RE.findall(head))) == 1:
                joined = head + " | " + line[:200]
                for size, price in _pairs_in(joined):
                    add(size, price, joined)
                break
    return found


_robots = {}


def allowed(url):
    """robots.txt check (cached for the life of the process)."""
    parts = urllib.parse.urlsplit(url)
    host = f"{parts.scheme}://{parts.netloc}"
    if host not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = requests.get(host + "/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=15)
            rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        except requests.RequestException:
            rp.parse([])
        _robots[host] = rp
    return _robots[host].can_fetch(USER_AGENT, url)


def read_source(src):
    """Fetch one source. Returns (rows, status text)."""
    url = src["url"]
    if not allowed(url):
        return [], "skipped: the site's robots.txt does not allow it"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"}, timeout=25, stream=True)
        if r.status_code != 200:
            return [], f"failed: the site answered {r.status_code}"
        raw = r.raw.read(MAX_BYTES, decode_content=True).decode(r.encoding or "utf-8", "replace")
    except requests.RequestException as exc:
        return [], f"failed: {type(exc).__name__}"
    rows = extract(text_of(raw))
    for row in rows:
        row.update(url=url, location=src.get("location", ""))
    return rows, f"ok: {len(rows)} prices" if rows else "ok, but no system prices found on the page"


# ---------------------------------------------------------------- refresh
_lock = threading.Lock()


def refresh_hours():
    return 24


def last_refresh():
    try:
        return dt.datetime.fromisoformat(db.get_setting("market_refreshed") or "")
    except ValueError:
        return None


def is_stale(hours=None):
    last = last_refresh()
    return last is None or dt.datetime.now(dt.timezone.utc) - last > dt.timedelta(hours=hours or refresh_hours())


def refresh():
    """Read every source now. Returns {source name: status}. Only one refresh runs at a time."""
    if not _lock.acquire(blocking=False):
        return None
    try:
        status = {}
        for i, src in enumerate(get_sources()):
            if i:
                time.sleep(2)   # be gentle: never hammer the sites
            try:
                rows, note = read_source(src)
            except Exception:
                log.exception("market source %s crashed", src["name"])
                rows, note = [], "failed: unexpected error"
            if rows:
                db.replace_market_prices(src["name"], rows)
            status[src["name"]] = {"note": note, "count": len(rows),
                                   "checked": dt.datetime.now(dt.timezone.utc).isoformat()}
            log.info("market %s -> %s", src["name"], note)
        db.set_setting("market_status", json.dumps(status))
        db.set_setting("market_refreshed", dt.datetime.now(dt.timezone.utc).isoformat())
        _cache.clear()
        return status
    finally:
        _lock.release()


def loop(get_hours=refresh_hours):
    """Background thread: keep the data fresh for as long as the service runs."""
    time.sleep(20)
    while True:
        try:
            if is_stale(get_hours()):
                refresh()
        except Exception:
            log.exception("market refresh failed")
        time.sleep(1800)


# ---------------------------------------------------------------- summary
_cache = {}


def bucket(size):
    """The standard size class a system belongs to: 3.2 and 3.5kVA are '3kVA class', 5.5 is '5kVA class'.

    A system counts in the largest class it reaches (within 5%), unless it is over 25% bigger."""
    reached = [s for s in STANDARD_SIZES if s <= size * 1.05]
    if not reached:
        return None
    best = max(reached)
    return best if size <= best * 1.25 else None


def _site(url):
    host = urllib.parse.urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _round(n):
    return int(round(n / 10.0) * 10)


def _stats(rows):
    prices = sorted(r["price"] for r in rows)
    if len(prices) >= 4:
        q = statistics.quantiles(prices, n=4, method="inclusive")
        low, high = q[0], q[2]
    else:
        low, high = prices[0], prices[-1]
    return {"low": _round(low), "high": _round(high), "median": _round(statistics.median(prices)),
            "min": prices[0], "max": prices[-1], "count": len(prices),
            "sources": len({_site(r["url"]) for r in rows})}   # two pages of one website count once


def summary():
    """{'sizes': {size: {'all': stats, 'towns': {town: stats}}}, 'refreshed': iso, 'total': n} (cached 5 min)."""
    hit = _cache.get("summary")
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    rows = db.market_prices()
    by_size = {}
    for r in rows:
        b = bucket(r["size_kva"])
        if b is not None:
            by_size.setdefault(b, []).append(r)
    sizes = {}
    for b, items in sorted(by_size.items()):
        whole = [r for r in items if r["kind"] != "kit"]      # prefer prices that are not equipment-only
        use = whole if len(whole) >= MIN_PRICES else items
        towns = {}
        for town in sorted({r["location"] for r in items if r["location"]}):
            local = [r for r in use if r["location"] == town]
            if local:
                towns[town] = _stats(local)
        sizes[b] = {"all": _stats(use), "towns": towns, "kit_only_count": len(items) - len(whole)}
    last = last_refresh()
    out = {"sizes": sizes, "refreshed": last.isoformat() if last else None, "total": len(rows)}
    _cache["summary"] = (time.time(), out)
    return out


def _enough(st):
    return bool(st) and st["count"] >= MIN_PRICES and st["sources"] >= MIN_SOURCES


def range_for(size_kva, town=""):
    """Typical advertised range for a system size, or None when there is too little data.

    Uses the client's town when that town alone has enough adverts, else the whole country."""
    try:
        b = bucket(float(size_kva))
    except (TypeError, ValueError):
        return None
    entry = summary()["sizes"].get(b) if b is not None else None
    if not entry:
        return None
    local = None
    for name, st in entry["towns"].items():
        if town and name.lower() == town.strip().lower() and _enough(st):
            local = dict(st, where=name)
    if local:
        return dict(local, size=b)
    return dict(entry["all"], where="Zimbabwe", size=b) if _enough(entry["all"]) else None


def status():
    try:
        return json.loads(db.get_setting("market_status") or "{}")
    except ValueError:
        return {}


def size_label(b):
    return f"{b:g}kVA"
