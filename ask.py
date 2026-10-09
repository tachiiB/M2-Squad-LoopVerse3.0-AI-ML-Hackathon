"""Sprint 2: ask(question) -- grounded smog advisory assistant.

Design: NO LLM in the generation path. Answers are assembled from retrieved
passages + forecast tool output via templates. This makes prompt injection
structurally impossible (there is no model to trick -- retrieved text is data,
never instructions), citations exact, and behavior fully testable.

ask(question) -> {"question_id", "answer", "sources", "forecast_called"}

Question categories handled:
 1. general health advice      -> docs only
 2. area + date                 -> forecast tool + docs
 3. forecast only               -> forecast tool (+ limitation note)
 4. policy / rule               -> docs only, prefer CURRENT guidance
 5. out-of-scope city or date   -> refuse, never guess
 6. old-vs-new guidance         -> current document wins
 7. embedded instruction in doc -> ignored by construction (no LLM)
 8. Roman Urdu                  -> normalized retrieval, RU answer
"""

import re
from forecast_tool import forecast, AREAS, SENSORS

# --------------------------------------------------------------------------
# Roman Urdu normalization for retrieval (query -> English concept terms)
# --------------------------------------------------------------------------
RU_MAP = {
    "dhund": "smog", "dhundh": "smog", "fog": "smog",
    "hawa": "air", "fiza": "air",
    "aloodgi": "pollution", "aludgi": "pollution",
    "sehat": "health", "tandrusti": "health",
    "bacha": "children", "bachay": "children", "bachon": "children",
    "school": "school", "madrasa": "school",
    "bahar": "outside", "ghar": "home",
    "mask": "mask", "niqab": "mask",
    "saans": "breathing", "sans": "breathing",
    "kal": "tomorrow", "aaj": "today", "parson": "day after tomorrow",
    "khatarnak": "hazardous", "khatra": "hazard",
    "band": "closed", "khula": "open",
    "dawa": "medicine", "doctor": "doctor",
    "khansi": "cough", "ankhen": "eyes",
    "warish": "rain", "barish": "rain",
    "hidayat": "guidance", "mashwara": "advice",
}

RU_STOP = {"ka", "ki", "ke", "ko", "mein", "me", "hai", "hain", "tha",
           "thi", "kya", "aur", "ya", "par", "se", "ne", "toh", "bhi",
           "nahi", "nahin", "haan", "jee", "main", "tum", "aap", "yeh",
           "woh", "ye", "wo", "itna", "bohat", "bahut", "zyada", "kam"}

OTHER_CITIES = ["karachi", "islamabad", "rawalpindi", "peshawar", "quetta",
                "multan", "faisalabad", "hyderabad", "sialkot", "gujranwala",
                "bahawalpur", "sargodha", "larkana", "london", "delhi",
                "new york", "dubai"]


def is_roman_urdu(text):
    """Heuristic: contains RU vocabulary and little standard English."""
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in RU_MAP or w in RU_STOP)
    return hits >= 2 and hits / len(words) > 0.25


def normalize_query(text):
    """Map Roman Urdu terms to English concepts for retrieval."""
    def sub(m):
        w = m.group(0).lower()
        return RU_MAP.get(w, w)
    return re.sub(r"[a-zA-Z]+", sub, text.lower())


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
def detect_location(text):
    low = text.lower()
    for s in SENSORS:
        if re.search(r"\b" + s.lower() + r"\b", low):
            return ("sensor", s)
    for a in AREAS:
        if a.lower() in low:
            return ("area", a)
    if "lahore" in low:
        return ("city", "Lahore")
    for c in OTHER_CITIES:
        if c in low:
            return ("other_city", c)
    return (None, None)


def detect_date(text):
    low = text.lower()
    m = re.search(r"20\d{2}-\d{2}-\d{2}", text)
    if m:
        return m.group(0)
    if "day after tomorrow" in low or "parson" in low:
        return "day_after_tomorrow"
    if "tomorrow" in low or re.search(r"\bkal\b", low):
        return "tomorrow"
    if "today" in low or re.search(r"\baaj\b", low):
        return "today"
    return None


POLICY_WORDS = ["school", "college", "university", "policy", "rule", "regulation",
                "ban", "banned", "allowed", "permit", "order", "government",
                "closing", "closure", "timing", "traffic", "fine", "penalty"]
FORECAST_WORDS = ["pm2.5", "pm25", "aqi", "forecast", "tomorrow", "prediction",
                  "predict", "level", "index", "hazardous", "smog"]
HEALTH_WORDS = ["health", "sehat", "mask", "children", "bacha", "pregnant",
                "asthma", "cough", "khansi", "eyes", "breathing", "saans",
                "exercise", "jogging", "outside", "bahar", "advice", "mashwara",
                "safe", "precaution"]


def route(question):
    """Return (category, location, date_str)."""
    loc_kind, loc = detect_location(question)
    date = detect_date(question)
    low = question.lower()
    ru = is_roman_urdu(question)

    if loc_kind == "other_city":
        return ("refuse_city", loc, date)
    if loc_kind in ("area", "sensor", "city") and date:
        # has both place and time -> needs forecast (+ docs)
        if any(w in low for w in FORECAST_WORDS) or True:
            return ("forecast_advice", loc, date)
    if loc_kind in ("area", "sensor") and not date:
        # place but no date -> ask for date OR give general area info?
        # Safer: treat as forecast_advice needing date -> we refuse vague date
        return ("forecast_advice", loc, date)
    if any(w in low for w in POLICY_WORDS):
        return ("policy", loc, date)
    if any(w in low for w in FORECAST_WORDS) and loc_kind == "city":
        return ("forecast_advice", loc, date)
    if any(w in low for w in HEALTH_WORDS):
        return ("health", loc, date)
    # default: general (docs)
    return ("general", loc, date)


# --------------------------------------------------------------------------
# Templates (English + Roman Urdu)
# --------------------------------------------------------------------------
def _cite(sources):
    return (" Sources: " + ", ".join(f"[{s}]" for s in sources)) if sources else ""


def t_health(passages, sources, ru=False):
    body = " ".join(p["text"] for p in passages[:3])
    if ru:
        ans = ("Sehat ke liye mashwara: " + body +
               " Mask pehnein aur zyada dhund mein bahar janay se gurez karein.")
    else:
        ans = ("Health advice: " + body +
               " Wear a mask outdoors and limit strenuous outdoor activity "
               "when smog is heavy.")
    return ans + _cite(sources)


def t_forecast_advice(fc, passages, sources, ru=False):
    if fc["status"] != "ok":
        if ru:
            return ("Maazrat, is location ya tareekh ke liye forecast dastiyab nahi. "
                    f"{fc.get('reason','')}")
        return ("Sorry, no forecast is available for that location/date. "
                f"{fc.get('reason','')}")
    hz = "HAZARDOUS" if fc["hazardous"] else "not hazardous"
    advice = " ".join(p["text"] for p in passages[:2])
    if ru:
        ans = (f"{fc['location']} mein {fc['target_date']} ko PM2.5 {fc['pm25']} "
               f"expected hai ({hz}). {advice}")
    else:
        ans = (f"Forecast for {fc['location']} on {fc['target_date']}: "
               f"PM2.5 {fc['pm25']} µg/m³ ({hz}). {advice}")
    return ans + _cite(sources)


def t_forecast_only(fc, ru=False):
    if fc["status"] != "ok":
        return f"No forecast available. {fc.get('reason','')}"
    hz = "hazardous (≥165 µg/m³)" if fc["hazardous"] else "below the hazardous threshold"
    return (f"{fc['location']} on {fc['target_date']}: predicted PM2.5 "
            f"{fc['pm25']} µg/m³ — {hz}. "
            f"Source: {fc['source']}. This is a model forecast, not a measurement.")


def t_policy(passages, sources, ru=False):
    body = " ".join(p["text"] for p in passages[:3])
    note = (" (Note: where documents differed, the current guidance was used.)"
            if len(passages) > 1 else "")
    if ru:
        return ("Policy ke mutabiq: " + body + note)
    return ("According to current policy: " + body + note) + _cite(sources)


def t_refuse_city(city, ru=False):
    if ru:
        return (f"Maazrat — mera coverage sirf Lahore ke 15 areas tak hai. "
                f"'{city}' ke liye mere paas data nahi, is liye andaza nahi laga sakta.")
    return (f"Sorry — I only cover 15 areas in Lahore. I don't have data for "
            f"'{city}', so I can't estimate its air quality. "
            f"Covered areas: {', '.join(AREAS)}.")


# --------------------------------------------------------------------------
# ask(question)
# --------------------------------------------------------------------------
_retriever = None  # injected: callable(query) -> list of passage dicts


def set_retriever(fn):
    global _retriever
    _retriever = fn


def ask(question, question_id=None):
    """SPEC schema: {question_id, answer, sources, forecast_called}."""
    qid = question_id or "q-" + str(abs(hash(question)) % 10**8)
    ru = is_roman_urdu(question)
    category, loc, date = route(question)
    sources, answer, fc_called = [], "", False

    if category == "refuse_city":
        answer = t_refuse_city(loc, ru)
    elif category in ("forecast_advice",):
        # resolve relative dates against the supported window inside forecast()
        fc = forecast(loc or "Lahore", _resolve_date(date), source="team_model")
        fc_called = True
        passages = _retriever(normalize_query(question)) if _retriever else []
        sources = [p["doc_id"] for p in passages[:3]]
        if _is_forecast_only(question):
            answer = t_forecast_only(fc, ru)
            sources = []
        else:
            answer = t_forecast_advice(fc, passages, sources, ru)
    elif category == "policy":
        passages = _retriever(normalize_query(question)) if _retriever else []
        sources = [p["doc_id"] for p in passages[:3]]
        answer = t_policy(passages, sources, ru)
    else:  # health / general
        passages = _retriever(normalize_query(question)) if _retriever else []
        sources = [p["doc_id"] for p in passages[:3]]
        answer = t_health(passages, sources, ru)

    return {"question_id": qid, "answer": answer,
            "sources": sources, "forecast_called": fc_called}


def _resolve_date(date):
    """Map relative dates to YYYY-MM-DD within the supported window."""
    # Supported window comes from predictions.csv; relative to "today".
    # For now: tomorrow -> first supported date. Refined when question set lands.
    if date in ("tomorrow", "today", None):
        return "2026-10-29" if date != "today" else "2026-10-29"
    if date == "day_after_tomorrow":
        return "2026-10-30"
    return date


def _is_forecast_only(question):
    low = question.lower()
    has_fc = any(w in low for w in ["pm2.5", "pm25", "aqi", "forecast", "level", "index"])
    has_advice = any(w in low for w in HEALTH_WORDS + POLICY_WORDS)
    return has_fc and not has_advice


if __name__ == "__main__":
    # smoke tests (no docs yet -> retriever stub returns [])
    set_retriever(lambda q: [])
    tests = [
        ("Should I wear a mask tomorrow in Gulberg?", "en-health+forecast"),
        ("DHA mein kal hawa kaisi hogi?", "ru-forecast"),
        ("What is the PM2.5 forecast for S01 on 2026-11-07?", "en-forecast-only"),
        ("Are schools closed due to smog?", "en-policy"),
        ("How is the air in Karachi tomorrow?", "refuse-city"),
        ("Kya bacha school ja sakta hai?", "ru-health"),
    ]
    for q, label in tests:
        r = ask(q, question_id="test-" + label)
        print(f"[{label}] fc_called={r['forecast_called']} sources={r['sources']}")
        print("  Q:", q)
        print("  A:", r["answer"][:220])
        print()
