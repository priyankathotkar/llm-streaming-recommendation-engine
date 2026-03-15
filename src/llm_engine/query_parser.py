import os
import json
import requests
from difflib import get_close_matches
from typing import List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------
# OpenAI (optional)
# ---------------------------------------------------
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    _OPENAI_AVAILABLE = False

# ---------------------------------------------------
# Ollama config (local Llama3 — your project default)
# ---------------------------------------------------
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
_OLLAMA_AVAILABLE = False

try:
    r = requests.get("http://localhost:11434", timeout=2)
    _OLLAMA_AVAILABLE = r.status_code == 200
except Exception:
    pass

# ---------------------------------------------------
# Keyword maps
# ---------------------------------------------------
EMOTION_KEYWORDS = {
    "sad":      ["sad", "depressed", "unhappy", "down", "cry", "upset", "grief", "heartbroken", "miserable"],
    "happy":    ["happy", "joyful", "cheerful", "great", "wonderful", "good mood", "elated", "glad"],
    "angry":    ["angry", "mad", "furious", "rage", "frustrated", "irritated", "annoyed"],
    "bored":    ["bored", "boring", "nothing to do", "dull", "uneventful"],
    "excited":  ["excited", "pumped", "hyped", "thrilled", "energetic", "eager"],
    "romantic": ["romantic", "date", "love", "anniversary", "valentine", "crush", "relationship"],
    "anxious":  ["anxious", "nervous", "worried", "stressed", "tense", "overwhelmed"],
    "lonely":   ["lonely", "alone", "isolated", "miss", "empty", "disconnected"],
    "tired":    ["tired", "exhausted", "sleepy", "relax", "chill", "rest"],
    "scared":   ["scared", "afraid", "frightened", "fear", "terrified", "horror"],
    "surprised":["surprised", "shocked", "unexpected", "twist", "wow"],
    "neutral":  ["neutral", "anything", "whatever", "no preference", "random"],
}

GENRE_KEYWORDS = [
    "action", "comedy", "drama", "thriller", "romance",
    "sci-fi", "science fiction", "fantasy", "horror",
    "animation", "animated", "documentary", "mystery",
    "adventure", "crime", "family", "musical", "music",
    "western", "biography", "history", "sport", "war",
]

# Normalise genre aliases
GENRE_ALIASES = {
    "science fiction": "sci-fi",
    "animated":        "animation",
    "biographical":    "biography",
    "biographical drama": "biography",
    "romcom":          "romance",
    "rom-com":         "romance",
    "rom com":         "romance",
    "superhero":       "action",
    "kids":            "family",
    "children":        "family",
}

# ---------------------------------------------------
# LLM prompt template
# ---------------------------------------------------
LLM_PROMPT = """You are a movie recommendation assistant.
Analyze the user's input and extract ONLY the following fields.
Return ONLY valid JSON — no explanation, no markdown, no code blocks.

Format:
{{
  "reference_movies": [],
  "genres": [],
  "emotion": null
}}

Rules:
- reference_movies: movie titles explicitly mentioned (list of strings)
- genres: movie genres mentioned or implied (list of strings, lowercase)
- emotion: single dominant emotion word or null

User input: \"{query}\"
"""


class QueryParser:
    """
    Parses user input to extract reference movies, genres, and emotion.
    Priority: Ollama (Llama3) → OpenAI → rule-based fallback
    """

    def __init__(self, movie_titles: List[str]):
        self.movie_titles = [t.strip() for t in movie_titles if t]

    # ---------------------------------------------------
    # LLM: Ollama / Llama3 (local)
    # ---------------------------------------------------

    def _ollama_parse(self, user_query: str) -> Optional[dict]:
        """
        Call local Ollama Llama3 model to parse the query.
        """
        if not _OLLAMA_AVAILABLE:
            return None

        try:
            payload = {
                "model":  OLLAMA_MODEL,
                "prompt": LLM_PROMPT.format(query=user_query),
                "stream": False,
                "options": {"temperature": 0}
            }

            response = requests.post(OLLAMA_URL, json=payload, timeout=15)
            response.raise_for_status()

            raw = response.json().get("response", "").strip()
            raw = self._clean_json(raw)

            return json.loads(raw)

        except Exception as e:
            print(f"[QueryParser] Ollama error: {e}")
            return None

    # ---------------------------------------------------
    # LLM: OpenAI (cloud fallback)
    # ---------------------------------------------------

    def _openai_parse(self, user_query: str) -> Optional[dict]:
        """
        Call OpenAI GPT to parse the query.
        """
        if not _OPENAI_AVAILABLE:
            return None

        try:
            client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": LLM_PROMPT.format(query=user_query)}],
                temperature=0,
            )

            raw = response.choices[0].message.content.strip()
            raw = self._clean_json(raw)

            return json.loads(raw)

        except Exception as e:
            print(f"[QueryParser] OpenAI error: {e}")
            return None

    # ---------------------------------------------------
    # Rule-based fallback parser
    # ---------------------------------------------------

    def _fallback_parse(self, user_query: str) -> Tuple[List[str], List[str], Optional[str]]:
        """
        Keyword + fuzzy matching fallback when no LLM is available.
        """
        query_lower = user_query.lower().strip()

        # ── Movie detection (exact substring first) ──────────────────────────
        detected_movies = []
        for title in self.movie_titles:
            if title.lower() in query_lower:
                detected_movies.append(title)

        # Fuzzy match only if no exact match found — higher cutoff to avoid false positives
        if not detected_movies:
            matches = get_close_matches(user_query, self.movie_titles, n=2, cutoff=0.6)
            detected_movies.extend(matches)

        # ── Genre detection ──────────────────────────────────────────────────
        detected_genres = []
        for g in GENRE_KEYWORDS:
            if g in query_lower:
                canonical = GENRE_ALIASES.get(g, g)
                if canonical not in detected_genres:
                    detected_genres.append(canonical)

        # also check aliases directly
        for alias, canonical in GENRE_ALIASES.items():
            if alias in query_lower and canonical not in detected_genres:
                detected_genres.append(canonical)

        # ── Emotion detection ────────────────────────────────────────────────
        detected_emotion = None
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                detected_emotion = emotion
                break

        return detected_movies, detected_genres, detected_emotion

    # ---------------------------------------------------
    # Clean raw LLM JSON output
    # ---------------------------------------------------

    def _clean_json(self, raw: str) -> str:
        """
        Strip markdown code fences and extract the first JSON object.
        """
        # Remove ```json ... ``` or ``` ... ```
        if "```" in raw:
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        # Extract first { ... } block
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        return raw.strip()

    # ---------------------------------------------------
    # Validate and sanitise LLM output
    # ---------------------------------------------------

    def _sanitise(self, parsed: dict) -> dict:
        """
        Validate types, normalise genres, fuzzy-match movie titles.
        """
        rm = parsed.get("reference_movies", [])
        gs = parsed.get("genres", [])
        em = parsed.get("emotion", None)

        # Ensure correct types
        rm = rm if isinstance(rm, list) else []
        gs = gs if isinstance(gs, list) else []
        em = em if isinstance(em, str) else None

        # Validate movie titles with fuzzy matching
        validated_movies = []
        for m in rm:
            if not isinstance(m, str):
                continue
            matches = get_close_matches(m, self.movie_titles, n=1, cutoff=0.55)
            if matches:
                validated_movies.extend(matches)
            else:
                # keep LLM's title if no match — still useful for semantic search
                validated_movies.append(m)

        # Deduplicate movies
        seen = set()
        validated_movies = [x for x in validated_movies if not (x in seen or seen.add(x))]

        # Normalise genres (lowercase + alias resolution)
        normalised_genres = []
        for g in gs:
            if not isinstance(g, str):
                continue
            g_lower   = g.lower().strip()
            canonical = GENRE_ALIASES.get(g_lower, g_lower)
            if canonical not in normalised_genres:
                normalised_genres.append(canonical)

        # Normalise emotion
        if em:
            em = em.lower().strip()
            # if LLM returned something not in our map, try keyword match
            if em not in EMOTION_KEYWORDS:
                for emotion, keywords in EMOTION_KEYWORDS.items():
                    if em in keywords or emotion in em:
                        em = emotion
                        break
                else:
                    em = None

        return {
            "reference_movies": validated_movies,
            "genres":           normalised_genres,
            "emotion":          em,
        }

    # ---------------------------------------------------
    # Main parse entry point
    # ---------------------------------------------------

    def parse(self, user_query: str) -> dict:
        """
        Parse user query. Returns:
        {
            "reference_movies": [...],
            "genres":           [...],
            "emotion":          "..." or null
        }
        """
        empty = {"reference_movies": [], "genres": [], "emotion": None}

        if not user_query or not user_query.strip():
            return empty

        parsed = None

        # Priority 1 — Ollama (local Llama3)
        if _OLLAMA_AVAILABLE:
            print("[QueryParser] Using Ollama Llama3")
            parsed = self._ollama_parse(user_query)

        # Priority 2 — OpenAI
        if parsed is None and _OPENAI_AVAILABLE:
            print("[QueryParser] Using OpenAI")
            parsed = self._openai_parse(user_query)

        # Priority 3 — Rule-based fallback
        if parsed is None:
            print("[QueryParser] Using rule-based fallback")
            movies, genres, emotion = self._fallback_parse(user_query)
            parsed = {
                "reference_movies": movies,
                "genres":           genres,
                "emotion":          emotion,
            }

        return self._sanitise(parsed)