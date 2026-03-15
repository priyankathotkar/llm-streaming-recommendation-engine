import os
import json
import requests
import threading
from typing import List, Optional

# ---------------------------------------------------
# Config — tuned for SPEED
# ---------------------------------------------------
OLLAMA_URL     = os.getenv("OLLAMA_URL",   "http://localhost:11434/api/generate")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3")

# ⚡ Key speed settings:
MAX_TOKENS     = 60    # was 180 — shorter = much faster
TEMPERATURE    = 0.6
TIMEOUT_SEC    = 8     # was 30 — hard cutoff, fall back instantly

# ---------------------------------------------------
# Mood personalities — greeting + closing are instant
# (shown immediately, no LLM needed)
# ---------------------------------------------------
MOOD_PERSONALITY = {

    "sad": {
        "greeting": "Hey, I can see you're going through a tough time. 💙 Here are some movies to lift your spirits.",
        "closing":  "I hope one of these helps you feel better. You deserve it. 🌟",
        "prompt":   "User is SAD. In 1-2 short sentences, warmly explain why these uplifting movies will cheer them up. Be like a caring friend.",
    },
    "angry": {
        "greeting": "Take a deep breath. 😌 These light, fun films will help you unwind.",
        "closing":  "Give yourself a break — you've earned it. 🧘",
        "prompt":   "User is ANGRY. In 1-2 short sentences, calmly explain how these movies will help them relax. Be soothing.",
    },
    "happy": {
        "greeting": "Great mood — let's keep that energy going! 🚀 These movies match your vibe perfectly.",
        "closing":  "Enjoy the ride! These are going to be a blast. 🎉",
        "prompt":   "User is HAPPY. In 1-2 short sentences, enthusiastically recommend these movies. Be fun and upbeat.",
    },
    "excited": {
        "greeting": "You are READY! ⚡ Here are some epic picks to fuel that excitement.",
        "closing":  "Get the popcorn ready! 🍿🔥",
        "prompt":   "User is EXCITED. In 1-2 short sentences, hype up these movies with energy.",
    },
    "anxious": {
        "greeting": "Hey, it's okay. 🌿 These light, easy films will help quiet your mind.",
        "closing":  "Press play and let go for a while. You've got this. 💚",
        "prompt":   "User is ANXIOUS. In 1-2 short sentences, gently explain how these comforting movies will ease their mind.",
    },
    "lonely": {
        "greeting": "You're not alone — I'm here. 🤗 These films are full of warmth and human connection.",
        "closing":  "A great story is the best company. Hope these feel like a warm hug. 💛",
        "prompt":   "User is LONELY. In 1-2 short sentences, warmly explain how these movies celebrate friendship and connection.",
    },
    "tired": {
        "greeting": "You need a rest. 😴 No heavy stuff — just easy, feel-good watching.",
        "closing":  "Get comfortable and just enjoy. No thinking required. 🌙",
        "prompt":   "User is TIRED. In 1 short sentence, explain how effortlessly relaxing these movies are.",
    },
    "bored": {
        "greeting": "Boredom? Not on my watch! 😄 These picks will hook you in the first five minutes.",
        "closing":  "You will NOT be bored. Pick one and thank me later. 😎",
        "prompt":   "User is BORED. In 1-2 short sentences, make these movies sound completely irresistible.",
    },
    "romantic": {
        "greeting": "Feeling the love? 💕 These beautiful films are perfect for the mood.",
        "closing":  "Enjoy every moment. 🌹",
        "prompt":   "User is ROMANTIC. In 1-2 short sentences, warmly describe why these love stories are perfect right now.",
    },
    "fear": {
        "greeting": "No scary stuff tonight! 🛡️ These fun, safe films will keep you comfortable.",
        "closing":  "You're safe — all feel-good picks! 😊",
        "prompt":   "User is SCARED. In 1-2 short sentences, reassuringly explain how safe and fun these movies are.",
    },
    "stressed": {
        "greeting": "You need a break — you deserve it! 🌈 These films will help you switch off completely.",
        "closing":  "Step away from everything. These are your reset button. ✨",
        "prompt":   "User is STRESSED. In 1-2 short sentences, explain how effortless and refreshing these movies are.",
    },
    "surprise": {
        "greeting": "Caught off guard? 🤔 Here are some films full of twists you won't see coming!",
        "closing":  "Prepare to have your mind blown. 🌀",
        "prompt":   "User is SURPRISED. In 1-2 short sentences, excitedly describe the mystery and unexpected twists in these movies.",
    },
    "disgust": {
        "greeting": "Need something fun and wholesome? 😄 These picks will brighten your day.",
        "closing":  "Pure fun, pure laughs. 🎭",
        "prompt":   "User is feeling DISGUST. In 1-2 short sentences, cheerfully explain how these fun movies will lift them up.",
    },
    "neutral": {
        "greeting": "Not sure what you're in the mood for? 🎬 Here are some great all-rounders.",
        "closing":  "Any of these would be a great watch. Enjoy! 🍿",
        "prompt":   "User has NEUTRAL mood. In 1-2 short sentences, conversationally recommend these movies.",
    },
}

DEFAULT_PERSONALITY = MOOD_PERSONALITY["neutral"]


# ---------------------------------------------------
# Instant rule-based explanation (< 1ms, no LLM)
# ---------------------------------------------------
def _instant_explanation(emotion: str, movies: List[dict]) -> str:

    personality = MOOD_PERSONALITY.get(emotion, DEFAULT_PERSONALITY)
    top         = movies[0] if movies else {}
    title       = top.get("title", "these films")
    genres      = top.get("genres", "").replace("|", ", ")

    mood_desc = {
        "sad":      "uplifting and feel-good",
        "angry":    "calming and light-hearted",
        "happy":    "exciting and thrilling",
        "excited":  "epic and action-packed",
        "anxious":  "comforting and easy",
        "lonely":   "warm and full of heart",
        "tired":    "relaxing and effortless",
        "bored":    "fast-paced and gripping",
        "romantic": "beautiful and heartfelt",
        "fear":     "safe and fun",
        "stressed": "light and refreshing",
        "neutral":  "well-crafted and engaging",
    }.get(emotion, "perfectly matched")

    return (
        f"{title} is a top pick — a {genres} film that is {mood_desc}. "
        f"All recommendations were selected using semantic AI search across 62,000 movies "
        f"specifically to help with your current mood."
    )


# ---------------------------------------------------
# Fast LLM call with hard timeout
# ---------------------------------------------------
def _fast_llm_call(prompt: str) -> Optional[str]:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  prompt,
                "stream":  False,
                "options": {
                    "temperature":  TEMPERATURE,
                    "num_predict":  MAX_TOKENS,   # ⚡ 60 tokens max
                    "num_ctx":      512,           # ⚡ smaller context window
                    "top_k":        10,            # ⚡ faster sampling
                }
            },
            timeout=TIMEOUT_SEC    # ⚡ hard 8s cutoff
        )

        if response.status_code != 200:
            return None

        result = response.json().get("response", "").strip()
        return result if result else None

    except requests.exceptions.Timeout:
        print(f"[LLMExplainer] Timed out after {TIMEOUT_SEC}s — using instant response")
        return None
    except Exception as e:
        print(f"[LLMExplainer] Error: {e}")
        return None


# ---------------------------------------------------
# Build short, focused prompt
# ---------------------------------------------------
def _build_prompt(query: str, emotion: str, movies: List[dict]) -> str:

    personality = MOOD_PERSONALITY.get(emotion, DEFAULT_PERSONALITY)

    # Only top 3 movies, short format
    movie_lines = "\n".join([
        f"- {m.get('title','?')} ({m.get('genres','?')})"
        for m in movies[:3]
    ])

    return (
        f"{personality['prompt']}\n\n"
        f"Movies: {movie_lines}\n"
        f"{'Query: ' + query if query else ''}\n\n"
        f"Reply in max 2 sentences. No bullet points."
    )


# ---------------------------------------------------
# Check Ollama (fast, 1s timeout)
# ---------------------------------------------------
def _ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------
# Main entry point — returns INSTANTLY
# ---------------------------------------------------
def explain_recommendations(
    query:   str,
    emotion: str,
    movies:  List[dict]
) -> dict:
    """
    Returns mood-aware response INSTANTLY.
    - greeting + closing are pre-written (0ms)
    - explanation tries LLM with 8s hard timeout,
      falls back to instant rule-based if too slow

    Returns:
    {
        "greeting":    str
        "explanation": str
        "closing":     str
    }
    """

    if not movies:
        return {
            "greeting":    "Hmm, couldn't find anything right now.",
            "explanation": "Try a different search or mood.",
            "closing":     "I'll do better next time! 🎬"
        }

    emotion_key = (emotion or "neutral").lower().strip()
    personality = MOOD_PERSONALITY.get(emotion_key, DEFAULT_PERSONALITY)

    # ── Greeting + closing are always instant ────────────────────────────────
    greeting = personality["greeting"]
    closing  = personality["closing"]

    # ── Try LLM with hard timeout, else use instant fallback ─────────────────
    explanation = None

    if _ollama_available():
        prompt      = _build_prompt(query, emotion_key, movies)
        explanation = _fast_llm_call(prompt)

    if not explanation:
        explanation = _instant_explanation(emotion_key, movies)

    return {
        "greeting":    greeting,
        "explanation": explanation,
        "closing":     closing,
    }


# ---------------------------------------------------
# Optional: async background LLM upgrade
# Call this after returning instant response to
# progressively update the UI with a better explanation
# ---------------------------------------------------
def explain_async(query: str, emotion: str, movies: List[dict], callback) -> None:
    """
    Runs LLM in background thread and calls callback(result_dict) when done.
    Use this in main.py if you want to stream the improved explanation later.

    Example:
        explain_async(query, emotion, movies, lambda r: send_to_websocket(r))
    """
    def _run():
        result = explain_recommendations(query, emotion, movies)
        callback(result)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()