import os
import warnings
import pickle
import faiss
import numpy as np
import requests

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from src.features.hf_emotion import detect_emotion


# ---------------------------------------------------
# Suppress warnings
# ---------------------------------------------------

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------

app = FastAPI(title="AI Movie Recommendation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
# Load Movie Dataset
# ---------------------------------------------------

print("Loading movie data...")

MODEL_PATH = os.path.join("models", "movie_embeddings.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("movie_embeddings.pkl not found")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

movies = data["movies"]

movies["genres"] = movies["genres"].fillna("")
movies["genres"] = movies["genres"].replace("(no genres listed)", "")

print(f"Loaded {len(movies)} movies")


# ---------------------------------------------------
# Load FAISS Index
# ---------------------------------------------------

print("Loading FAISS index...")

FAISS_PATH = os.path.join("models", "movie_faiss.index")

if not os.path.exists(FAISS_PATH):
    raise RuntimeError("movie_faiss.index not found")

index = faiss.read_index(FAISS_PATH)

print("FAISS index loaded successfully")


# ---------------------------------------------------
# Load Sentence Transformer
# ---------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model ready")


# ---------------------------------------------------
# Drift Detection
# ---------------------------------------------------

user_history = []
drift_threshold = 0.30


def detect_drift(new_vector):

    global user_history

    user_history.append(new_vector)

    # keep memory limited
    if len(user_history) > 100:
        user_history.pop(0)

    if len(user_history) < 20:
        return False

    past = np.array(user_history[:10])
    recent = np.array(user_history[-10:])

    diff = np.mean(np.abs(past.mean(axis=0) - recent.mean(axis=0)))

    return bool(diff > drift_threshold)


# ---------------------------------------------------
# Mood Adjustment
# ---------------------------------------------------

def adjust_query_for_mood(query, emotion):

    mood_map = {
        "sad": "uplifting comedy feel good",
        "angry": "comedy action stress relief",
        "fear": "adventure fantasy inspiring",
        "disgust": "comedy feel good",
        "neutral": "popular movies",
        "happy": ""
    }

    query = query.strip()

    if emotion in mood_map:

        mood = mood_map[emotion]

        if query == "":
            return mood

        return f"{query} {mood}".strip()

    return query


# ---------------------------------------------------
# LLM Explanation
# ---------------------------------------------------

def explain_recommendations(query, emotion, movies_list):

    if not movies_list:
        return ""

    movie_titles = "\n".join(
        [f"{m['title']} ({m['genres']})" for m in movies_list[:5]]
    )

    prompt = f"""
User emotion: {emotion}
User query: {query}

Recommended movies:
{movie_titles}

Explain briefly why these movies match the user's mood.
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        if response.status_code != 200:
            return ""

        result = response.json()

        return result.get("response", "").strip()

    except Exception as e:

        print("LLM error:", e)
        return ""


# ---------------------------------------------------
# Home Route
# ---------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "AI Movie Recommendation API running",
        "movies_loaded": int(len(movies))
    }


# ---------------------------------------------------
# Emotion Detection Endpoint
# ---------------------------------------------------

@app.post("/detect_emotion")
async def detect_emotion_api(image: UploadFile = File(...)):

    try:

        image_bytes = await image.read()

        result = detect_emotion(image_bytes)

        return result

    except Exception:

        return {"emotion": "neutral"}


# ---------------------------------------------------
# Recommendation Endpoint
# ---------------------------------------------------

@app.post("/recommend")
async def recommend_movies(
    query: str = Form(""),
    image: UploadFile = File(None)
):

    emotion = "neutral"

    # Emotion Detection
    if image is not None:

        try:

            image_bytes = await image.read()

            emotion_result = detect_emotion(image_bytes)

            emotion = str(emotion_result.get("emotion", "neutral"))

        except Exception:

            emotion = "neutral"

    # Mood Adjustment
    query = query.strip()

    final_query = adjust_query_for_mood(query, emotion)

    # Generate Embedding
    query_embedding = model.encode([final_query])

    # Normalize embedding for FAISS cosine similarity
    faiss.normalize_L2(query_embedding)

    # Drift Detection
    drift = detect_drift(query_embedding[0])

    # FAISS Search
    TOP_K = 20
    distances, indices = index.search(query_embedding, TOP_K)

    results = []
    seen = set()

    for i, idx in enumerate(indices[0]):

        if idx < 0 or idx >= len(movies):
            continue

        title = str(movies.iloc[idx]["title"])

        if title in seen:
            continue

        seen.add(title)

        genre_value = movies.iloc[idx]["genres"] or ""

        score = float(1 - distances[0][i])

        results.append({
            "title": title,
            "genres": str(genre_value),
            "score": score
        })

    # LLM Explanation
    explanation = explain_recommendations(
        final_query,
        emotion,
        results
    )

    # API Response
    return {

        "detected_emotion": str(emotion),
        "query_used": str(final_query),
        "drift_detected": bool(drift),
        "total_results": int(len(results)),
        "recommendations": results,
        "llm_explanation": str(explanation)

    }