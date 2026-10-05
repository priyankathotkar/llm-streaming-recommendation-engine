# 🎬 CineAI — Emotion-Aware AI Movie Recommendation Engine

> **"Don't just match the mood — improve it."**
> CineAI detects your facial emotion in real-time and recommends movies
> specifically chosen to make you feel *better*, not keep you stuck.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** Priyanka Thotkar
**GitHub:** [@priyankathotkar](https://github.com/priyankathotkar)

---

## Table of Contents

- [Overview](#overview)
- [The Core Idea](#the-core-idea)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [File-by-File Breakdown](#file-by-file-breakdown)
- [Dataset](#dataset)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Key Design Decisions](#key-design-decisions)
- [Attribution](#attribution)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)

---

## Overview

CineAI is a full-stack AI movie recommendation system that combines:

- **Computer Vision** — real-time facial emotion detection via MediaPipe face mesh + HuggingFace transformer
- **Semantic Search** — 384-dimensional sentence embeddings searched via FAISS across 62,000+ movies
- **Opposite-Mood Logic** — maps detected emotion to counter-mood genres (sad → comedy, angry → calm)
- **LLM Explanations** — Ollama Llama3 generates personalised, mood-aware explanations for every recommendation
- **Preference Drift Detection** — LightGBM with Stratified K-Fold CV tracks when your taste changes over time

---

## The Core Idea

Most recommendation systems ask *"what matches how you feel right now?"*

CineAI asks *"what would make you feel better?"*

If you are **sad**, the system searches for uplifting comedies.
If you are **angry**, it finds calming, light-hearted films.
If you are **anxious**, it returns easy, comforting animations.

This **opposite-mood mapping** is the central design decision of the project
and is implemented in `recommender.py` via `EMOTION_GENRE_MAP` and `EMOTION_QUERY_MAP`.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│         (index.html — MediaPipe + Chart.js)             │
└───────────────────────┬─────────────────────────────────┘
                        │  HTTP POST /recommend
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend (fastapi_app.py)               │
└──┬──────────────┬──────────────┬──────────────┬─────────┘
   │              │              │              │
   ▼              ▼              ▼              ▼
emotion_      query_        recommender.  llm_
detector.py   parser.py     py            explainer.py
   │              │              │              │
HuggingFace   Ollama /      FAISS +       Ollama
Transformer   Rules         SentenceT.    Llama3
                                │
                         drift_detector.py
                         (LightGBM CV)
                                │
                         TMDB API (posters)
                                │
                         Frontend Display
```

---

## Project Structure

```
cineai/
│
├── fastapi_app.py                      # FastAPI backend — all API endpoints
├── recommender.py               # Core ML: FAISS search + opposite-mood logic
├── llm_explainer.py             # Ollama Llama3 mood-aware explanations
├── query_parser.py              # NLP query parsing (LLM → rules fallback)
├── drift_detector.py            # LightGBM preference drift detection
│
├── src/
│   └── features/
│       └── emotion_detector.py  # HuggingFace facial emotion classifier
│
├── models/
│   ├── movie_embeddings.pkl     # Pre-computed 384-dim movie embeddings
│   └── movie_faiss.index        # FAISS flat index (62,000 movies)
│
├── UI/
│   └── src/
│       └── index.html           # Frontend (MediaPipe, Chart.js, TMDB API)
│
├── requirements.txt             # All Python dependencies
└── README.md
```

---

## File-by-File Breakdown

### `recommender.py` — Core ML Module
The most important file. Contains the full recommendation pipeline.

**What it does:**
- Loads pre-computed movie embeddings and a FAISS flat index at startup
- Encodes user queries using `all-MiniLM-L6-v2` (384-dim sentence transformer)
- Applies **opposite-mood genre mapping** — detected emotion → counter-mood genres
- Runs **semantic FAISS search** using the emotion's natural-language query string
- Supplements results with **genre-centroid search** (mean embedding of all genre movies)
- Returns ranked results via a **5-level fallback chain** — always returns something

**Key classes/functions:**
| Name | Purpose |
|------|---------|
| `EMOTION_GENRE_MAP` | Maps 14 emotions to counter-mood genre lists |
| `EMOTION_QUERY_MAP` | Natural language queries per emotion for semantic search |
| `MovieRecommender._semantic_search()` | Encodes text → FAISS nearest-neighbour lookup |
| `MovieRecommender.recommend_by_genre()` | Genre-centroid based FAISS retrieval |
| `MovieRecommender.recommend_by_emotion()` | Full opposite-mood pipeline |
| `MovieRecommender.recommend()` | Unified entry point with fallback chain |

---

### `emotion_detector.py` — Facial Emotion Classification
Detects the user's dominant emotion from an image or video frame.

**What it does:**
- Accepts image bytes or an OpenCV frame
- Runs inference using `dima806/facial_emotions_image_detection` from HuggingFace
- Returns 7 emotion classes: happy, sad, angry, fear, disgust, surprise, neutral
- Falls back to `neutral` on any error to prevent pipeline failures

**Key functions:**
| Name | Purpose |
|------|---------|
| `detect_emotion(image_bytes)` | Single image inference — returns emotion + confidence |
| `EmotionDetector.detect_emotion_from_frame(frame)` | OpenCV frame inference using DeepFace |

---

### `llm_explainer.py` — Mood-Aware LLM Explanations
Generates personalised natural language explanations for recommendations.

**What it does:**
- Maintains 13 distinct **mood personalities** — each emotion has a unique tone,
  greeting, closing message, and LLM prompt instruction
- Each mood has **multiple greeting/closing variations** picked randomly — so responses
  never feel repetitive
- Calls **Ollama Llama3** locally with an 8-second hard timeout and 80-token cap
- Falls back to instant rule-based explanation if Ollama is unavailable

**Key design:**
- `MOOD_PERSONALITY` dict — 13 emotions × {greetings[], closings[], prompt}
- Returns `{greeting, explanation, closing}` dict — shown separately in the UI
- Temperature set to `0.85` for natural, varied output

---

### `query_parser.py` — NLP Query Parser
Extracts structured intent from free-text user input.

**What it does:**
- Parses user queries to extract: movie titles, genres, and emotional keywords
- Priority: **Ollama Llama3** → **OpenAI GPT** → **rule-based keyword fallback**
- Handles aliases (`rom-com` → `romance`, `sci-fi` → `science fiction`)
- Fuzzy movie title matching with cutoff `0.6` to avoid false positives
- Normalises all genres to lowercase canonical form

**Output format:**
```python
{
    "reference_movies": ["Inception", "Interstellar"],
    "genres":           ["sci-fi", "thriller"],
    "emotion":          "excited"
}
```

---

### `drift_detector.py` — Preference Drift Detection
Detects when a user's movie taste changes significantly over time using
machine learning — not just a simple rule.

**What it does:**
- Encodes each movie interaction as a **one-hot genre vector** (15 genres:
  Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Fantasy,
  Horror, Mystery, Romance, Sci-Fi, Thriller, War, Western)
- Automatically retrains every 10 new interactions once 30+ events exist

**Cross-Validation Pipeline (the core ML piece):**
```
Interaction history (30+ events)
           │
           ▼
  Label split: first half → 0 (old behaviour)
               second half → 1 (new behaviour)
           │
           ▼
  StandardScaler normalisation
           │
           ▼
  Stratified K-Fold CV (5 folds, shuffle=True)
  ┌─────────────────────────────────────┐
  │  For each fold:                     │
  │  - Train LGBMClassifier             │
  │  - Predict probabilities on val set │
  │  - Compute ROC-AUC score            │
  │  - Log: "Fold N AUC: X.XXXX"       │
  └─────────────────────────────────────┘
           │
           ▼
  Mean CV AUC reported
           │
           ▼
  Final model trained on full data
           │
           ▼
  Score last 10 interactions →
  drift_score = mean(predict_proba[:, 1])
           │
     drift_score > 0.65?
        YES → drift detected → adjust recommendations
        NO  → preferences stable
```

**Why Stratified K-Fold specifically:**
With small interaction histories the class balance between old/new behaviour
can be uneven. Stratified splitting ensures each fold maintains the original
class ratio — giving reliable AUC estimates even with as few as 30 events.

**LightGBM hyperparameters used:**
| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_estimators` | 100 | Enough trees without overfitting sparse vectors |
| `learning_rate` | 0.05 | Conservative — avoids overfitting small datasets |
| `max_depth` | 4 | Shallow trees for 15-feature one-hot input |
| `num_leaves` | 15 | Consistent with max_depth=4 |
| `min_child_samples` | 5 | Prevents splits on tiny genre subsets |
| `subsample` | 0.8 | Row sampling for regularisation |
| `colsample_bytree` | 0.8 | Feature sampling per tree |

**Key methods:**
| Name | Purpose |
|------|---------|
| `genre_to_vector(genre_string)` | Convert genre string to 15-dim one-hot vector |
| `add_event(genre)` | Add interaction, auto-retrain every 10 events |
| `train()` | Run Stratified K-Fold CV + train final model |
| `detect_drift()` | Score recent 10 interactions, return bool |
| `get_feature_importances()` | Which genres are driving the drift |
| `get_report()` | Full report: AUC per fold, mean AUC, drift status, top genres |

**Why LightGBM over simpler approaches:**
A basic mean-vector distance (like cosine drift) cannot distinguish *which*
genres are changing or whether the change is statistically meaningful.
LightGBM gives a probability score, per-feature importances, and cross-validated
confidence — making the drift signal interpretable and actionable.

---

### `index.html` — Frontend
Single-file frontend with no framework dependencies.

**Features:**
- **MediaPipe Face Mesh** — live face mesh overlay with 468 landmarks drawn on camera feed
- **Mood + Genre chips** — click to select your mood or genre with live hint text
- **Camera / Upload toggle** — switch between live camera and image upload; camera stops automatically when upload is selected
- **Skeleton cards** — movie cards appear instantly with animated placeholders; posters load in background via `Promise.all` and fade in
- **Profile dashboard** — 5-tab user profile: Genre Taste, Mood History, Preference Drift, Sessions, Taste Radar
- **LLM box** — displays greeting, AI explanation, and closing message separately, each styled differently

---

## Dataset

**62,000+ movies** — custom scraped dataset combining:
- Movie metadata (title, genres, overview, popularity)
- Pre-computed 384-dimensional sentence embeddings using `all-MiniLM-L6-v2`
  applied to each movie's title + overview + genre string
- FAISS flat index built from normalised embeddings for cosine similarity search

**Note:** The raw dataset and pre-computed model files (`movie_embeddings.pkl`,
`movie_faiss.index`) are not included in this repository due to file size.
To use the system you will need to generate them — see [Installation](#installation--setup).

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally
- Node.js (only needed if regenerating embeddings)

### 1. Clone the repository

```bash
git clone https://github.com/priyankathotkar/cineai.git
cd cineai
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the LLM model

```bash
ollama pull llama3
```

### 5. Generate embeddings (first time only)

```bash
python scripts/build_embeddings.py
```

This will create `models/movie_embeddings.pkl` and `models/movie_faiss.index`.

---

## How to Run

### Start the backend

```bash
uvicorn fastapi_app:app --reload --port 8000
```

### Start the frontend

```bash
python -m http.server 8080 --directory UI/src
```

### Open in browser

```
http://localhost:8080/index.html
```

### API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check |
| `POST` | `/recommend` | Get movie recommendations |
| `POST` | `/detect_emotion` | Detect emotion from image |

**Example request:**
```bash
curl -X POST http://localhost:8000/recommend \
  -F "query=something to cheer me up" \
  -F "image=@my_photo.jpg"
```

**Example response:**
```json
{
  "detected_emotion": "sad",
  "recommendations": [
    {"title": "The Grand Budapest Hotel", "genres": "Comedy|Drama", "score": 0.921},
    {"title": "Paddington 2",            "genres": "Comedy|Family", "score": 0.908}
  ],
  "llm_explanation": {
    "greeting":    "Rough day? I've got you. These are guaranteed to make you smile.",
    "explanation": "The Grand Budapest Hotel is a perfect pick — it's witty, visually gorgeous, and impossible not to smile at.",
    "closing":     "Things will look up. Until then — movies help. I promise."
  }
}
```

---

## Key Design Decisions

### 1. Opposite-mood logic over mood-matching
Traditional recommenders reinforce your current state. CineAI counters it.
A sad user gets comedy. An angry user gets calm, light-hearted films.
This was the founding hypothesis of the project.

### 2. Semantic search over keyword filtering
Instead of filtering movies by genre tag, each query is encoded into a
384-dimensional embedding and searched against the full movie corpus via
FAISS cosine similarity. This captures thematic nuance — a "calming movie"
query finds films that *feel* calming regardless of genre label.

### 3. Genre-centroid retrieval
The mean embedding of all movies in a genre serves as a FAISS query vector.
This finds movies that are *semantically central* to a genre, not just tagged with it.

### 4. LightGBM + Stratified K-Fold for drift detection
Genre interactions are encoded as 15-dimensional one-hot vectors. A LightGBM
classifier is trained to distinguish old behaviour (label 0, first half of
history) from new behaviour (label 1, second half).

Stratified K-Fold cross-validation (5 folds) is used rather than a simple
train/test split because interaction histories are small and class balance
is often uneven. Stratification preserves the class ratio in each fold,
giving reliable per-fold AUC scores and a trustworthy mean CV AUC.

After CV, a final model is trained on the full history. The last 10
interactions are scored — if `mean(predict_proba[:, 1]) > 0.65`, drift
is flagged and the recommendation system adapts. Feature importances
from the trained model identify *which* genres are responsible for the
detected shift, making the output interpretable.

### 5. 5-level fallback chain
Every search path has a fallback. The system never returns an empty response:
text query → emotion search → genre search → emotion keyword → dataset top rows.

---

## Attribution

**All architecture, system design, and core ideas were conceived and developed
solely by Priyanka Thotkar.**

Specifically, the following are original contributions:
- Opposite-mood emotion-to-genre mapping concept
- Overall system architecture and pipeline design
- FAISS genre-centroid retrieval strategy
- LightGBM preference drift detection approach
- 5-level fallback chain design



**Open-source libraries used** (all properly licensed):

| Library | Purpose | License |
|---------|---------|---------|
| [FAISS](https://github.com/facebookresearch/faiss) | Vector similarity search | MIT |
| [sentence-transformers](https://www.sbert.net) | Sentence embeddings | Apache 2.0 |
| [HuggingFace Transformers](https://huggingface.co) | Emotion classification | Apache 2.0 |
| [FastAPI](https://fastapi.tiangolo.com) | Backend API framework | MIT |
| [LightGBM](https://lightgbm.readthedocs.io) | Drift detection classifier | MIT |
| [MediaPipe](https://mediapipe.dev) | Face mesh overlay | Apache 2.0 |
| [Ollama](https://ollama.ai) | Local LLM inference | MIT |
| [Chart.js](https://www.chartjs.org) | Frontend visualisations | MIT |
| [TMDB API](https://www.themoviedb.org/documentation/api) | Movie poster retrieval | Terms of Service |

---

## Future Improvements

### Short-term
- [ ] **Collaborative filtering** — combine embedding similarity with user-user similarity signals
- [ ] **Persistent user profiles** — store session history in a database so preferences survive page refresh
- [ ] **Better emotion model** — fine-tune on a larger facial expression dataset for higher accuracy
- [ ] **Streaming LLM responses** — stream Llama3 tokens to the frontend for faster perceived response time

### Medium-term
- [ ] **Reinforcement learning from feedback** — learn from explicit thumbs up/down on recommendations
- [ ] **Multi-modal embeddings** — combine movie poster image embeddings with text embeddings for richer retrieval
- [ ] **IVF FAISS index** — replace flat index with IVF for sub-linear search time at larger scale
- [ ] **Streaming service integration** — surface which platform (Netflix, Prime, Disney+) each movie is on

### Long-term
- [ ] **Conversational recommendations** — multi-turn chat interface powered by Llama3
- [ ] **Real-time drift adaptation** — automatically adjust recommendations mid-session as drift is detected
- [ ] **Multi-user support** — user authentication, cross-user collaborative signals
- [ ] **Mobile app** — React Native wrapper around the existing API

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with clear commit messages
4. Add or update tests if applicable
5. Open a pull request with a clear description of what you changed and why

For major changes please open an issue first to discuss the approach.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [Priyanka Thotkar](https://github.com/priyankathotkar)*
