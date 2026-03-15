import pandas as pd
import pickle
import os
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

print("Loading movie dataset...")

movies = pd.read_csv("data/raw/movies.csv")

print("Dataset columns:", movies.columns)

# Keep only necessary columns
movies = movies[["title", "genres"]]

movies = movies.dropna()

print(f"Total movies loaded: {len(movies)}")


# ---------------------------------------------------
# Create text for embeddings
# ---------------------------------------------------

print("Preparing text data for embeddings...")

texts = (movies["title"] + " " + movies["genres"]).tolist()


# ---------------------------------------------------
# Load Sentence Transformer
# ---------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully.")


# ---------------------------------------------------
# Generate Embeddings
# ---------------------------------------------------

print("Generating movie embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=64
)

print("Embeddings generated.")
print("Embedding shape:", embeddings.shape)


# ---------------------------------------------------
# Build FAISS Index
# ---------------------------------------------------

print("Building FAISS vector index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"Total vectors indexed: {index.ntotal}")


# ---------------------------------------------------
# Save Models
# ---------------------------------------------------

print("Saving embeddings and FAISS index...")

os.makedirs("models", exist_ok=True)

with open("models/movie_embeddings.pkl", "wb") as f:
    pickle.dump(
        {
            "movies": movies,
            "embeddings": embeddings
        },
        f
    )

faiss.write_index(index, "models/movie_faiss.index")

print("Files saved successfully!")

print("Created files:")
print("models/movie_embeddings.pkl")
print("models/movie_faiss.index")