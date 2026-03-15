# src/faiss_index_builder.py

import os
import pickle
import numpy as np
import faiss
from pathlib import Path


class FAISSIndexBuilder:
    """
    Builds a FAISS index from pre-generated movie embeddings.

    Why FAISS?
    ----------
    With 2M+ movies, brute-force cosine similarity (sklearn) would
    compare every query against every movie — O(N) per query.
    FAISS uses Approximate Nearest Neighbour (ANN) algorithms to
    reduce this to O(log N), making real-time search feasible.

    This is exactly how Spotify, Netflix, and YouTube serve
    recommendations at scale.
    """

    def __init__(
        self,
        embeddings_pkl: str = "models/movie_embeddings.pkl",
        index_save_path: str = "models/faiss_index.index",
        use_gpu: bool = False,
    ):
        self.embeddings_pkl = Path(embeddings_pkl)
        self.index_save_path = Path(index_save_path)
        self.use_gpu = use_gpu

        self.embeddings: np.ndarray | None = None
        self.movies = None
        self.index: faiss.Index | None = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load_embeddings(self) -> None:
        """Load embeddings and movie metadata from pickle file."""
        if not self.embeddings_pkl.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {self.embeddings_pkl}\n"
                "Run src/embedding_generator.py first."
            )

        print(f"Loading embeddings from {self.embeddings_pkl} ...")

        with open(self.embeddings_pkl, "rb") as f:
            data = pickle.load(f)

        self.movies = data["movies"]
        raw_embeddings = data["embeddings"]

        # ── Validate ───────────────────────────────────────────────────
        if len(raw_embeddings) != len(self.movies):
            raise ValueError(
                f"Mismatch: {len(raw_embeddings)} embeddings "
                f"but {len(self.movies)} movies. "
                "Re-run embedding_generator.py."
            )

        if raw_embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embeddings, got shape {raw_embeddings.shape}"
            )

        # FAISS requires float32
        self.embeddings = raw_embeddings.astype(np.float32)

        print(
            f"Loaded {len(self.movies):,} movies | "
            f"Embedding dim: {self.embeddings.shape[1]}"
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build_index(self) -> None:
        """
        Build FAISS index.

        Index type selection:
        ─────────────────────
        - Under 100K movies  → IndexFlatIP  (exact, fast enough)
        - 100K – 1M movies   → IndexIVFFlat (approximate, much faster)
        - Over 1M movies     → IndexIVFPQ   (compressed, production-grade)

        Since embeddings are already L2-normalised (from generator),
        inner product (IP) == cosine similarity.
        """
        if self.embeddings is None:
            raise RuntimeError("Call load_embeddings() first.")

        n_vectors, dim = self.embeddings.shape
        print(f"\nBuilding FAISS index for {n_vectors:,} vectors of dim {dim}...")

        if n_vectors < 100_000:
            # ── Exact search ──────────────────────────────────────────
            print("Using IndexFlatIP (exact cosine search)")
            self.index = faiss.IndexFlatIP(dim)

        elif n_vectors < 1_000_000:
            # ── Approximate search with IVF ───────────────────────────
            n_clusters = min(4096, n_vectors // 39)
            print(f"Using IndexIVFFlat with {n_clusters} clusters")

            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT
            )

            print("Training IVF index (this takes a few minutes)...")
            self.index.train(self.embeddings)

            # How many clusters to visit at query time
            # Higher = more accurate but slower
            self.index.nprobe = 64

        else:
            # ── Production-grade compressed index ─────────────────────
            # PQ compresses each vector into 64 bytes
            # Reduces memory from ~3GB to ~128MB for 2M movies
            n_clusters = 8192
            pq_segments = 64   # must divide dim evenly
            print(
                f"Using IndexIVFPQ with {n_clusters} clusters, "
                f"{pq_segments} PQ segments"
            )

            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFPQ(
                quantizer, dim, n_clusters, pq_segments, 8
            )

            print("Training IVFPQ index (this may take 10-20 minutes)...")
            # Train on a sample for speed — 500K is enough
            sample_size = min(500_000, n_vectors)
            sample_idx = np.random.choice(n_vectors, sample_size, replace=False)
            self.index.train(self.embeddings[sample_idx])
            self.index.nprobe = 128

        # ── GPU acceleration (optional) ───────────────────────────────
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("Moving index to GPU...")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # ── Add all vectors ───────────────────────────────────────────
        print("Adding vectors to index...")
        # Add in chunks to show progress
        chunk_size = 100_000
        for start in range(0, n_vectors, chunk_size):
            end = min(start + chunk_size, n_vectors)
            self.index.add(self.embeddings[start:end])
            print(f"  Added {end:,} / {n_vectors:,} vectors", end="\r")

        print(f"\nIndex built with {self.index.ntotal:,} vectors.")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save_index(self) -> None:
        """Save FAISS index to disk."""
        if self.index is None:
            raise RuntimeError("Call build_index() first.")

        index_dir = self.index_save_path.parent
        if index_dir:
            index_dir.mkdir(parents=True, exist_ok=True)

        # If on GPU, move back to CPU before saving
        if self.use_gpu and faiss.get_num_gpus() > 0:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(self.index_save_path))
        else:
            faiss.write_index(self.index, str(self.index_save_path))

        print(f"FAISS index saved to: {self.index_save_path}")

    @classmethod
    def load_index(cls, index_path: str = "models/faiss_index.index") -> faiss.Index:
        """Load a previously saved FAISS index."""
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {path}\n"
                "Run src/faiss_index_builder.py first."
            )
        index = faiss.read_index(str(path))