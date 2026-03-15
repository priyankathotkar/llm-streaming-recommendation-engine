import numpy as np
from src.ml.drift_detector import DriftDetector

class UserProfile:
    """
    Tracks user embeddings, genre preferences, and interaction history.
    Integrates DriftDetector (LightGBM + Cross-Validation) to automatically
    detect when user taste has shifted over time.
    """

    def __init__(self):

        # embedding history (one per recommendation session)
        self.embeddings = []

        # raw genre strings per interaction e.g. "Action|Drama"
        self.genre_history = []

        # drift detector — LightGBM + CV
        self.drift_detector = DriftDetector(n_splits=5, drift_threshold=0.65)

        # flag set when drift is detected
        self.drift_detected = False

        # stores last drift report for inspection
        self.last_drift_report = {}


    # ---------------------------------------------------
    # Add a new interaction
    # ---------------------------------------------------

    def add_interaction(self, embedding, genres):
        """
        Records embedding + genres from a recommendation session.
        Automatically feeds genre data into the drift detector.

        Args:
            embedding : np.ndarray  — movie/query embedding vector
            genres    : str         — pipe-separated genres e.g. "Action|Drama"
        """

        # store embedding
        self.embeddings.append(embedding)

        # store genre string
        if genres:
            self.genre_history.append(genres)

            # feed each genre individually into drift detector
            for genre in genres.split("|"):
                genre = genre.strip()
                if genre:
                    self.drift_detector.add_event(genre)

        # check for drift after every interaction
        self._check_drift()


    # ---------------------------------------------------
    # Internal drift check
    # ---------------------------------------------------

    def _check_drift(self):
        """
        Runs drift detection and updates self.drift_detected.
        Only runs when detector has enough history (>= 30 events).
        """

        if len(self.drift_detector.history) >= 30:
            self.drift_detected    = self.drift_detector.detect_drift()
            self.last_drift_report = self.drift_detector.get_report()

            if self.drift_detected:
                print("[UserProfile] Drift detected — user preferences have shifted.")


    # ---------------------------------------------------
    # Get mean user embedding
    # ---------------------------------------------------

    def get_user_embedding(self):
        """
        Returns the mean of all stored embeddings as the user's
        current preference vector. Returns None if no history.
        """

        if len(self.embeddings) == 0:
            return None

        return np.mean(self.embeddings, axis=0)


    # ---------------------------------------------------
    # Get genre preferences ranked by frequency
    # ---------------------------------------------------

    def get_preferences(self):
        """
        Returns genres ranked by interaction frequency (most to least).
        If drift was detected, returns only preferences from the
        second half of history to reflect updated taste.
        """

        # if drift detected, use only recent half of history
        if self.drift_detected and len(self.genre_history) >= 10:
            recent_history = self.genre_history[len(self.genre_history) // 2:]
        else:
            recent_history = self.genre_history

        counts = {}

        for g in recent_history:
            for genre in g.split("|"):
                genre = genre.strip()
                if genre:
                    counts[genre] = counts.get(genre, 0) + 1

        return sorted(counts, key=counts.get, reverse=True)


    # ---------------------------------------------------
    # Get genre counts dict (for charts/API response)
    # ---------------------------------------------------

    def get_genre_counts(self):
        """
        Returns dict of genre -> count for visualization.
        Reflects drift-adjusted history if drift was detected.
        """

        history = (
            self.genre_history[len(self.genre_history) // 2:]
            if self.drift_detected and len(self.genre_history) >= 10
            else self.genre_history
        )

        counts = {}

        for g in history:
            for genre in g.split("|"):
                genre = genre.strip()
                if genre:
                    counts[genre] = counts.get(genre, 0) + 1

        return counts


    # ---------------------------------------------------
    # Get drift status summary (for API / frontend)
    # ---------------------------------------------------

    def get_drift_status(self):
        """
        Returns a human-readable drift summary dict.
        Includes CV AUC scores and top drift-driving genres.
        """

        report = self.last_drift_report

        return {
            "drift_detected":   self.drift_detected,
            "total_events":     report.get("total_events", 0),
            "mean_cv_auc":      report.get("mean_cv_auc"),
            "cv_fold_scores":   report.get("cv_fold_scores", []),
            "drift_threshold":  report.get("drift_threshold", 0.65),
            "top_drift_genres": report.get("top_drift_genres", {})
        }


    # ---------------------------------------------------
    # Summary string
    # ---------------------------------------------------

    def __repr__(self):

        return (
            f"UserProfile("
            f"interactions={len(self.embeddings)}, "
            f"genres_tracked={len(self.genre_history)}, "
            f"drift={self.drift_detected})"
        )