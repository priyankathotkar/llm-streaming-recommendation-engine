# src/features/emotion_detector.py

from deepface import DeepFace
import cv2
from typing import Optional

class EmotionDetector:
    """
    Detect human facial emotion from image frames using DeepFace.
    """

    def __init__(self, model_name: str = "Emotion"):
        self.model_name = model_name

    def detect_emotion_from_image(self, image_path: str) -> Optional[str]:
        """
        Runs emotion detection on an image file path.
        Returns emotion label in lowercase or None on failure.
        """
        try:
            analysis = DeepFace.analyze(img_path=image_path, actions=["emotion"], enforce_detection=False)
            emotion = analysis["dominant_emotion"]
            return emotion.lower()
        except Exception as e:
            print(f"Emotion detection failed (image): {e}")
            return None

    def detect_emotion_from_frame(self, frame) -> Optional[str]:
        """
        Runs emotion detection on an OpenCV image frame (numpy array).
        Returns emotion label in lowercase or None on failure.
        """
        try:
            analysis = DeepFace.analyze(img=frame, actions=["emotion"], enforce_detection=False)
            emotion = analysis["dominant_emotion"]
            return emotion.lower()
        except Exception as e:
            print(f"Emotion detection failed (frame): {e}")
            return None