from transformers import pipeline
from PIL import Image
import io


# ---------------------------------------------------
# Load Emotion Detection Model
# ---------------------------------------------------

emotion_pipeline = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection"
)


# ---------------------------------------------------
# Detect Emotion Function
# ---------------------------------------------------

def detect_emotion(image_bytes):
    """
    Detect facial emotion from image bytes.
    Returns the dominant emotion and confidence score.
    """

    try:

        # Convert bytes → PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Ensure RGB format
        image = image.convert("RGB")

        # Run model
        results = emotion_pipeline(image)

        # Sort results by score
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        top_emotion = results[0]["label"].lower()
        confidence = float(results[0]["score"])

        return {
            "emotion": top_emotion,
            "confidence": confidence
        }

    except Exception as e:

        print("Emotion detection error:", e)

        return {
            "emotion": "neutral",
            "confidence": 0.0
        }