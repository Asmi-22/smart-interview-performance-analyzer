from collections import Counter
from PIL import Image
import numpy as np

from utils.eye_contact import analyze_eye_contact


EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def analyze_video_frames(frame_paths, model):
    """
    Analyze extracted frames and return:
    - Most common emotion
    - Average eye contact score
    - Number of analyzed frames
    """

    emotion_predictions = []
    eye_scores = []

    for frame_path in frame_paths:
        try:
            image = Image.open(frame_path)

            # Emotion prediction
            processed = preprocess_image(image)
            predictions = model.predict(processed, verbose=0)
            predicted_index = np.argmax(predictions)
            emotion = EMOTIONS[predicted_index]
            emotion_predictions.append(emotion)

            # Eye contact
            eye_score, _ = analyze_eye_contact(image)
            eye_scores.append(eye_score)

        except Exception:
            # Skip unreadable frames
            continue

    if not emotion_predictions:
        return "Unknown", 0, 0

    most_common_emotion = Counter(emotion_predictions).most_common(1)[0][0]
    average_eye_score = int(sum(eye_scores) / len(eye_scores))

    return (
        most_common_emotion,
        average_eye_score,
        len(emotion_predictions)
    )