import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from utils.eye_contact import analyze_eye_contact
from utils.audio_analysis import analyze_speech
from utils.video_analysis import extract_frames, extract_audio
from utils.video_scoring import analyze_video_frames
#from utils.whisper_analysis import analyze_transcription

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Smart Interview Performance Analyzer",
    page_icon="🎯",
    layout="centered"
)

# ==========================================================
# Load Trained Emotion Model
# ==========================================================
@st.cache_resource
def load_emotion_model():
    return load_model("models/emotion_model.keras")


model = load_emotion_model()

# Emotion labels
EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ==========================================================
# Image Preprocessing
# ==========================================================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# ==========================================================
# Helper Functions
# ==========================================================
def get_emotion_score(predicted_emotion):
    emotion_scores = {
        "Happy": 90,
        "Neutral": 85,
        "Surprise": 75,
        "Fear": 50,
        "Sad": 45,
        "Angry": 40,
        "Disgust": 35
    }
    return emotion_scores.get(predicted_emotion, 60)


def get_speech_score(pace_label):
    if pace_label == "Good Pace":
        return 90
    elif pace_label in ["Too Slow", "Too Fast"]:
        return 65
    else:
        return 60


def get_rating(score):
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 55:
        return "Average"
    else:
        return "Needs Improvement"


def display_transcription_results(audio_path):
    """
    Show Whisper transcription and filler word analysis.
    """
    with st.spinner("Transcribing speech with Whisper..."):
        (
            transcript,
            actual_word_count,
            filler_count,
            filler_words_found
        ) = analyze_transcription(audio_path)

    st.subheader("📝 Speech Transcript")
    st.write(transcript if transcript else "No speech detected.")

    st.subheader("📌 Communication Analysis")
    st.write(f"Actual Word Count: {actual_word_count}")
    st.write(f"Filler Word Count: {filler_count}")

    if filler_words_found:
        st.write("Filler Words Used:")
        for word, count in filler_words_found.items():
            st.write(f"- {word}: {count}")
    else:
        st.write("No filler words detected.")

    # Communication feedback
    if filler_count == 0:
        st.success("Excellent communication clarity! No filler words detected.")
    elif filler_count <= 3:
        st.success("Good communication. Minimal filler word usage.")
    elif filler_count <= 8:
        st.warning("Moderate filler word usage. Try to reduce hesitations.")
    else:
        st.error("High filler word usage. Practice speaking more confidently.")


# ==========================================================
# Title
# ==========================================================
st.title("🎯 Smart Interview Performance Analyzer")
st.write(
    "Analyze interview performance using image, audio, and video."
)

# ==========================================================
# IMAGE ANALYSIS SECTION
# ==========================================================
st.header("🖼️ Image Analysis")

uploaded_image = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png"],
    key="image_uploader"
)

predicted_emotion = None
eye_score = None
pace_label = None

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    with st.spinner("Analyzing image..."):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)

        predicted_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        predicted_emotion = EMOTIONS[predicted_index]

    st.success(f"Predicted Emotion: {predicted_emotion}")
    st.write(f"Confidence: {confidence:.2%}")

    # Eye contact analysis
    eye_score, eye_message = analyze_eye_contact(image)

    st.subheader("👁️ Eye Contact Analysis")
    st.write(f"Eye Contact Score: {eye_score}/100")
    st.write(f"Assessment: {eye_message}")

    # Emotion probabilities
    st.subheader("📊 Emotion Probabilities")
    for emotion, probability in zip(EMOTIONS, predictions[0]):
        st.write(f"{emotion}: {probability:.2%}")

# ==========================================================
# AUDIO ANALYSIS SECTION
# ==========================================================
st.header("🎤 Speech Analysis")

audio_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3"],
    key="audio_uploader"
)

if audio_file is not None:
    temp_audio_path = "temp_audio.wav"

    # Save uploaded audio
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.read())

    # Basic speech metrics
    with st.spinner("Analyzing speech pace..."):
        duration, estimated_words, wpm, pace_label = analyze_speech(
            temp_audio_path
        )

    st.subheader("📊 Speech Metrics")
    st.write(f"Duration: {duration:.2f} seconds")
    st.write(f"Estimated Words: {estimated_words}")
    st.write(f"Words Per Minute (WPM): {wpm:.2f}")
    st.write(f"Pace Assessment: {pace_label}")

    # Whisper transcription and filler analysis
    display_transcription_results(temp_audio_path)

# ==========================================================
# OVERALL INTERVIEW SCORE (Image + Audio)
# ==========================================================
if (
    predicted_emotion is not None
    and eye_score is not None
    and pace_label is not None
):
    emotion_score = get_emotion_score(predicted_emotion)
    speech_score = get_speech_score(pace_label)

    overall_score = int(
        (0.4 * emotion_score)
        + (0.3 * eye_score)
        + (0.3 * speech_score)
    )

    rating = get_rating(overall_score)

    st.header("🏆 Overall Interview Score")
    st.metric("Interview Score", f"{overall_score}/100")
    st.write(f"Rating: {rating}")

    st.subheader("💡 Personalized Feedback")

    if predicted_emotion in ["Angry", "Fear", "Sad"]:
        st.write("- Try to appear more relaxed and confident.")

    if eye_score < 70:
        st.write("- Maintain stronger eye contact with the camera.")

    if pace_label == "Too Slow":
        st.write("- Speak a little faster to sound more engaging.")
    elif pace_label == "Too Fast":
        st.write("- Slow down slightly to improve clarity.")
    else:
        st.write("- Your speaking pace is well balanced.")

# ==========================================================
# VIDEO ANALYSIS SECTION
# ==========================================================
st.header("🎥 Video Analysis")

video_file = st.file_uploader(
    "Upload an interview video",
    type=["mp4"],
    key="video_uploader"
)

if video_file is not None:
    temp_video_path = "temp_video.mp4"

    # Save uploaded video
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())

    # Extract frames and audio
    with st.spinner("Processing video..."):
        frame_paths = extract_frames(
            temp_video_path,
            output_folder="frames",
            interval_seconds=2
        )

        audio_path = extract_audio(
            temp_video_path,
            output_audio="video_audio.wav"
        )

    # Analyze frames
    with st.spinner("Analyzing video frames..."):
        (
            video_emotion,
            video_eye_score,
            analyzed_frames
        ) = analyze_video_frames(frame_paths, model)

    # Analyze audio pace
    with st.spinner("Analyzing video speech pace..."):
        (
            duration,
            estimated_words,
            wpm,
            video_pace_label
        ) = analyze_speech(audio_path)

    # Compute score
    emotion_score = get_emotion_score(video_emotion)
    speech_score = get_speech_score(video_pace_label)

    overall_score = int(
        (0.4 * emotion_score)
        + (0.3 * video_eye_score)
        + (0.3 * speech_score)
    )

    rating = get_rating(overall_score)

    # Display summary
    st.success("Video analysis completed successfully!")

    st.subheader("🎥 Video Summary")
    st.write(f"Frames Extracted: {len(frame_paths)}")
    st.write(f"Frames Analyzed: {analyzed_frames}")
    st.write(f"Dominant Emotion: {video_emotion}")
    st.write(f"Average Eye Contact Score: {video_eye_score}/100")
    st.write(f"Duration: {duration:.2f} seconds")
    st.write(f"Words Per Minute (WPM): {wpm:.2f}")
    st.write(f"Pace Assessment: {video_pace_label}")

    # Show first frame
    if len(frame_paths) > 0:
        st.subheader("🖼️ First Extracted Frame")
        st.image(frame_paths[0], width="stretch")

    # Whisper transcription on extracted audio
    display_transcription_results(audio_path)

    # Final score
    st.header("🏆 Video Interview Score")
    st.metric("Overall Score", f"{overall_score}/100")
    st.write(f"Rating: {rating}")

    # Video feedback
    st.subheader("💡 Video Feedback")

    if video_emotion in ["Angry", "Fear", "Sad"]:
        st.write("- Try to appear more calm and confident.")

    if video_eye_score < 70:
        st.write("- Maintain better eye contact throughout the interview.")

    if video_pace_label == "Too Slow":
        st.write("- Speak a little faster.")
    elif video_pace_label == "Too Fast":
        st.write("- Slow down slightly.")
    else:
        st.write("- Your speaking pace is well balanced.")

    if overall_score >= 85:
        st.success("Excellent video interview performance!")
    elif overall_score >= 70:
        st.success("Good performance with minor improvements possible.")
    else:
        st.warning("There is room for improvement.")