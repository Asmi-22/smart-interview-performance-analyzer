import librosa


def analyze_speech(audio_path):
    """
    Analyze an audio file and estimate speaking pace.

    Returns:
        duration_seconds (float)
        estimated_words (int)
        words_per_minute (float)
        pace_label (str)
    """

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # Get duration
    duration_seconds = librosa.get_duration(y=y, sr=sr)

    if duration_seconds <= 0:
        return 0, 0, 0, "No speech detected"

    # Approximate word count:
    # Assume average speaking rate ≈ 2.5 words/second
    estimated_words = int(duration_seconds * 2.5)

    # Calculate WPM
    words_per_minute = estimated_words / (duration_seconds / 60)

    # Classify speaking pace
    if words_per_minute < 110:
        pace_label = "Too Slow"
    elif words_per_minute <= 160:
        pace_label = "Good Pace"
    else:
        pace_label = "Too Fast"

    return (
        duration_seconds,
        estimated_words,
        words_per_minute,
        pace_label
    )