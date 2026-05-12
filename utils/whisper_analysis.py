# utils/whisper_analysis.py

import whisper

# Load the Whisper model once when the module is imported.
# "base" offers a good balance between speed and accuracy.
model = whisper.load_model("base")


def analyze_transcription(audio_path):
    """
    Transcribe speech and detect filler words.

    Parameters:
        audio_path (str): Path to the audio file.

    Returns:
        transcript (str)
        word_count (int)
        filler_count (int)
        filler_words_found (dict)
    """

    # Transcribe audio
    result = model.transcribe(audio_path)

    # Extract transcript text
    transcript = result["text"].strip()

    # Count actual words
    words = transcript.split()
    word_count = len(words)

    # Common filler words to detect
    fillers = [
        "um",
        "uh",
        "like",
        "actually",
        "basically",
        "you know"
    ]

    transcript_lower = transcript.lower()

    filler_words_found = {}
    filler_count = 0

    for filler in fillers:
        count = transcript_lower.count(filler)
        if count > 0:
            filler_words_found[filler] = count
            filler_count += count

    return (
        transcript,
        word_count,
        filler_count,
        filler_words_found
    )