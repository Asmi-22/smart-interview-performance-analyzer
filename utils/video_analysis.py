import cv2
import os
from moviepy import VideoFileClip


def extract_frames(video_path, output_folder="frames", interval_seconds=2):
    """
    Extract one frame every N seconds.

    Returns:
        list of frame file paths
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return []

    frame_interval = int(fps * interval_seconds)

    frame_paths = []
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(
                output_folder,
                f"frame_{saved_count}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1

        frame_count += 1

    cap.release()
    return frame_paths


def extract_audio(video_path, output_audio="video_audio.wav"):
    """
    Extract audio from video.

    Returns:
        output audio path
    """
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(
        output_audio,
        logger=None
    )
    clip.close()

    return output_audio