import cv2
import numpy as np
from PIL import Image


def analyze_eye_contact(image):
    """
    Analyze eye contact using OpenCV face and eye detection.

    Returns:
        (score, message)
    """

    # Ensure PIL image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))

    # Convert to RGB, then numpy array
    image = image.convert("RGB")
    image_np = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Load OpenCV pretrained classifiers
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return 0, "No face detected"

    # Use the largest detected face
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face

    # Region of interest
    face_gray = gray[y:y + h, x:x + w]

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )

    num_eyes = len(eyes)

    # Score based on number of visible eyes
    if num_eyes >= 2:
        return 90, "Excellent eye contact"
    elif num_eyes == 1:
        return 65, "Moderate eye contact"
    else:
        return 30, "Poor eye contact"