import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import random
import pyttsx3
import os
import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(message):
    """Convert text to speech."""
    engine.say(message)
    engine.runAndWait()

# Define openers for each emotion
openers = {
    "anger": [
        "Whoa, let's cool down! How about a fun fact to lighten the mood?",
        "I sense some fiery energy! Want to hear a joke to break the ice?",
        "Anger is just passion in disguise. Let me channel it into something fun!"
    ],
    "disgust": [
        "Ew, right? Let’s shake it off with a quirky thought!",
        "Not a fan, huh? Maybe a joke will make things better.",
        "Let me distract you with something unexpected."
    ],
    "fear": [
        "Feeling a bit jumpy? Don’t worry, I’ve got your back with a light-hearted opener!",
        "Scared? Let me share something to make you smile.",
        "Let’s turn that fear into curiosity. Ready?"
    ],
    "happy": [
        "Hello, sunshine! Ready to make this moment even brighter?",
        "Happiness looks good on you! Let’s keep the good vibes going.",
        "Feeling great? Me too! Let’s make this chat unforgettable."
    ],
    "sad": [
        "Feeling down? Let me try to cheer you up with something fun!",
        "Sadness is just a passing cloud. Let’s find the silver lining together.",
        "Hey, it’s okay to feel sad. How about a little joke to lift your spirits?"
    ],
    "surprise": [
        "Surprised? Let’s make this moment even more unexpected!",
        "Didn’t see this coming, did you? Well, here’s something fun!",
        "Life is full of surprises, and I’m here to add another!"
    ],
    "neutral": [
        "Hey there! Let’s turn this ordinary moment into something extraordinary.",
        "Feeling neutral? Let’s spice things up with a quirky opener!",
        "Not much going on? Let me change that with something fun."
    ]
}

def get_opener(emotion):
    """Get a random opener based on the detected emotion."""
    if emotion in openers:
        return random.choice(openers[emotion])
    else:
        return "I’m not sure what you’re feeling, but let’s make this moment awesome!"

def detect_face_and_emotion():
    print("Initializing models...")

    # Load YOLO face detection model
    face_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    face_model = YOLO(face_model_path)

    # Load emotion classification model
    emotion_processor = AutoImageProcessor.from_pretrained("RickyIG/emotion_face_image_classification_v3")
    emotion_model = AutoModelForImageClassification.from_pretrained("RickyIG/emotion_face_image_classification_v3")

    print("Models loaded successfully.")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    try:
        while True:
            print("Waiting for face detection...")

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                continue

            # Detect faces using YOLO
            results = face_model(frame)
            if len(results[0].boxes) > 0:  # If faces detected
                # Process the first detected face
                box = results[0].boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_region = frame[y1:y2, x1:x2]

                # Convert face to PIL Image for emotion detection
                pil_image = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))

                # Perform emotion detection
                inputs = emotion_processor(pil_image, return_tensors="pt")
                with torch.no_grad():
                    logits = emotion_model(**inputs).logits
                predicted_emotion = emotion_model.config.id2label[logits.argmax().item()]

                # Display detected emotion and send to chatbot
                print(f"\nDetected Emotion: {predicted_emotion}")
                opener = get_opener(predicted_emotion)
                print(f"Chatbot Response: {opener}")
                speak(opener)

                # Wait 3 seconds before restarting detection
                print("Waiting 3 seconds before restarting...")
                time.sleep(3)
                continue

            # Show live preview
            cv2.imshow("Face Detection (Press 'q' to quit)", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting program.")
                break

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detect_face_and_emotion()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
