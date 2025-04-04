import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from gtts import gTTS
import pygame

# Load trained model
model = load_model("C:\\sign_language_project\\models\\saved_models\\best_model.h5")

# Class labels (A-Z)
class_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Initialize MediaPipe Hands with LOWER detection threshold
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

def preprocess_image(frame):
    """Preprocess image for model."""
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(frame):
    """Predict sign only if a hand is detected."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.flip(rgb_frame, 1)  # Flip image for correct orientation
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        img = preprocess_image(frame)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        confidence = np.max(prediction) * 100
        return predicted_label, confidence
    else:
        return None, 0  # No hand detected

def text_to_speech(text):
    """Convert text to speech."""
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Real-time detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame.")
        break

    cv2.putText(frame, "Press 'p' to predict", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        predicted_label, confidence = predict_image(frame)
        if predicted_label:
            print(f"✅ Predicted Label: {predicted_label}, Confidence: {confidence:.2f}%")
            text_to_speech(f"The predicted sign is {predicted_label}")
        else:
            print("⚠ No hand detected. Try again.")
    
    elif key == ord('q'):
        break

    cv2.imshow("Sign Language Detection", frame)

# Cleanup
cap.release()
cv2.destroyAllWindows()
