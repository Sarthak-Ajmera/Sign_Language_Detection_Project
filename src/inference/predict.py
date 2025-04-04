import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from gtts import gTTS
import pygame

# Load the best model
model = load_model(os.path.join("C:\\", "sign_language_project", "models", "saved_models", "best_model.h5"))

# Class labels (ASL alphabet)
class_labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]

def preprocess_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    confidence = np.max(prediction) * 100
    return predicted_label, confidence

def text_to_speech(text):
    # Convert text to speech
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    
    # Play the audio using pygame
    pygame.mixer.init()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait for the audio to finish playing
        pygame.time.Clock().tick(10)

# Example usage
if __name__ == "__main__":
    try:
        image_path = os.path.join("C:\\", "sign_language_project", "data", "raw", "asl_alphabet_test", "test_I.jpg")  # Replace with your image path
        predicted_label, confidence = predict_image(image_path)
        print(f"Predicted Label: {predicted_label}, Confidence: {confidence:.2f}%")
        text_to_speech(f"The predicted sign is {predicted_label}")
    except Exception as e:
        print(f"Error: {str(e)}")