
# Sign_Language_Detection_Project
=======

=======
# Sign Language to Text/Speech Conversion

A real-time system that converts American Sign Language (ASL) hand gestures into text and speech using deep learning. Built with Python, OpenCV, and TensorFlow.

## Features
- ✋ Real-time hand gesture detection using **MediaPipe**
- 🔤 ASL alphabet classification (A-Z) with **CNN model**
- 🔊 Text-to-speech output using **gTTS**
- 🎮 Interactive interface with keyboard controls

## Technologies Used
| Component          | Technology |
|--------------------|------------|
| Computer Vision    | OpenCV, MediaPipe |
| Deep Learning      | TensorFlow/Keras |
| Text-to-Speech     | gTTS, Pygame |
| Language           | Python 3.8+ |

## Prerequisites
- Python 3.8+
- Webcam
- NVIDIA GPU (recommended for faster inference)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Sarthak-Ajmera/sign-language-project.git
   cd sign-language-project

$$# Sign Language to Text/Speech Conversion

![Project Demo](demo.gif) *(Optional: Add a GIF/video demo at the top)*

A real-time system that converts American Sign Language (ASL) hand gestures into text and speech using deep learning. Built with Python, OpenCV, and TensorFlow.

## Features
- ✋ Real-time hand gesture detection using **MediaPipe**
- 🔤 ASL alphabet classification (A-Z) with **CNN model**
- 🔊 Text-to-speech output using **gTTS**
- 🎮 Interactive interface with keyboard controls

## Technologies Used
| Component          | Technology |
|--------------------|------------|
| Computer Vision    | OpenCV, MediaPipe |
| Deep Learning      | TensorFlow/Keras |
| Text-to-Speech     | gTTS, Pygame |
| Language           | Python 3.8+ |

## Prerequisites
- Python 3.8+
- Webcam
- NVIDIA GPU (recommended for faster inference)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sign-language-project.git
   cd sign-language-project

2. Install Dependencies 
   ```bash
   pip install -r requirements.txt

## Dataset
Dataset
We used the ASL Alphabet Dataset containing:
- 87,000 images of hand gestures (A-Z)
- 29 classes (26 letters + 3 extra gestures)
- 200x200px color images
  ```bash
  data/raw/
  ├── A/
  │   ├── A1.jpg
  │   └── ...
  ├── B/
  └── ...

## How to run
1. Real-Time Prediction 
   ```bash
   python src/inference/predict_real_time.py
   
2. Controls:
- Press P to predict the current gesture
- Press Q to quit

