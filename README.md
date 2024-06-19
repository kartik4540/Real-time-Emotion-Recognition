# Real-time-Emotion-Recognition
This project implements real-time facial emotion recognition using deep learning techniques, combining OpenCV for face detection and TensorFlow/Keras for emotion classification. The system captures live video from a webcam, detects faces using a pre-trained Haar Cascade classifier, and predicts emotions such as Angry, Disgusted, Fear, Happy, Sad, Surprise, or Neutral.

(Download raw file of model for recognition.zip  and update the path file in code)

Dependencies:
Python 3.x
OpenCV (opencv-python)
TensorFlow (tensorflow)
Keras (keras)
Matplotlib (matplotlib)
Numpy (numpy)


Overview:
The core of the project includes a deep learning model trained on the FER-2013 dataset, capable of accurately predicting emotions from facial expressions. The model architecture, implemented in TensorFlow/Keras, processes resized face images and classifies them into one of seven emotion categories.

Application:
This technology finds applications in human-computer interaction, sentiment analysis, and emotion-driven interfaces. It provides real-time feedback on emotional states, enhancing user experience in interactive systems and contributing to understanding user sentiment in various applications.

Usage:

Ensure your webcam is connected and accessible.
Run emotion_detection.py to start the real-time emotion recognition system.
Press 'q' to quit the application.
