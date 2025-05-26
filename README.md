Walking Signature: A New Approach to Human Identification

This project is a deep learning-based gait recognition system that identifies individuals based on their unique walking patterns. It uses MediaPipe Pose for skeletal keypoint detection and CNN-LSTM models for feature extraction and classification.
The system is integrated into a web interface for real-time registration, training, and prediction.

Key Features
Human pose detection using MediaPipe (33 keypoints)

Gait-based feature extraction using CNN-LSTM

Real-time prediction through a Flask web application

Database integration for storing gait profiles

Supports secure and contactless identification

 Technologies Used
Tool -	Purpose
Python - 	Core programming language
OpenCV -	Video frame extraction and preprocessing
MediaPipe Pose-	Skeletal keypoint detection
CNN + LSTM	Deep learning-based classification
Flask	Backend API and web integration
HTML/CSS/JS	Frontend interface
MySQL	Database to store gait features
