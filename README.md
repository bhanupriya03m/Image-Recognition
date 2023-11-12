# Image-Recognition
## Overview
This project uses the DeepFace library to perform emotion analysis and face detection on images provided via URL. It includes a Jupyter notebook with Python code that utilizes DeepFace for emotion analysis and OpenCV for face detection.
## Features
Emotion Analysis: DeepFace library is used to analyze the dominant emotion in the provided image.
Face Detection: OpenCV is employed for face detection, and rectangles are drawn around detected faces with labels indicating the dominant emotion.
## Installation
pip install deepface
pip install deepface opencv-python-headless matplotlib ipywidgets
## Usage
1. Open the Jupyter notebook in your Jupyter environment.
2. Run the notebook cells to execute the code.
3. Enter the URL of the image you want to analyze.
4. Click the "Analyze" button to perform emotion analysis and face detection.
5. View the analyzed image with rectangles around detected faces and corresponding dominant emotions.
## Dependencies
deepface: A Python library for deep learning-based face analysis.
opencv-python-headless: OpenCV library for computer vision and image processing tasks.
matplotlib: A Python plotting library for creating visualizations.
ipywidgets: Interactive widgets for the Jupyter notebook.
