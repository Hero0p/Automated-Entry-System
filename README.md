**Automated Entry System**
This project, Automated Entry System, is developed by Nishant Chaudhary, Pradhuman Singh Rajvi, and Aryan. It aims to automate the vehicle entry process by utilizing advanced technologies such as computer vision and optical character recognition (OCR).

The primary goal of this system is to detect vehicle number plates using a trained Convolutional Neural Network (CNN) and extract the plate's text using EasyOCR. The extracted data is then stored in an Excel database along with the current date and time, streamlining the entry process and eliminating the need for manual intervention.

Features
Real-Time Number Plate Detection:

Uses live camera feed (via mobile or IP camera).
Detects and crops the number plate using a Haarcascade model.
Text Extraction:

Leverages EasyOCR for highly accurate text recognition from number plates.
Data Storage:

Automatically logs vehicle details into an Excel sheet for record-keeping.
User-Friendly Deployment:

The app is deployed locally using Flask, with public access enabled via Ngrok for demonstration purposes.
Repository Overview
Code: Includes Python scripts for number plate detection.
Dataset: Contains positive and negative images for training the CNN model.
Documentation: Includes a project report and presentation slides.
This project demonstrates how AI can improve efficiency in entry management systems, making it highly applicable in industries such as parking management and gated communities. It represents the collective effort and technical expertise of the development team.
