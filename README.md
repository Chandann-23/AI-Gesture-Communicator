# 🤖 AI Gesture Communication Interface
### Real-time Sign Language Interpretation using Computer Vision & MediaPipe

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](YOUR_STREAMLIT_URL_HERE)

## 📌 Overview
This project is a **Full-Stack AI Application** designed to bridge the communication gap for individuals with speech or hearing impairments. By leveraging **MediaPipe's BlazePalm** model and **OpenCV**, the system performs real-time hand landmark detection and translates specific finger orientations into semantic meaning (e.g., "Need Help", "Thirsty", "Stop").

The application is fully hardware-independent and runs entirely in a web browser via **Streamlit WebRTC**.

## 🚀 Live Demo
Check out the live application here: [Your Streamlit App Link](YOUR_STREAMLIT_URL_HERE)

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **AI/ML Framework:** MediaPipe (Hand Landmarking)
* **Computer Vision:** OpenCV (Headless)
* **Web Framework:** Streamlit
* **Real-time Streaming:** Streamlit-WebRTC (ICE/STUN servers)
* **Math/Logic:** NumPy (Coordinate Normalization)

## 🧠 How it Works
1.  **Coordinate Acquisition:** MediaPipe extracts 21 unique 3D landmarks from the hand.
2.  **Normalization:** To ensure the system works regardless of the hand's distance from the camera, all coordinates are normalized using a bounding-box scaling algorithm:
    $$norm\_lms = \frac{landmark\_array - min\_vals}{max\_vals - min\_vals + 1e-6}$$
3.  **Heuristic Mapping:** The system analyzes the relative Y-coordinates of fingertips compared to their lower joints to determine which fingers are extended.
4.  **Semantic Translation:** A pattern-matching dictionary translates finger configurations (tuples) into human-readable text.

## 📁 Project Structure
```text
├── app.py              # Main Streamlit application logic
├── requirements.txt    # Cloud-optimized dependencies (Headless CV2)
└── README.md           # Project documentation
