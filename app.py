import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Gesture Assistant", layout="wide")
st.title("🤖 AI Gesture Communication Interface")
st.write("Current Tech Stack: Python | MediaPipe | Streamlit WebRTC")

# --- INITIALIZE MEDIAPIPE ---

hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.8, min_tracking_confidence=0.8)
# --- GESTURE LOGIC ---
def recognize_gesture(landmarks):
    fingertip_ids = [4, 8, 12, 16, 20]
    lower_joint_ids = [2, 6, 10, 14, 18]
    fingers = []
    
    landmark_array = np.array([(lm.x, lm.y) for lm in landmarks])
    min_vals, max_vals = landmark_array.min(axis=0), landmark_array.max(axis=0)
    norm_lms = (landmark_array - min_vals) / (max_vals - min_vals + 1e-6)
    
    for tip, lower in zip(fingertip_ids[1:], lower_joint_ids[1:]):
        fingers.append(1 if norm_lms[tip, 1] < norm_lms[lower, 1] - 0.03 else 0)
    
    thumb_ext = norm_lms[4, 0] > norm_lms[2, 0] + 0.03
    fingers.insert(0, 1 if thumb_ext else 0)
    
    gestures = {
        (0, 0, 0, 0, 0): "Need Help",
        (1, 0, 0, 0, 0): "Thirsty",
        (0, 1, 0, 0, 0): "Pain/Hurt",
        (1, 1, 0, 0, 0): "Call Family/Friend",
        (0, 1, 1, 0, 0): "No",
        (1, 1, 1, 1, 1): "Stop",
        (1, 1, 1, 0, 0): "Yes",
        (0, 0, 1, 0, 0): "I'm Tired",
        (1, 0, 1, 1, 1): "Medicine",
        (0, 0, 0, 1, 1): "Hungry",
        (1, 0, 1, 0, 0): "Help Me",
    }
    return gestures.get(tuple(fingers), "Analyzing Gesture...")

# --- VIDEO PROCESSING CLASS ---
class GestureProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # AI Processing
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)
        
        gesture_text = "Waiting for Hand..."
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                gesture_text = recognize_gesture(hand_lms.landmark)

        # UI Overlay (Bottom Glassmorphism Bar)
        cv2.rectangle(img, (0, h-80), (w, h), (30, 30, 30), -1)
        cv2.putText(img, gesture_text, (int(w/4), h-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- WEB DEPLOYMENT INTERFACE ---
ctx = webrtc_streamer(
    key="gesture-filter",
    video_processor_factory=GestureProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # Required for public web access
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    st.info("AI Model is active. Please allow camera access in your browser.")

st.sidebar.header("About the Project")
st.sidebar.write("""
This application uses **MediaPipe's BlazePalm** model to detect hand landmarks and translates them into semantic meaning.
""")
