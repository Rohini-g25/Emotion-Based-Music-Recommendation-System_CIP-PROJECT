import cv2
import threading
import time
import numpy as np
from collections import Counter # Added for voting
from flask import Flask, Response, jsonify
from fer import FER

app = Flask(__name__)

detector = FER(mtcnn=False)
video_capture = cv2.VideoCapture(0)

# --- NEW: Smoothing Variables ---
current_emotion = "neutral"
latest_frame = None
emotion_window = [] 
WINDOW_SIZE = 10 # Remembers the last 10 detections
# --------------------------------

songs = {
    "happy": {"name": "Uptown Funk", "url": "https://www.youtube.com/embed/OPf0YbXqDm0"},
    "sad": {"name": "Let Me Down Slowly", "url": "https://www.youtube.com/embed/jLNrvmXboj8"},
    "angry": {"name": "Believer", "url": "https://www.youtube.com/embed/7wtfhZwyrcc"},
    "neutral": {"name": "Perfect", "url": "https://www.youtube.com/embed/2Vv-BfVoq4g"},
    "surprise": {"name": "Sugar", "url": "https://www.youtube.com/embed/09R8_2nJtjg"},
    "fear": {"name": "Thriller", "url": "https://www.youtube.com/embed/sOnqjkJTMaA"},
    "disgust": {"name": "Bad Guy", "url": "https://www.youtube.com/embed/DyDfgMOUjCI"}
}

def capture_frames():
    global latest_frame
    while True:
        success, frame = video_capture.read()
        if success:
            latest_frame = frame
        else:
            time.sleep(0.1)

def detect_emotion():
    global current_emotion, latest_frame, emotion_window
    while True:
        if latest_frame is not None:
            try:
                # Pre-processing for better accuracy
                gray = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                processed_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                
                emotions = detector.detect_emotions(processed_frame)
                
                if emotions:
                    # Get raw scores
                    res = emotions[0]["emotions"]
                    
                    # Logic: Prioritize active emotions over neutral
                    # If an active emotion is > 20%, consider it a candidate
                    active_candidates = {k: v for k, v in res.items() if k != 'neutral' and v > 0.20}
                    
                    if active_candidates:
                        detected = max(active_candidates, key=active_candidates.get)
                    else:
                        detected = "neutral"
                    
                    # --- SMOOTHING LOGIC ---
                    emotion_window.append(detected)
                    if len(emotion_window) > WINDOW_SIZE:
                        emotion_window.pop(0)
                    
                    # The "Current Emotion" is now the most frequent one in the window
                    most_common = Counter(emotion_window).most_common(1)[0][0]
                    current_emotion = most_common
                    
            except Exception as e:
                print(f"Error: {e}")
        
        time.sleep(0.3) # Faster detection, but smoothed by the window

def generate_video():
    global latest_frame, current_emotion
    while True:
        if latest_frame is None: continue
        
        display_frame = latest_frame.copy()
        # Visual feedback of the "Smoothed" emotion
        cv2.rectangle(display_frame, (10, 10), (300, 60), (0,0,0), -1)
        cv2.putText(display_frame, f"MOOD: {current_emotion.upper()}", 
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04)

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <style>
            body { background: #0f172a; color: white; text-align: center; font-family: sans-serif; }
            .container { margin-top: 30px; }
            #player { border: 5px solid #38bdf8; border-radius: 15px; background: #000; }
            .status-box { font-size: 24px; font-weight: bold; color: #38bdf8; margin: 20px; }
        </style>
    </head>
    <body>
        <h1>🎵 Stabilized Emotion Player</h1>
        <div class="container">
            <img src="/video_feed" width="500" style="border-radius: 10px;">
            <div class="status-box" id="stat">Detecting...</div>
            <iframe id="player" width="600" height="340" src="" frameborder="0" allow="autoplay"></iframe>
        </div>
        <script>
            let lastSong = "";
            async function update() {
                try {
                    const res = await fetch('/emotion');
                    const data = await res.json();
                    document.getElementById("stat").innerText = "Current Mood: " + data.emotion.toUpperCase();
                    if (lastSong !== data.song.url) {
                        lastSong = data.song.url;
                        document.getElementById("player").src = lastSong + "?autoplay=1";
                    }
                } catch(e) {}
            }
            setInterval(update, 2000);
        </script>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion')
def get_emotion():
    return jsonify({"emotion": current_emotion, "song": songs.get(current_emotion, songs["neutral"])})

if __name__ == "__main__":
    threading.Thread(target=capture_frames, daemon=True).start()
    threading.Thread(target=detect_emotion, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)