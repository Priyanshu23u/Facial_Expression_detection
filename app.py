from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Initialize the Flask app
app = Flask(__name__)

# Load the emotion detection model
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load model from JSON and its weights
with open('emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("emotion_model.h5")

# Start the webcam
camera = cv2.VideoCapture(0)

def generate_frames():
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (1280, 720))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                # Predict emotion
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                
                # Display the emotion label on the frame
                cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Encode the frame to JPEG format for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    # Render the HTML template (see below for `index.html`)
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    # Route to stream video feed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
