# Install necessary libraries if not already installed in your Colab environment
#!pip install opencv-python mediapipe numpy

import cv2
import mediapipe as mp
import numpy as np
from google.colab import files

# Upload the video file
uploaded = files.upload()
video_path = list(uploaded.keys())[0]
print(f"Uploaded video: {video_path}")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Open the input video
cap = cv2.VideoCapture(video_path)

# Get video properties for output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create output video writer
output_path = 'output_pushups.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

counter = 0
stage = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # No flip needed for pre-recorded video; adjust if necessary
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 100), (50, 50, 50), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # Always display the push-up counter
        cv2.putText(frame, f'Push-ups: {counter}', (width//2 - 150, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

        try:
            landmarks = results.pose_landmarks.landmark

            # Check visibility of required landmarks
            assert landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > 0.5, "shoulder not seen"
            assert landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > 0.5, "elbow not seen"
            assert landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > 0.5, "wrist not seen"

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]

            angle = calculate_angle(shoulder, elbow, wrist)

            cv2.putText(frame, f'Elbow Angle: {int(angle)}°', (30, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 0), 2)

            if angle > 160:
                stage = "up"
            if angle < 90 and stage == 'up':
                stage = "down"
                counter += 1

            bar_fill = np.interp(angle, (90, 160), (350, 0))

            for i in range(int(350 - bar_fill), 350):
                color = (0, int(255 * (i - (350 - bar_fill)) / bar_fill), 255 - int(255 * (i - (350 - bar_fill)) / bar_fill))
                cv2.line(frame, (100, i + 150), (150, i + 150), color, 1)
                cv2.line(frame, (width - 150, i + 150), (width - 100, i + 150), color, 1)

            cv2.rectangle(frame, (100, 150), (150, 500), (255, 255, 255), 2)
            cv2.rectangle(frame, (width - 150, 150), (width - 100, 500), (255, 255, 255), 2)

            cv2.circle(frame, (125, 150), 10, (255, 0, 255), -1)
            cv2.circle(frame, (width - 125, 150), 10, (255, 0, 255), -1)

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=6, circle_radius=8),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=6)
            )

        except AssertionError as e:
            # Display the error message just below the push-up counter with same size and style
            cv2.putText(frame, f"Pushup cannot be detected ({str(e)})", (width//2 - 150, 120),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        except:
            # Handle other exceptions (e.g., no landmarks detected) with same size and style
            cv2.putText(frame, "Pushup cannot be detected (landmarks not found)", (width//2 - 150, 120),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

        # Write the processed frame to output video
        out.write(frame)

# Release resources
cap.release()
out.release()

# Download the output video
files.download(output_path)

print(f"Total push-ups detected: {counter}")
