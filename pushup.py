#CELL1 : !pip install mediapipe opencv-python

#CELL2 :
import cv2
import mediapipe as mp
import numpy as np
import os
from google.colab import files

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Constants for angle calculations and thresholds
ANGLE_THRESHOLD_PUSHUP = 160  # Elbow angle for push-up "up" position
ANGLE_THRESHOLD_SITUP = 120   # Knee angle for sit-up "up" position
ANGLE_TOLERANCE = 20          # Tolerance for angle detection

# Initialize counters and states
pushup_counter = 0
situp_counter = 0
pushup_stage = None
situp_stage = None
  
#cell3 :
uploaded = files.upload()
video_path = list(uploaded.keys())[0]
print(f"Uploaded video: {video_path}")

#cell4 :  
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # Third point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

#cell5 :
# Initialize the pose detection model
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open the input video
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the output video writer
output_path = '/content/output_video.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame
    results = pose.process(frame_rgb)
    
    # Draw landmarks if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame_rgb,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        landmarks_list = [(landmark.x * frame_width, landmark.y * frame_height) for landmark in landmarks]
        
        # Push-up detection (using right elbow: landmarks 12, 14, 16 - shoulder, elbow, wrist)
        if len(landmarks_list) > 16:
            shoulder = landmarks_list[12]  # Right shoulder
            elbow = landmarks_list[14]    # Right elbow
            wrist = landmarks_list[16]    # Right wrist
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Count push-up
            if angle > ANGLE_THRESHOLD_PUSHUP - ANGLE_TOLERANCE:
                if pushup_stage == 'down':
                    pushup_stage = 'up'
                    pushup_counter += 1
            elif angle < ANGLE_THRESHOLD_PUSHUP + ANGLE_TOLERANCE:
                pushup_stage = 'down'
        
        # Sit-up detection (using right knee: landmarks 24, 26, 28 - hip, knee, ankle)
        if len(landmarks_list) > 28:
            hip = landmarks_list[24]      # Right hip
            knee = landmarks_list[26]     # Right knee
            ankle = landmarks_list[28]    # Right ankle
            angle = calculate_angle(hip, knee, ankle)
            
            # Count sit-up
            if angle > ANGLE_THRESHOLD_SITUP - ANGLE_TOLERANCE:
                if situp_stage == 'down':
                    situp_stage = 'up'
                    situp_counter += 1
            elif angle < ANGLE_THRESHOLD_SITUP + ANGLE_TOLERANCE:
                situp_stage = 'down'
    
    # Convert back to BGR for saving
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Add counter text
    cv2.putText(frame_bgr, f'Push-ups: {pushup_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_bgr, f'Sit-ups: {situp_counter}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Write the frame to output
    out.write(frame_bgr)

cap.release()
out.release()
pose.close()

print(f"Output video saved as: {output_path}")
print(f"Total Push-ups: {pushup_counter}")
