# Install necessary libraries
!pip install opencv-python mediapipe numpy
import cv2
import mediapipe as mp
import numpy as np
from google.colab import files
import math
import os

# Upload the video file
print("Upload a video file for push-up detection:")
uploaded = files.upload()
video_path = list(uploaded.keys())[0]
print(f"Uploaded video: {video_path}")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class EnhancedPushupDetector:
    @classmethod
    def setup(cls, video_path):
        cls.cap = cv2.VideoCapture(video_path)
        cls.width = int(cls.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cls.height = int(cls.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cls.fps = cls.cap.get(cv2.CAP_PROP_FPS)
        cls.total_frames = int(cls.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cls.output_path = 'enhanced_pushups_output.mp4'
        
        # Use H.264 codec for better compatibility
        cls.out = cv2.VideoWriter(cls.output_path, cv2.VideoWriter_fourcc(*'mp4v'), cls.fps, (cls.width, cls.height))
        
        # Tracking variables
        cls.counter = 0
        cls.stage = "unknown"
        cls.frame_count = 0
        
        # Improved pose detection settings
        cls.pose = mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )

    @staticmethod
    def calculate_angle(a, b, c):
        """Calculate angle between three points with improved precision"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure valid range
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    @classmethod
    def draw_enhanced_overlays(cls, frame):
        """Draw enhanced UI overlays"""
        overlay = frame.copy()
        
        # Main info panel
        cv2.rectangle(overlay, (0, 0), (cls.width, 80), (50, 50, 50), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        # Push-up counter in the middle of the video - smaller font
        cv2.putText(frame, f'Push-ups: {cls.counter}', (cls.width//2 - 100, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)
        
        return frame

    @classmethod
    def draw_enhanced_progress_bar(cls, frame, angle):
        """Draw smaller progress bars matching reference style"""
        bar_fill = np.interp(angle, (90, 160), (200, 0))

        for i in range(int(200 - bar_fill), 200):
            color = (0, int(255 * (i - (200 - bar_fill)) / bar_fill), 255 - int(255 * (i - (200 - bar_fill)) / bar_fill))
            cv2.line(frame, (80, i + 120), (110, i + 120), color, 1)
            cv2.line(frame, (cls.width - 110, i + 120), (cls.width - 80, i + 120), color, 1)

        cv2.rectangle(frame, (80, 120), (110, 320), (255, 255, 255), 2)
        cv2.rectangle(frame, (cls.width - 110, 120), (cls.width - 80, 320), (255, 255, 255), 2)

        cv2.circle(frame, (95, 120), 6, (255, 0, 255), -1)
        cv2.circle(frame, (cls.width - 95, 120), 6, (255, 0, 255), -1)

    @classmethod
    def count_pushups(cls, angle):
        """Simple push-up counting logic matching reference"""
        if angle > 160:
            cls.stage = "up"
        if angle < 90 and cls.stage == 'up':
            cls.stage = "down"
            cls.counter += 1

    @classmethod
    def process_frame(cls, frame, results):
        """Simplified frame processing with if-else instead of try-except"""
        cls.frame_count += 1
        frame = cls.draw_enhanced_overlays(frame)
        
        # Always draw pose landmarks if they exist
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=6, circle_radius=8),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=6)
            )

            landmarks = results.pose_landmarks.landmark
            
            # Check if all required landmarks are visible
            if (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > 0.5 and
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility > 0.5 and
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > 0.5):
                
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * cls.width,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * cls.height]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * cls.width,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * cls.height]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * cls.width,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * cls.height]

                angle = cls.calculate_angle(shoulder, elbow, wrist)
                cv2.putText(frame, f'Elbow Angle: {int(angle)}°', (20, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
                cls.count_pushups(angle)
                cls.draw_enhanced_progress_bar(frame, angle)
            
            else:
                # Determine which landmark is not visible
                if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility <= 0.5:
                    cv2.putText(frame, "Pushup cannot be detected (shoulder not seen)", (20, 100),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                elif landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility <= 0.5:
                    cv2.putText(frame, "Pushup cannot be detected (elbow not seen)", (20, 100),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Pushup cannot be detected (wrist not seen)", (20, 100),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Pushup cannot be detected (landmarks not found)", (20, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

        return frame

    @classmethod
    def process_video(cls, video_path):
        """Process the entire video"""
        # Get file extension to check if it's a video file
        file_extension = os.path.splitext(video_path)[1].lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        
        # Assert statement to check if uploaded file is not an image
        assert file_extension not in image_extensions, f"Error: Uploaded file '{video_path}' appears to be an image ({file_extension}). Please upload a video file instead. Supported video formats: {', '.join(video_extensions)}"
        
        # Additional check to ensure it's a recognized video format
        assert file_extension in video_extensions, f"Error: Uploaded file '{video_path}' has unsupported format ({file_extension}). Supported video formats: {', '.join(video_extensions)}"
        
        cls.setup(video_path)
        
        # Additional validation: Check if the video can be opened and has frames
        assert cls.total_frames > 0, f"Error: The uploaded file '{video_path}' cannot be processed as a video or contains no frames. Please upload a valid video file."
        assert cls.fps > 0, f"Error: The uploaded file '{video_path}' has invalid frame rate. Please upload a valid video file."
        
        print(f"Processing video: {cls.width}x{cls.height} @ {cls.fps} fps")
        print(f"Total frames: {cls.total_frames}")
        
        while cls.cap.isOpened():
            ret, frame = cls.cap.read()
            if not ret:
                break

            # Convert color space for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = cls.pose.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Process frame and add overlays
            frame = cls.process_frame(frame, results)
            cls.out.write(frame)
            
            # Progress feedback
            if cls.frame_count % 30 == 0:  # Every second at 30fps
                progress = (cls.frame_count / cls.total_frames) * 100
                print(f"Progress: {progress:.1f}% - Push-ups detected: {cls.counter}")

    @classmethod
    def release_resources(cls):
        """Clean up resources and return results"""
        cls.cap.release()
        cls.out.release()
        cls.pose.close()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total push-ups detected: {cls.counter}")
        print(f"Output saved as: {cls.output_path}")
        
        return cls.output_path, cls.counter

# Run the enhanced detector
print("Starting push-up detection...")
try:
    EnhancedPushupDetector.process_video(video_path)
    output_path, total_pushups = EnhancedPushupDetector.release_resources()
    
    # Download the output video
    print(f"\nDownloading processed video...")
    files.download(output_path)
    
    print(f"\n🎉 Analysis Complete!")
    print(f"📊 Results: {total_pushups} push-ups detected")
    print(f"📹 Enhanced video with overlays saved and downloaded")
    
except AssertionError as e:
    print(f"\n❌ {str(e)}")
    print("Please upload a valid video file and try again.")
except Exception as e:
    print(f"\n❌ An error occurred during processing: {str(e)}")
    print("Please check your video file and try again.")
