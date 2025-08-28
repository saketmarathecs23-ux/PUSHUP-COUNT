# Enhanced Push-up Detector
An intelligent computer vision system that automatically detects and counts push-ups in video files using AI pose estimation. Built with MediaPipe and OpenCV, this tool provides real-time analysis with visual feedback and progress tracking.
## Features:
* Accurate Push-up Detection: Uses advanced pose estimation to detect proper push-up form
* Real-time Counter: Displays live count of completed push-ups
* Angle Analysis: Shows elbow angle measurements for form assessment
* Visual Progress Bars: Interactive progress indicators on both sides of the video
* Enhanced Overlays: Professional-looking UI with pose landmarks visualization
* Robust Error Handling: Validates input files and provides clear error messages
* Google Colab Ready: Optimized for easy use in Jupyter notebooks and Google Colab
## How It Works
### Detection Algorithm
* Pose Estimation: Uses MediaPipe's advanced pose detection to identify key body landmarks
* Angle Calculation: Measures the angle between shoulder, elbow, and wrist joints
* Movement Analysis: Tracks the elbow angle to determine push-up phases: 1.Up Position: Elbow angle > 160° 2.Down Position: Elbow angle < 90°
* Counting Logic: Increments counter when transitioning from up to down position

