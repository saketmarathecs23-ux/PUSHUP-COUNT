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
### Key Components
* EnhancedPushupDetector: Main class handling video processing
* Pose Landmark Detection: Real-time body pose estimation
* Angle Calculation: Precise joint angle measurements
* Visual Overlays: Progress bars, counters, and pose visualization
* File Validation: Robust input validation with clear error messages
### On-Screen Display
* Push-up Counter: Prominently displayed in the center
* Elbow Angle: Real-time angle measurement
* Progress Bars: Visual representation of movement range
* Pose Landmarks: Full body pose visualization
* Status Messages: Clear feedback when landmarks aren't detected
### Color Coding
* Yellow: Pose landmarks and connections
* Magenta: Joint markers and progress indicators
* Red: Error messages and warnings
* Cyan: Counter display
## Important Notes
### File Requirements
* Video Files Only: Images are not supported and will trigger validation errors
* Minimum Quality: Ensure good lighting and clear view of the person
* Camera Angle: Side view works best for accurate elbow angle measurement
* Visibility: Left shoulder, elbow, and wrist must be visible
### Limitations
* Currently analyzes left arm only
* Requires clear visibility of key landmarks
* Works best with side-view camera angles
* May not detect partial or improper push-ups
### Detection Warnings
* "Pushup cannot be detected (shoulder not seen)"
* "Pushup cannot be detected (elbow not seen)"
* "Pushup cannot be detected (wrist not seen)"
* "Pushup cannot be detected (landmarks not found)"
