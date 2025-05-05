# Air_signature_Sahil
Overview

This project implements a touchless digital signature system using hand-tracking and gesture recognition with a regular camera. It captures air-drawn signatures, processes them, and provides options for further authentication or storage.
Built using Python, OpenCV, and Google Mediapipe.

Features
	•	Real-time hand tracking.
	•	Captures fingertip movement to create digital air signatures.
	•	Processes, cleans, and saves air signatures for further use.
	•	Modular code structure with easy-to-extend functionality.
 Requirements
	•	Python 3.7+
	•	OpenCV (cv2)
	•	Mediapipe
 Project Structure
	•	Hand Tracking:
Detects hand landmarks (especially index fingertip) using Mediapipe Hands module.
	•	Signature Processing:
Captures fingertip path and smooths the signature using basic image processing techniques.
	•	Display and Save:
Displays the drawn signature in real-time and provides options to save it.

Sample Workflow
	1.	The system activates the camera.
	2.	Tracks the user’s index finger.
	3.	Records the path as the user “draws” their signature in the air.
	4.	Processes the drawing into a clean signature image.
	5.	Optionally saves or verifies the signature.

Future Enhancements
	•	Signature verification/authentication models.
	•	Enhanced smoothing and noise removal.
	•	GUI-based user interaction.
	•	Mobile or web integration.

Credits
	•	OpenCV for real-time computer vision.
	•	Google Mediapipe for efficient hand-tracking.
