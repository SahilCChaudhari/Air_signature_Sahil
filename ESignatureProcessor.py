# Import for OpenCV library - video frame rate capturing
import cv2

# Import for ML Handtracking, finger identification
import mediapipe as mp

# Import to keep track of time
import time
import numpy as np
import HandTrackingModule as htm


class HandDrawingApp:
    def __init__(self):
        # Initialize variables for time tracking and video capture
        self.pTime = 0
        self.cap = cv2.VideoCapture(0)  # Capture video from the default camera
        self.detector = htm.handDetector()  # Initialize hand detector

        # Variables for drawing smooth curve
        self.prev_point = None
        self.curve_points = []  # Store points of the curve
        self.smoothed_curve = []  # Store the smoothed curve separately
        self.last_drawn_time = 0  # Track the last time a point was drawn

        # Variables for eraser
        self.eraser_active = False  # Track if eraser is active
        self.eraser_start_time = None  # Track the start time of eraser activation
        self.eraser_text = "Eraser"  # Text to display when eraser is active
        self.eraser_radius = 50  # Radius of the eraser circle

    def get_control_points(self, p1, p2, alpha=0.2):
        # Calculate control points between p1 and p2 for smooth curve
        cx1 = p1[0] + alpha * (p2[0] - p1[0])
        cy1 = p1[1] + alpha * (p2[1] - p1[1])
        cx2 = p2[0] - alpha * (p2[0] - p1[0])
        cy2 = p2[1] - alpha * (p2[1] - p1[1])
        return (int(cx1), int(cy1)), (int(cx2), int(cy2))

    def smooth_strokes(self):
        # Smooth the curve points using a moving average
        smoothed_points = []

        for i in range(len(self.curve_points)):
            if i < 2:
                smoothed_points.append(self.curve_points[i])
            else:
                # Check if any point in the window is None
                if None in self.curve_points[i-2:i+1]:
                    smoothed_points.append(None)
                    continue  # Skip if any point is None
                x = sum([p[0] for p in self.curve_points[i-2:i+1]]) / 3
                y = sum([p[1] for p in self.curve_points[i-2:i+1]]) / 3
                smoothed_points.append((int(x), int(y)))
        return smoothed_points

    def erase_curve_strokes(self, X1, Y1):
        # Erase points in the curve that are within the eraser radius
        updated_curve_points = []
        most_recent_point = None

        for point in self.curve_points:
            if point is None or ((point[0] - X1) ** 2 + (point[1] - Y1) ** 2 > self.eraser_radius ** 2):
                updated_curve_points.append(point)
                most_recent_point = point
            elif most_recent_point is not None:
                updated_curve_points.append(None)

        return updated_curve_points

    def run(self):
        while True:
            success, img = self.cap.read()  # Read a frame from the camera
            img = cv2.flip(img, 1)  # Flip the image horizontally
            img = self.detector.findHands(img)  # Detect hands in the image
            lmList, _ = self.detector.findPosition2(img, draw=False)  # Get finger position landmarks

            # Hand Detected
            if lmList:
                # Get the coordinates of index finger and thumb
                X1, Y1 = lmList[8][1], lmList[8][2]  # Index finger
                X2, Y2 = lmList[4][1], lmList[4][2]  # Thumb

                # Check if eraser should be activated
                # Check if eraser should be deactivated
                if self.eraser_active and X1 > img.shape[1] - 300 and Y1 < 100:
                    if self.eraser_start_time is None:
                        self.eraser_start_time = time.time()
                    else:
                        if time.time() - self.eraser_start_time > 2:
                            self.eraser_active = False
                            self.eraser_text = "Eraser"
                elif not self.eraser_active and X1 > img.shape[1] - 300 and Y1 < 100:
                    if self.eraser_start_time is None:
                        self.eraser_start_time = time.time()
                    else:
                        if time.time() - self.eraser_start_time > 2:
                            self.eraser_active = True
                            self.eraser_text = "Eraser (Active)"
                else:
                    self.eraser_start_time = None

                if not self.eraser_active:
                    # Check if both fingers are touching
                    distance_squared = (X2 - X1)**2 + (Y2 - Y1)**2
                    if distance_squared <= 90**2:
                        cv2.putText(img, "Fingers Touching", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        # Draw a smooth curve from the previous point to the current point
                        current_point = (int(X2), int(Y2))
                        self.curve_points.append(current_point)
                        self.last_drawn_time = time.time()

                    # User took pause from drawing
                    else:
                        if len(self.curve_points) > 0 and time.time() - self.last_drawn_time > 2 and self.curve_points[-1] is not None:
                            self.curve_points.append(None)

                # Handle eraser functionality
                else:
                    cv2.putText(img, "Eraser Active", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Draw a circle around the index finger
                    cv2.circle(img, (int(X1), int(Y1)),
                               self.eraser_radius, (255, 255, 255), -1)

                    self.curve_points = self.erase_curve_strokes(X1, Y1)

            # No Hand Detected check and handle pause situation
            else:
                if len(self.curve_points) > 0 and time.time() - self.last_drawn_time > 2 and self.curve_points[-1] is not None:
                    cv2.putText(img, "Pause Detected", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    self.curve_points.append(None)

            # Draw the smoothed curve
            if len(self.curve_points) >= 2:
                self.smoothed_curve = self.smooth_strokes()

                for i in range(1, len(self.smoothed_curve)):
                    if self.smoothed_curve[i-1] is not None and self.smoothed_curve[i] is not None:
                        cv2.line(img, self.smoothed_curve[i-1],
                                 self.smoothed_curve[i], (0, 0, 255), 2)

            # Display eraser text
            cv2.putText(img, self.eraser_text, (img.shape[1] - 200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Calculate and display FPS
            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime

            cv2.imshow("Image", img)  # Display the image

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
                break


# Main Method
if __name__ == "__main__":
    app = HandDrawingApp()
    app.run()




# TODO Tasks : 

    # 1. Increase the smoothness of the lines curve 

    # 2. Make this library customizable so that it can be imported in html web pages 
        # Currently - Python based
        # Planning  - Make it browser compatible (I will have to write the whole code in JavaScript)

    # 3. Add Erasing the drawn signature feature

