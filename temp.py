import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm

class HandDrawingApp:
    def __init__(self):
        self.pTime = 0
        self.cap = cv2.VideoCapture(0)
        self.detector = htm.handDetector()

        # Variables for drawing smooth curve
        self.prev_point = None
        self.curve_points = []
        self.smoothed_curve = []  # Store the smoothed curve separately

        # Variables for eraser
        self.eraser_active = False
        self.eraser_start_time = None
        self.eraser_text = "Eraser"
        self.eraser_radius = 30  # Radius of the eraser circle

    def get_control_points(self, p1, p2, alpha=0.2):
        # Calculate control points between p1 and p2
        cx1 = p1[0] + alpha * (p2[0] - p1[0])
        cy1 = p1[1] + alpha * (p2[1] - p1[1])
        cx2 = p2[0] - alpha * (p2[0] - p1[0])
        cy2 = p2[1] - alpha * (p2[1] - p1[1])
        return (int(cx1), int(cy1)), (int(cx2), int(cy2))

    def smooth_strokes(self, points):
        smoothed_points = []
        for i in range(len(points)):
            if i < 2:
                smoothed_points.append(points[i])
            else:
                x = sum([p[0] for p in points[i-2:i+1]]) / 3
                y = sum([p[1] for p in points[i-2:i+1]]) / 3
                smoothed_points.append((int(x), int(y)))
        return smoothed_points

    def run(self):
        while True:
            success, img = self.cap.read()
            img = cv2.flip(img, 1)
            img = self.detector.findHands(img)
            lmList, _ = self.detector.findPosition2(img)

            if lmList:
                # Get the coordinates of index finger and thumb
                X1, Y1 = lmList[8][1], lmList[8][2]  # Index finger
                X2, Y2 = lmList[4][1], lmList[4][2]  # Thumb

                if not self.eraser_active and X1 > img.shape[1] - 300 and Y1 < 100:
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
                        self.curve_points.append(current_point)  # Append the current point

                        if len(self.curve_points) >= 3:  # Draw a Bezier curve if we have at least 3 points
                            self.smoothed_curve = self.smooth_strokes(self.curve_points)
                            for i in range(1, len(self.smoothed_curve)):
                                cv2.line(img, self.smoothed_curve[i-1],
                                            self.smoothed_curve[i], (0, 0, 255), 2)
                
                # Handle eraser functionality
                else:
                    cv2.putText(img, "Eraser Active", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Draw a circle around the index finger
                    cv2.circle(img, (int(X1), int(Y1)), self.eraser_radius, (255, 255, 255), -1)

                    # Remove all curve points that are within the eraser circle
                    self.curve_points = [point for point in self.curve_points if
                                        (point[0] - X1) ** 2 + (point[1] - Y1) ** 2 > self.eraser_radius ** 2]

            # Draw the smoothed curve
            if len(self.smoothed_curve) >= 2:
                for i in range(1, len(self.smoothed_curve)):
                    cv2.line(img, self.smoothed_curve[i-1],
                            self.smoothed_curve[i], (0, 0, 255), 2)

            # Display eraser text
            cv2.putText(img, self.eraser_text, (img.shape[1] - 200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime

            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    app = HandDrawingApp()
    app.run()
