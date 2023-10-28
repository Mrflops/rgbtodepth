import cv2
import numpy as np

# Create a video capture object for your camera
cap = cv2.VideoCapture(0)  # Use the correct camera index

# Create a blank image for calibration
calibration_image = np.zeros((720, 1280, 3), dtype=np.uint8)

# Initialize corner points
corners = []  # Store the four corner points

# Define a callback function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append((x, y))
        cv2.circle(calibration_image, (x, y), 5, (0, 255, 0), -1)

# Create a window and set the mouse callback
cv2.namedWindow('Calibration')
cv2.setMouseCallback('Calibration', mouse_callback)

while len(corners) < 4:
    ret, frame = cap.read()

    for corner in corners:
        cv2.circle(frame, corner, 5, (0, 255, 0), -1)

    cv2.imshow('Calibration', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Calibration completed
        break
    elif key == ord('r'):
        # Reset calibration
        corners = []
        calibration_image = np.zeros((720, 1280, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()

# Print the four corner coordinates
if len(corners) == 4:
    print("Calibration completed.")
    for i, corner in enumerate(corners, 1):
        print(f"Corner {i}: {corner}")
else:
    print("Calibration was not completed.")
