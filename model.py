import cv2
import numpy as np

# Set the desired projection resolution
projection_width = 1920
projection_height = 1080

# Variables for calibration points
calibration_points = []  # To store the four corner points (top-left, top-right, bottom-left, bottom-right)

# Function to handle mouse events for calibration
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_points.append((x, y))
        cv2.circle(calibration_image, (x, y), 5, (0, 255, 0), -1)  # Draw a green dot where you clicked

# Create a blank calibration image
calibration_image = np.zeros((projection_height, projection_width, 3), np.uint8)

# Create a calibration window
cv2.namedWindow('Calibration')
cv2.setMouseCallback('Calibration', mouse_callback)

while len(calibration_points) < 4:
    cv2.putText(calibration_image, "Click on the four corners of the projection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the calibration image
    cv2.imshow('Calibration', calibration_image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Sort calibration points in the order (top-left, top-right, bottom-left, bottom-right)
calibration_points = sorted(calibration_points, key=lambda x: (x[0], x[1]))

# Set the perspective transformation matrix
src_pts = np.float32(calibration_points)
dst_pts = np.float32(
    [[0, 0], [projection_width, 0], [0, projection_height], [projection_width, projection_height]])
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Close the calibration window
cv2.destroyWindow('Calibration')

while True:
    # Create a blank frame to display the calibrated output
    frame = np.zeros((projection_height, projection_width, 3), np.uint8)

    # Apply the perspective transformation
    frame = cv2.warpPerspective(frame, M, (projection_width, projection_height))

    # Display the calibrated frame
    cv2.imshow('Calibrated Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
