import cv2
import numpy as np

# Set the size of the checkerboard pattern in pixels
pattern_size = (10, 10)  # 10x10 pixel squares

# Variables for the checkerboard pattern size
pattern_width = 1980
pattern_height = 1080

# Create a full-screen window for the checkerboard pattern
cv2.namedWindow('Full Screen Checkerboard', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Full Screen Checkerboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Create a black image with the specified size
checkerboard = np.ones((pattern_height, pattern_width, 3), dtype=np.uint8) * 255  # Set the background to white

# Determine the size of the black corners
corner_size = 100  # Adjust to your preference

# Add the four black corners
checkerboard[:corner_size, :corner_size] = [0, 0, 0]  # Top-left corner
checkerboard[:corner_size, -corner_size:] = [0, 0, 0]  # Top-right corner
checkerboard[-corner_size:, :corner_size] = [0, 0, 0]  # Bottom-left corner
checkerboard[-corner_size:, -corner_size:] = [0, 0, 0]  # Bottom-right corner

# Display the full-screen checkerboard pattern
cv2.imshow('Full Screen Checkerboard', checkerboard)

# Variables for calibration
obj_points = []  # 3D points in real-world space
img_points = []  # 2D points in the image plane

# Define the coordinates of the four corners
corner_coordinates = np.array([
    [0, 0],
    [pattern_width - corner_size, 0],
    [0, pattern_height - corner_size],
    [pattern_width - corner_size, pattern_height - corner_size]
], dtype=np.float32)

# Create a video capture object (0 for default camera)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw and display the corners
    for point in corner_coordinates:
        x, y = point
        frame = cv2.rectangle(frame, (int(x), int(y)), (int(x) + corner_size, int(y) + corner_size), (0, 0, 0), -1)

    # Display the frame
    cv2.imshow('Calibration', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print the coordinates of the four corners
print("Corner Coordinates:")
print(corner_coordinates)
