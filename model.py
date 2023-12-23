import numpy as np
import cv2
from keras.models import load_model

# Load the model and class names
model = load_model("converted_keras/keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("converted_keras/labels.txt", "r").readlines()]

# Set the camera resolution
camera_width, camera_height = 224, 224

# Camera for calibration
calibration_camera = cv2.VideoCapture(0)
calibration_camera.set(3, camera_width)  # Width
calibration_camera.set(4, camera_height)  # Height

# Calibration
calibration_image = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
calibration_corners = []  # Store the four corner points

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(calibration_corners) < 4:
        calibration_corners.append((x, y))
        cv2.circle(calibration_image, (x, y), 5, (0, 255, 0), -1)

cv2.namedWindow('Calibration')
cv2.setMouseCallback('Calibration', mouse_callback)

while len(calibration_corners) < 4:
    ret, frame = calibration_camera.read()

    for corner in calibration_corners:
        cv2.circle(frame, corner, 5, (0, 255, 0), -1)

    cv2.imshow('Calibration', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        break

calibration_camera.release()
cv2.destroyAllWindows()

# Main camera for red ball detection
camera = cv2.VideoCapture(0)
camera.set(3, camera_width)  # Width
camera.set(4, camera_height)  # Height

while True:
    ret, image = camera.read()
    if not ret:
        break

    original_image = image.copy()

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Define range for red color and create a mask
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_image, lower_red, upper_red)

    mask = mask1 + mask2

    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found and draw a bounding box
    if contours:
        # Combine all contours into one and find the bounding box
        all_contours = np.vstack(contours).squeeze()
        x, y, w, h = cv2.boundingRect(all_contours)

        # Draw the bounding box
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw a square using the corners from calibration
        if len(calibration_corners) == 4:
            square_corners = calibration_corners
            cv2.polylines(original_image, [np.array(square_corners)], isClosed=True, color=(255, 0, 0), thickness=2)

    # Display the result
    cv2.imshow("Red Ball Detection", original_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
