import numpy as np
import cv2

class_names = [line.strip() for line in open("converted_keras/labels.txt", "r").readlines()]

# Set the camera resolution
camera_width, camera_height = 224, 224

# Camera 1 for calibration
calibration_camera1 = cv2.VideoCapture(0)
calibration_camera1.set(3, camera_width)  # Width
calibration_camera1.set(4, camera_height)  # Height

# Camera 2 for calibration
calibration_camera2 = cv2.VideoCapture(1)
calibration_camera2.set(3, camera_width)  # Width
calibration_camera2.set(4, camera_height)  # Height

# Calibration for Camera 1
calibration_image1 = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
calibration_corners1 = []  # Store the four corner points for Camera 1

# Calibration for Camera 2
calibration_image2 = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
calibration_corners2 = []  # Store the four corner points for Camera 2


def mouse_callback1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(calibration_corners1) < 4:
        calibration_corners1.append((x, y))
        cv2.circle(calibration_image1, (x, y), 5, (0, 255, 0), -1)


def mouse_callback2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(calibration_corners2) < 4:
        calibration_corners2.append((x, y))
        cv2.circle(calibration_image2, (x, y), 5, (0, 255, 0), -1)


cv2.namedWindow('Calibration Camera 1')
cv2.setMouseCallback('Calibration Camera 1', mouse_callback1)

cv2.namedWindow('Calibration Camera 2')
cv2.setMouseCallback('Calibration Camera 2', mouse_callback2)

while len(calibration_corners1) < 4 or len(calibration_corners2) < 4:
    ret1, frame1 = calibration_camera1.read()
    ret2, frame2 = calibration_camera2.read()

    for corner1 in calibration_corners1:
        cv2.circle(frame1, corner1, 5, (0, 255, 0), -1)

    for corner2 in calibration_corners2:
        cv2.circle(frame2, corner2, 5, (0, 255, 0), -1)

    cv2.imshow('Calibration Camera 1', frame1)
    cv2.imshow('Calibration Camera 2', frame2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and len(calibration_corners1) == 4 and len(calibration_corners2) == 4:
        break

calibration_camera1.release()
calibration_camera2.release()
cv2.destroyAllWindows()

camera.release()
cv2.destroyAllWindows()
