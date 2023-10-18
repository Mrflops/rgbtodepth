import cv2
import numpy as np

# Set the number of rows and columns on the checkerboard pattern
rows, cols = 6, 9  # Adjust based on your checkerboard

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Prepare object points, which are (0,0,0), (1,0,0), (2,0,0), ... (cols-1,rows-1,0)
objp = np.zeros((cols * rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

# Open a video capture source (0 for default camera, or provide a video file path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw the corners on the frame
        cv2.drawChessboardCorners(frame, (cols, rows), corners, ret)

    cv2.imshow('Calibration', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Perform camera calibration
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("Camera calibration successful!")
        print("Camera matrix:")
        print(mtx)
        print("Distortion coefficients:")
        print(dist)
