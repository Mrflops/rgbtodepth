import cv2
import numpy as np

# Set up the projector screen dimensions (in pixels)
projector_screen_width = 1980
projector_screen_height = 1080

# Define your camera calibration parameters (replace these with your actual values)
fx = 0.0  # Focal length in the x-direction
fy = 0.0  # Focal length in the y-direction
cx = 0.0  # Principal point x-coordinate
cy = 0.0  # Principal point y-coordinate

k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0  # Distortion coefficients

# Create the camera matrix and distortion coefficients
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
dist_coefficients = np.array([k1, k2, p1, p2, k3])

# Create a video capture object for your camera
cap = cv2.VideoCapture(0)  # You may need to adjust the camera index

# Initialize the KLT tracker
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Variables for tracking
old_frame = None
old_gray = None
p0 = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort the frame using camera calibration parameters
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coefficients)

    frame_gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

    if old_frame is None:
        old_frame = undistorted_frame
        old_gray = frame_gray
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    if p0 is not None:
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                # Project the tracked position onto the projector screen coordinates
                projected_x = int(a * projector_screen_width / undistorted_frame.shape[1])
                projected_y = int(b * projector_screen_height / undistorted_frame.shape[0])

                # Draw a rectangle or marker at the projected position
                cv2.rectangle(frame, (projected_x - 10, projected_y - 10), (projected_x + 10, projected_y + 10), (0, 255, 0), 2)

            p0 = good_new.reshape(-1, 1, 2)
        else:
            p0 = None

    # Display the frame
    cv2.imshow("Object Tracking", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
