import cv2
import numpy as np

# ... (previous code for calibration and camera setup)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a square around the projected screen
    cv2.rectangle(frame, (0, 0), (projector_screen_width, projector_screen_height), (0, 255, 0), 2)

    # Replace this line with your actual object detection code
    # Your detection logic should return a list [x, y, width, height] for the detected object
    detection = [100, 100, 50, 50]  # Sample detection data

    # Extract bounding box coordinates
    x, y, w, h = detection

    # Project the tracked position onto the projector screen coordinates
    projected_x = int(x * projector_screen_width / frame.shape[1])
    projected_y = int(y * projector_screen_height / frame.shape[0])

    # Draw a rectangle or marker at the projected position
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Debug print: Projector coordinates and success status
    print(f"Projector X: {projected_x}, Projector Y: {projected_y}, Success: True")

    # Display the frame in the window
    cv2.imshow('Ball Tracking', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
