import cv2
import numpy as np
import inference
import supervision as sv

annotator = sv.BoxAnnotator()
API_KEY = "MhkSytDSM1IWRL6uFSkk"

# Set the camera resolution
camera_width, camera_height = 224, 224

# Camera for calibration
calibration_camera = cv2.VideoCapture(0)
calibration_camera.set(3, camera_width)  # Width
calibration_camera.set(4, camera_height)  # Height

# Calibration
calibration_image = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
calibrated_box = []  # Store the four corner points

def mouse_callback(event, x, y, flags, param):
    global calibrated_box

    if event == cv2.EVENT_LBUTTONDOWN and len(calibrated_box) < 4:
        calibrated_box.append((x, y))
        cv2.circle(calibration_image, (x, y), 5, (0, 255, 0), -1)

cv2.namedWindow('Calibration')
cv2.setMouseCallback('Calibration', mouse_callback)

while len(calibrated_box) < 4:
    ret, frame = calibration_camera.read()

    for corner in calibrated_box:
        cv2.circle(frame, corner, 5, (0, 255, 0), -1)

    cv2.imshow('Calibration', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        break

calibration_camera.release()
cv2.destroyAllWindows()

def point_in_box(point, box):
    x, y = point
    x1, y1 = box[0]
    x2, y2 = box[1]
    x3, y3 = box[2]
    x4, y4 = box[3]

    return x1 <= x <= x3 and y1 <= y <= y3

def calculate_middle_point(box):
    try:
        middle_x = int((box[0] + box[2]) / 2)
        middle_y = int((box[1] + box[3]) / 2)
        return middle_x, middle_y
    except Exception as e:
        raise ValueError(f"Error calculating middle point: {e}")

def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)

    for detection in detections:
        if isinstance(detection, tuple) and len(detection) > 0 and isinstance(detection[0], (list, tuple, np.ndarray)):
            try:
                middle_point = calculate_middle_point(detection[0])

                # Check if the middle point is in the calibrated box
                if point_in_box(middle_point, calibrated_box):
                    print("Pong!")

            except ValueError as e:
                print(f"Error processing tuple: {e}")
                continue
        else:
            middle_point = None

        print("Detection Tuple:", detection)
        print("Middle Point:", middle_point)

    # Display the annotated image
    cv2.imshow(
        "Prediction",
        annotator.annotate(
            scene=image,
            detections=detections,
            labels=labels
        )
    )
    cv2.waitKey(1)

inference.Stream(
    source="webcam",
    model="hands-detection-hknbq/3",
    output_channel_order="BGR",
    on_prediction=on_prediction,
)
