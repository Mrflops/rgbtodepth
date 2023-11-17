import cv2
import numpy as np
from keras.models import load_model

# Calibration code
cap = cv2.VideoCapture(0)  # Use the correct camera index
calibration_image = np.zeros((720, 1280, 3), dtype=np.uint8)
corners = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append((x, y))
        cv2.circle(calibration_image, (x, y), 5, (0, 255, 0), -1)

cv2.namedWindow('Calibration')
cv2.setMouseCallback('Calibration', mouse_callback)

while len(corners) < 4:
    ret, frame = cap.read()

    for corner in corners:
        cv2.circle(frame, corner, 5, (0, 255, 0), -1)

    cv2.imshow('Calibration', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        break
    elif key == ord('r'):
        corners = []
        calibration_image = np.zeros((720, 1280, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()

if len(corners) != 4:
    print("Calibration was not completed.")
    exit()

# Object detection and red ball detection code
model = load_model("converted_keras/keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("converted_keras/labels.txt", "r").readlines()]

camera_width, camera_height = 224, 224
camera = cv2.VideoCapture(0)
camera.set(3, camera_width)
camera.set(4, camera_height)

while True:
    ret, image = camera.read()
    if not ret:
        break

    # Calibration transformation
    transformation_matrix = cv2.getPerspectiveTransform(np.float32(corners), np.float32([[0, 0], [720, 0], [720, 1280], [0, 1280]]))
    calibrated_image = cv2.warpPerspective(image, transformation_matrix, (1280, 720))

    original_image = calibrated_image.copy()

    # Object detection code
    resized_for_prediction = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_AREA)
    prediction = model.predict(np.expand_dims(resized_for_prediction, axis=0))
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    print("Class:", class_name, "Confidence Score:", np.round(confidence_score * 100), "%")

    # Display the calibrated image and object detection result
    cv2.imshow("Calibrated Image", calibrated_image)
    cv2.imshow("Object Detection Result", original_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
