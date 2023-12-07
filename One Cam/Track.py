import cv2
import numpy as np  # Add this import statement for NumPy
import inference
import supervision as sv

annotator = sv.BoxAnnotator()
API_KEY = "MhkSytDSM1IWRL6uFSkk"

def calculate_middle_point(box):
    try:
        # Assuming box is an iterable structure representing bounding box
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
            except ValueError as e:
                print(f"Error processing tuple: {e}")
                continue
        else:
            # Handle other cases accordingly
            middle_point = None
        print("Detection Tuple:", detection)
        print("Middle Point:", middle_point)

    # Display the annotated image (comment this line if you don't want to display the image)
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
