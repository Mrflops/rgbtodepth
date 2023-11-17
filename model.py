'''
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
#
# # Load the trained model
# model = keras.models.load_model('keras_model.h5')
#
# # Camera feed
# cap = cv2.VideoCapture(0)  # Use 0 for the default camera
#
# while True:
#     # Read a frame
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Preprocess the frame for the model
#     resized_frame = cv2.resize(frame, (224, 224))  # Assuming model input size is (224, 224)
#     normalized_frame = resized_frame.astype('float32') / 255.0
#
#     # Predict flashlight beam in the frame
#     predictions = model.predict(np.expand_dims(normalized_frame, axis=0))
#     print(predictions)
#     # Extract bounding boxes from predictions and draw on the frame
#     # Assuming a single class (flashlight) detection
#     for box in predictions[0]:
#         start_point = (int(box[0] * frame.shape[1]), int(box[1] * frame.shape[0]))
#         end_point = (int(box[2] * frame.shape[1]), int(box[3] * frame.shape[0]))
#         frame = cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
#
#     # Display the frame with detections
#     cv2.imshow('Flashlight Detection', frame)
#
#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


from keras.models import load_model
import cv2
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

camera = cv2.VideoCapture(0)

while True:
    ret, image = camera.read()
    original_image = image.copy()

    resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    image = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    print("Class:", class_name, "Confidence Score:", np.round(confidence_score * 100), "%")

    if class_name == "RedCircle":
        # Color segmentation to find "RedCircle"
        lower_red = np.array([0, 0, 150])
        upper_red = np.array([50, 50, 255])
        mask = cv2.inRange(original_image, lower_red, upper_red)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Added: Logic to find the largest bounding box
        max_area = 0
        largest_box = None

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h  # Calculate area

            if area > max_area:  # Update max area and box
                max_area = area
                largest_box = (x, y, w, h)

        # Draw the largest bounding box
        if largest_box is not None:
            x, y, w, h = largest_box
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_x, center_y = x + w // 2, y + h // 2
            print(f"Center Coordinates of Largest Box: ({center_x}, {center_y})")

    cv2.imshow("Bounding Box", original_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
'''
from keras.models import load_model
import cv2
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("converted_keras old/keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("converted_keras old/labels.txt", "r").readlines()]

# Set the camera resolution
camera_width, camera_height = 224, 224

camera = cv2.VideoCapture(0)
camera.set(3, camera_width)  # Width
camera.set(4, camera_height)  # Height

while True:
    ret, image = camera.read()
    if not ret:
        break

    original_image = image.copy()

    # Convert the image to grayscale and apply adaptive thresholding
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found and draw a bounding box
    if contours:
        # Combine all contours into one and find the bounding box
        all_contours = np.vstack(contours).squeeze()
        x, y, w, h = cv2.boundingRect(all_contours)

        # Draw the bounding box
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Resize the image for prediction
    resized_for_prediction = cv2.resize(original_image, (224, 224), interpolation=cv2.INTER_AREA)

    # Print class and confidence score
    prediction = model.predict(np.expand_dims(resized_for_prediction, axis=0))
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    print("Class:", class_name, "Confidence Score:", np.round(confidence_score * 100), "%")

    # Display the contours and bounding boxes on the original image
    cv2.imshow("Contours and Bounding Box", original_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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

    # Display the result
    cv2.imshow("Red Ball Detection", original_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()