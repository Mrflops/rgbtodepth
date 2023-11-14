from keras.models import load_model
import cv2
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("converted_keras/keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("converted_keras/labels.txt", "r").readlines()]

# Set the camera resolution
camera_width, camera_height = 224, 224

camera = cv2.VideoCapture(0)
camera.set(3, camera_width)  # Width
camera.set(4, camera_height)  # Height

# Initialize smoothing variables
smooth_x, smooth_y, smooth_w, smooth_h = 0, 0, 0, 0
alpha = 0.2  # Smoothing factor

while True:
    ret, image = camera.read()
    original_image = image.copy()

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to create a binary image
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and bounding boxes on the original image
    for contour in contours:
        # Bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
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
    cv2.imshow("Contours and Bounding Boxes", original_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
