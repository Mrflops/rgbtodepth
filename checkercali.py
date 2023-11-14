from keras.models import load_model
import cv2
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("converted_keras/keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("converted_keras/labels.txt", "r").readlines()]

camera = cv2.VideoCapture(0)

# Initialize smoothing variables
smooth_x, smooth_y, smooth_w, smooth_h = 0, 0, 0, 0
alpha = 0.2  # Smoothing factor

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

    if confidence_score > 0.8 and class_name == "0 Ball":
        # Color segmentation to find "RedCircle"
        lower_red = np.array([0, 0, 150])
        upper_red = np.array([50, 50, 255])
        mask = cv2.inRange(original_image, lower_red, upper_red)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("Number of Contours:", len(contours))
        # Added: Logic to find the largest bounding box
        max_area = 0
        largest_box = None

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h  # Calculate area

            if area > max_area and area > 500:  # Update max area and box, filter small contours
                max_area = area
                largest_box = (x, y, w, h)

        # Draw the largest bounding box with smoothing
        if largest_box is not None:
            x, y, w, h = largest_box
            # Smooth the bounding box coordinates
            smooth_x = int(alpha * x + (1 - alpha) * smooth_x)
            smooth_y = int(alpha * y + (1 - alpha) * smooth_y)
            smooth_w = int(alpha * w + (1 - alpha) * smooth_w)
            smooth_h = int(alpha * h + (1 - alpha) * smooth_h)
            print(smooth_x)
            print(smooth_y)
            print(smooth_w)
            print(smooth_h)
            cv2.rectangle(original_image, (smooth_x, smooth_y), (smooth_x + smooth_w, smooth_y + smooth_h), (0, 255, 0), 2)
            center_x, center_y = smooth_x + smooth_w // 2, smooth_y + smooth_h // 2
            print(f"Center Coordinates of Largest Box: ({center_x}, {center_y})")

    cv2.imshow("Bounding Box", original_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("Number of Contours:", len(contours))
camera.release()
cv2.destroyAllWindows()
