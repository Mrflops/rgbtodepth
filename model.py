import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained Teachable Machine model
model = load_model('teachable_machine_model.h5')  # Replace with the path to your exported model

# Open a video capture source (0 for default camera, or provide a video file path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for input to the Teachable Machine model
    resized_frame = cv2.resize(frame, (224, 224))
    resized_frame = np.expand_dims(resized_frame, axis=0)
    resized_frame = resized_frame / 255.0  # Normalize the pixel values to [0, 1]

    # Use the Teachable Machine model to classify the object
    predictions = model.predict(resized_frame)
    class_index = np.argmax(predictions)
    class_label = "Hand" if class_index == 0 else "Not Hand"  # Adjust class labels accordingly

    # Draw a square around the detected object (in this case, a hand)
    if class_index == 0:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)

    # Display the classification label
    cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Teachable Machine Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()