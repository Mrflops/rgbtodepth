import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained Teachable Machine model
model = load_model('converted_keras/keras_model.h5')  # Replace with the path to your updated model

# Open a video capture source (0 for default camera, or provide a video file path)
cap = cv2.VideoCapture(1)

# Reduce frame resolution for faster processing
cap.set(3, 320)  # Width
cap.set(4, 240)  # Height

# Variables for frame processing
batch_size = 10  # Experiment with the batch size
frames_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for input to the Teachable Machine model
    resized_frame = cv2.resize(frame, (224, 224))
    resized_frame = np.expand_dims(resized_frame, axis=0)
    resized_frame = resized_frame / 255.0  # Normalize the pixel values to [0, 1]
    frames_buffer.append(resized_frame)

    if len(frames_buffer) >= batch_size:
        batch = np.concatenate(frames_buffer)
        predictions = model.predict(batch)
        for i in range(len(frames_buffer)):
            class_index = np.argmax(predictions[i])
            class_labels = ["Ball", "Background", "Balloon"]  # Adjust class labels accordingly

            # You can add logic here to detect the ball based on class_index

            # Draw a square around the detected object
            if class_index == 0:  # Ball
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)
            elif class_index == 2:  # Balloon
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)
            # You can add more conditions for other classes (e.g., Background)

            # Display the classification label
            class_label = class_labels[class_index]
            cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frames_buffer = []

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
