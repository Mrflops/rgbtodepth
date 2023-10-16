import tensorflow as tf
import cv2
import numpy as np

# Load your Teachable Machine model
model = tf.keras.models.load_model('converted_keras/keras_model.h5')

# Open a video capture source (0 for default camera, or provide a video file path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for input to the model
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0  # Normalize pixel values to [0, 1]

    # Use the model to classify the object
    predictions = model.predict(frame)
    class_indices = np.argsort(predictions[0])[::-1]  # Sort predictions in descending order

    for i in range(2):
        class_index = class_indices[i]
        confidence = predictions[0][class_index]

        if class_index == 0:
            label = "Ball"
        else:
            label = "Balloon"

        # Display the classification label and confidence
        cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Object Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
