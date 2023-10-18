import cv2
import numpy as np

# Set the size of the checkerboard pattern in pixels
pattern_size = (1920, 1080)  # Adjust to your desired resolution

# Create a black image with the specified size
checkerboard = np.zeros((pattern_size[1], pattern_size[0], 3), dtype=np.uint8)

# Create the 1x1 pixel squares by alternating between white and black
for y in range(0, pattern_size[1], 2):
    for x in range(0, pattern_size[0], 2):
        checkerboard[y:y+1, x:x+1] = [255, 255, 255]

for y in range(1, pattern_size[1], 2):
    for x in range(1, pattern_size[0], 2):
        checkerboard[y:y+1, x:x+1] = [255, 255, 255]

# Display the checkerboard pattern
cv2.imshow('Checkerboard Pattern', checkerboard)
cv2.waitKey(0)
cv2.destroyAllWindows()
