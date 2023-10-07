import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Set the display width and height
screen_width = 1980
screen_height = 1080

# Create a Pygame screen with the desired resolution and fullscreen flag
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)

# Set the window title
pygame.display.set_caption("Fullscreen Pygame Window")

# Load the "balloon.png" image
balloon_image = pygame.image.load("balloon.png")

# Get the dimensions of the image
original_width, original_height = balloon_image.get_size()

# Calculate the scaled dimensions
scaled_width = original_width // 3
scaled_height = original_height // 3

# Scale the image
balloon_image = pygame.transform.scale(balloon_image, (scaled_width, scaled_height))

# Generate random coordinates for the image's position within the screen bounds
x = random.randint(0, screen_width - scaled_width)
y = random.randint(0, screen_height - scaled_height)

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen with a white background
    screen.fill((255, 255, 255))

    # Draw the scaled balloon image at the random position
    screen.blit(balloon_image, (x, y))

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()
sys.exit()
