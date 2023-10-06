#Frontend, Use this on the projector
import pygame
import sys

# Initialize Pygame
pygame.init()

# Set the display width and height
screen_width = 1980
screen_height = 1080

# Create a Pygame screen with the desired resolution and full screen flag
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)

# Set the window title
pygame.display.set_caption("Full screen Pygame Window")

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen with a white background
    screen.fill((255, 255, 255))

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()
sys.exit()

