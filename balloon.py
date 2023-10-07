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

# Load the "balloon.png" image from the "assets" directory
balloon_image = pygame.image.load("assets/balloon.png")

# Load the "splat.png" image from the "assets" directory for when a balloon is popped
splat_image = pygame.image.load("assets/splat.png")

# Get the dimensions of the balloon image
original_width, original_height = balloon_image.get_size()

# Calculate the scaled dimensions
scaled_width = original_width // 3
scaled_height = original_height // 3

# Scale the balloon image
balloon_image = pygame.transform.scale(balloon_image, (scaled_width, scaled_height))

# Create a list to store balloon positions, speeds, and popped status
balloons = []

def spawn_balloon():
    x = random.randint(0, screen_width - scaled_width)
    y = screen_height  # Start balloons at the bottom of the screen
    speed = random.randint(1, 3)  # Randomize balloon rising speed
    popped = False
    return {"x": x, "y": y, "speed": speed, "popped": popped}

# Spawn an initial set of balloons
for _ in range(5):  # You can adjust the number of balloons
    balloons.append(spawn_balloon())

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left mouse button
            mouse_x, mouse_y = event.pos
            for balloon in balloons:
                x, y = balloon["x"], balloon["y"]
                if x <= mouse_x <= x + scaled_width and y <= mouse_y <= y + scaled_height:
                    # Balloon clicked, set its "popped" status to True
                    balloon["popped"] = True

    # Clear the screen with a white background
    screen.fill((255, 255, 255))

    # Update and draw the balloons
    for balloon in balloons:
        x, y, speed, popped = balloon["x"], balloon["y"], balloon["speed"], balloon["popped"]
        if not popped:
            screen.blit(balloon_image, (x, y))

            # Move the balloon upward
            balloon["y"] -= speed

            # Check if the balloon has gone off the screen
            if balloon["y"] + scaled_height < 0:
                balloons.remove(balloon)
                balloons.append(spawn_balloon())  # Respawn a new balloon
        else:
            # Display the splat image at the balloon's position
            screen.blit(splat_image, (x, y))

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()
sys.exit()
