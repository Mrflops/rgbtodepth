import pygame
import sys
# Initialize Pygame
pygame.init()
# Set the display width and height
screen_width = 1980
screen_height = 1080
# Create a Pygame screen with the desired resolution and fullscreen flag
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
# Set the window title
pygame.display.set_caption("Fullscreen Pygame Window")
# Create a list to store the positions of green circles
green_circles = []
# Fixed size for the green circles
circle_size = (50, 50)
# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left mouse button
            mouse_x, mouse_y = event.pos
            # Add a green circle with the fixed size at the mouse click position
            green_circles.append((mouse_x - circle_size[0] // 2, mouse_y - circle_size[1] // 2))
    # Clear the screen with a white background
    screen.fill((255, 255, 255))
    # Draw and update the green circles
    for circle_position in green_circles:
        x, y = circle_position
        pygame.draw.circle(screen, (0, 255, 0), (x + circle_size[0] // 2, y + circle_size[1] // 2), circle_size[0] // 2)
    # Update the display
    pygame.display.update()
# Quit Pygame
pygame.quit()
sys.exit()