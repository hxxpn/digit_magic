import pygame
from PIL import Image
from datetime import datetime

# Initialize Pygame
pygame.init()

# Set the window size
window_size = (280, 320)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Drawing App")

# Set colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Fill the screen with white
screen.fill(WHITE)

# Create a font for the button text
font = pygame.font.Font(None, 24)

# Create a button rectangle
button_rect = pygame.Rect(90, 290, 100, 30)

# Main loop
running = True
drawing = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                if button_rect.collidepoint(event.pos):
                    # Save button clicked
                    pygame.image.save(screen, "temp_drawing.png")
                    img = Image.open("temp_drawing.png")
                    img = img.convert("L")
                    img = img.resize((28, 28), Image.Resampling.LANCZOS)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"character_{timestamp}.png"

                    img.save(filename)
                    print(f"Image saved as {filename}")
                else:
                    drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                x, y = event.pos
                pygame.draw.circle(screen, BLACK, (x, y), 5)

    # Draw the button
    pygame.draw.rect(screen, GRAY, button_rect)
    button_text = font.render("Save", True, BLACK)
    text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, text_rect)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
