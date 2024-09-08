import matplotlib.pyplot as plt
import numpy as np

from easy_viz_cnn.utils import draw_rectangle


# Create a figure and axis
fig, ax = plt.subplots()
ax.set_aspect('equal')

# Number of rectangles
num_rectangles = 5
shift = 0.05  # Shift amount for each rectangle

# Generate rectangles with alternating dark and light colors
for i in range(num_rectangles):
    width = 0.7  # the width
    height = 0.4  # the height
    # Alternate between dark and light colors
    color = (0.4, 0.4, 0.4) if i % 2 == 1 else (0.8, 0.8, 0.8)
    center = np.array([0.5, 0.5]) + i * np.array([shift, -shift])
    draw_rectangle(ax, center, width, height, color)
    
for i in range(num_rectangles+4):
    width = 0.3  # the width
    height = 0.2 # the height
    # Alternate between dark and light colors
    color = (0.4, 0.4, 0.4) if i % 2 == 0 else (0.8, 0.8, 0.8)
    center = np.array([1.5, 0.6]) + i * np.array([shift, -shift])
    draw_rectangle(ax, center, width, height, color)
    
    
plt.ylim(-0.5, 1)
plt.xlim(0, 2.7)
# Remove axes
#ax.axis('off')

# Display the plot
plt.show()
