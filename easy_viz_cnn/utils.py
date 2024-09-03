import numpy as np
import matplotlib.pyplot as plt

from easy_viz_cnn.layers import Conv2DLayer as ConvLayer


OneUnitCNN = 0.1


# Function to draw a rectangle
def draw_rectangle(ax, center, width, height, color):
    rectangle = plt.Rectangle(
        center - np.array([width / 2, height / 2]),
        width,
        height,
        color=color,
        ec="black",
        linewidth=2,
    )
    ax.add_patch(rectangle)


def draw_conv_layer(conv_layer: ConvLayer, input_size):
    feature_map_size = conv_layer.output_shape(input_size)
    print(feature_map_size)
    


