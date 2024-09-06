import numpy as np
import matplotlib.pyplot as plt

from easy_viz_cnn.colors import DARK_LIGHT

OneUnitCNN = 0.03
OFFSET = 0.2


# Function to draw a rectangle
def draw_rectangle(ax, center, width, height, color):
    rectangle = plt.Rectangle(
        center - np.array([width / 2, height / 2]),
        width,
        height,
        color=color,
        ec="black",
        linewidth=1.2,
    )
    ax.add_patch(rectangle)


def draw_conv_layer(
    ax,
    # conv_layer: ConvLayer,
    feature_map_size,
    center=np.array([0.5, 0.5]),
    colors=DARK_LIGHT,
    title="",
    offset=OFFSET,
):
    if title:
        title += "\n"
    if title != "Input\n":
        vline_x = center[0] - 1 * offset / 2
        ax.plot([vline_x, vline_x], [0.5, 1], "r")
    width = OneUnitCNN * feature_map_size[0]
    height = OneUnitCNN * feature_map_size[1]
    center[0] += offset + width / 2
    if len(feature_map_size) == 3:
        num_channels = feature_map_size[-1]
        shift = 0.05
        center[1] += (
            num_channels // 2 - 1
        ) * shift  # Center adjustment based on the number of channels

        for i, color in zip(range(num_channels), colors):

            current_center = center + i * np.array(
                [shift, -shift]
            )  # Shift the center for each channel
            draw_rectangle(
                ax, current_center, width, height, color
            )  # Draw rectangle for each channel
            if (num_channels // 2 - 1) == i:
                ax.plot(
                    [center[0] - width / 2, current_center[0] - width / 2],
                    [0.5, 0.5],
                    "r",
                    zorder=1,
                )
        # Calculate the midpoint for text justification
        x_min = center[0] - width / 2  # Left boundary of the x-lim area
        x_max = current_center[0] + width / 2  # Right boundary of the x-lim area
        x_mid = (x_min + x_max) / 2  # Midpoint between the two

        # Place feature map size at the midpoint of the x-lim area
        ax.text(
            x_mid,
            -1.0,  # Position the text below the layer at the calculated midpoint
            f"Feature Map\n({feature_map_size[0]}x{feature_map_size[1]}x{feature_map_size[2]})",  # Text showing the size
            ha="center",
            fontsize=8,
            color="black",
        )
        return current_center[0] + width  # Return x-coordinate of the last rectangle

    if feature_map_size[0] == 1:
        width += OFFSET / 2
        center[0] += OFFSET / 2
        height *= 0.5
    draw_rectangle(ax, center, width, height, next(colors))
    # Place feature map size below the single rectangle
    ax.text(
        center[0],
        -1.5,  # Position the text below the layer
        f"{title}{feature_map_size[0]}x{feature_map_size[1]}",  # Text showing the size
        ha="center",
        fontsize=8,
        color="black",
    )

    return center[0] + width
