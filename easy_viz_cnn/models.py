from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from easy_viz_cnn.layers import Layer


class SimpleCNNModel:
    """
    Represents a simple CNN model composed of various layers.

    Parameters
    ----------
    layers : list[Layer]
        List of layers in the model.
    """

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def __str__(self) -> str:
        return " -> ".join([str(layer) for layer in self.layers])

    def create_figure(
        self,
        title: str = "CNN Architecture",
        title_fontsize: int = 12,
        layer_color: str = "lightblue",
        layer_fontsize: int = 10,
    ) -> Figure:
        """
        Plot the architecture of the CNN model.

        Parameters
        ----------
        title : str, optional
            Title of the plot. Default is "CNN Architecture".
        title_fontsize : int, optional
            Font size of the title. Default is 12.
        layer_color : str, optional
            Background color of the layer boxes. Default is "lightblue".
        layer_fontsize : int, optional
            Font size of the layer labels. Default is 10.
        """

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(title, fontsize=title_fontsize, pad=5)
        ax.axis("off")

        x_offset = 0.5
        y_offset = 0.80  # Start lower to fit more layers
        layer_height = 0.12  # Larger layer height
        layer_spacing = layer_height + 0.02  # Tight spacing between layers

        for layer in self.layers:
            ax.text(
                x_offset,
                y_offset,
                str(layer),
                ha="center",
                va="center",
                fontsize=layer_fontsize,
                bbox=dict(
                    facecolor=layer_color,
                    edgecolor="black",
                    boxstyle=f"round,pad={layer_height * 5}",
                ),
            )
            y_offset -= layer_spacing

        return fig
