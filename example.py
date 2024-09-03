import matplotlib.pyplot as plt

from easy_viz_cnn.layers import Conv2DLayer as ConvLayer, PoolLayer, FCLayer
from easy_viz_cnn.models import SimpleCNNModel

# Define a simple CNN model using our custom layer classes
model = SimpleCNNModel(
    [
        ConvLayer(1, 32, 3),
        PoolLayer(2),
        ConvLayer(32, 64, 3),
        PoolLayer(2),
        FCLayer(64 * 6 * 6, 128),
        FCLayer(128, 10),
    ]
)

# Visualize the model
fig = model.create_figure(title="My Sample Model")

# Optionally customize the figure further
fig.suptitle("An Additional Super Title")

# Save the figure to a file instead of showing it
fig.savefig("cnn_model_visualization.png", dpi=300)

# Explicitly call plt.show() to keep the window open
plt.show()
