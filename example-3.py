from matplotlib import pyplot as plt
import numpy as np
from easy_viz_cnn import colors
from easy_viz_cnn.layers import Conv2DLayer as ConvLayer, FCLayer, PoolLayer, InputLayer
from easy_viz_cnn.models import SimpleCNNModel
from easy_viz_cnn.utils import draw_conv_layer

# Create a figure and axis
fig, ax = plt.subplots()
ax.set_aspect("equal")
OFFSET = 0.2


lenet_model = SimpleCNNModel(
    [
        InputLayer((32, 32)),
        ConvLayer(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
        PoolLayer(kernel_size=2, stride=2),
        ConvLayer(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
        PoolLayer(kernel_size=2, stride=2),
        FCLayer(16 * 4 * 4, 120),
    ]
)

x = 0

plt.plot([-1, 8], [0.5, 0.5], color="#808080", zorder=1)

for idx, feature_map in enumerate(lenet_model.features()):
    if idx == 0:
        title = "Input"
    else:
        title = "Fully Connected"
    x = draw_conv_layer(
        ax,
        feature_map,
        np.array([x, 0.5]),
        colors=colors.TEAL_GRAY,
        title=title,
        offset=OFFSET,
    )
    print(x, feature_map)

# plt.xlim(-1, 8)
# plt.ylim(-0.5, 2)

# Remove axes
ax.axis("off")

print(list(lenet_model.features()))

plt.show()
