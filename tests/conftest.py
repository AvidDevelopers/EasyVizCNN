from easy_viz_cnn.layers import Conv2DLayer as ConvLayer, FCLayer, PoolLayer, InputLayer
from easy_viz_cnn.models import SimpleCNNModel
from easy_viz_cnn.utils import draw_conv_layer

from pytest import fixture


# LeNet implementation


@fixture(scope="session")
def lenet_model():
    # Define the LeNet model with the desired outputs
    lenet_model = SimpleCNNModel(
        [
            InputLayer((32, 32)),
            ConvLayer(
                1, 6, kernel_size=(5, 5), stride=1, padding=0
            ),  # Conv1: (32x32x1) -> (28x28x6)
            PoolLayer(kernel_size=2, stride=2),  # Pool1: (28x28x6) -> (14x14x6)
            ConvLayer(
                6, 16, kernel_size=(5, 5), stride=1, padding=0
            ),  # Conv2: (14x14x6) -> (10x10x16)
            PoolLayer(kernel_size=2, stride=2),  # Pool2: (10x10x16) -> (5x5x16)
            FCLayer(5 * 5 * 16, 120),  # FC1: (5x5x16) -> (1x120)
        ]
    )
    return lenet_model
