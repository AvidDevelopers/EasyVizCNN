from easy_viz_cnn.layers import Conv2DLayer as ConvLayer, FCLayer, PoolLayer
from easy_viz_cnn.models import SimpleCNNModel
from easy_viz_cnn.utils import draw_conv_layer

# LeNet implementation


lenet_model = SimpleCNNModel(
    [
        ConvLayer(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
        ConvLayer(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
        PoolLayer(kernel_size=(2, 2), stride=(2, 2)),
        ConvLayer(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
        PoolLayer(kernel_size=(2, 2), stride=(2, 2)),
        FCLayer(16 * 4 * 4, 120),
    ]
)
    
PoolLayer(kernel_size=(2, 2), stride=(2, 2)).output_shape((2, 2))

for size in lenet_model.features((32, 32)):
    print(size)
