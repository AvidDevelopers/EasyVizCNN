class Layer:
    """Base class for all layers in the CNN model."""

    def __str__(self) -> str:
        raise NotImplementedError("Subclasses should implement this!")


class ConvLayer(Layer):
    """
    Represents a convolutional layer in a CNN.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int | tuple[int, int]
        Size of the kernel.
    stride : int | tuple[int, int], optional
        Stride of the convolution. Default is 1.
    padding : int | tuple[int, int], optional
        Padding of the convolution. Default is 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        return None

    def __str__(self) -> str:
        return f"Conv({self.in_channels}, {self.out_channels}, {self.kernel_size}, {self.stride}, {self.padding})"


class PoolLayer(Layer):
    """
    Represents a pooling layer in a CNN.

    Parameters
    ----------
    kernel_size : int | tuple[int, int]
        Size of the kernel.
    stride : int | tuple[int, int], optional
        Stride of the pooling. Default is the same as the kernel size.
    pool_type : str, optional
        Type of the pooling. Can be "max" or "avg". Default is "max".
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = None,
        pool_type: str = "max",
    ):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.pool_type = pool_type

    def __str__(self) -> str:
        return f"{self.pool_type.capitalize()}Pool({self.kernel_size}, {self.stride})"


class FCLayer(Layer):
    """
    Represents a fully connected (dense) layer in a CNN.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    """

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

    def __str__(self) -> str:
        return f"FC({self.in_features}, {self.out_features})"
