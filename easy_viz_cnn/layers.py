from typing import TypeVar


T = TypeVar("T", int, tuple[int, int])


class Layer:
    """Base class for all layers in the CNN model."""

    def __str__(self) -> str:
        raise NotImplementedError("Subclasses should implement this!")

    def output_shape(self, input_size):
        raise NotImplementedError("Subclasses should implement this!")


class InputLayer(Layer):
    """Represents an input layer in a CNN."""

    def __init__(self, input_size):
        self.input_size = input_size

    def __str__(self) -> str:
        return f"Input({self.input_size})"

    def output_shape(self, *args, **kwargs):
        return self.input_size


class Conv1DLayer(Layer):
    """
    Represents a 1D convolutional layer in a CNN.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the kernel.
    stride : int, optional
        Stride of the convolution. Default is 1.
    padding : int, optional
        Padding of the convolution. Default is 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __str__(self) -> str:
        return f"Conv1D({self.in_channels}, {self.out_channels}, {self.kernel_size}, {self.stride}, {self.padding})"

    def output_shape(self, input_size):
        output_size = (input_size - self.kernel_size + 2 * self.padding) // self.stride + 1
        return (output_size, self.out_channels)


class Conv2DLayer(Layer):
    """
    Represents a 2D convolutional layer in a CNN.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple[int, int]
        Size of the kernel.
    stride : int or tuple[int, int], optional
        Stride of the convolution. Default is 1.
    padding : int or tuple[int, int], optional
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

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def __str__(self) -> str:
        return f"Conv2D({self.in_channels}, {self.out_channels}, {self.kernel_size}, {self.stride}, {self.padding})"

    def output_shape(self, input_size):
        output_height = (input_size[0] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[
            0
        ] + 1
        output_width = (input_size[1] - self.kernel_size[1] + 2 * self.padding[0]) // self.stride[
            0
        ] + 1
        return (output_height, output_width, self.out_channels)


class PoolLayer(Layer):
    """
    Represents a pooling layer in a CNN.

    Parameters
    ----------
    kernel_size : int
        Size of the kernel.
    stride : int, optional
        Stride of the pooling. Default is the same as the kernel size.
    pool_type : str, optional
        Type of the pooling. Can be "max" or "avg". Default is "max".
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int = None,
        pool_type: str = "max",
    ):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.pool_type = pool_type

    def __str__(self) -> str:
        return f"{self.pool_type.capitalize()}Pool({self.kernel_size}, {self.stride})"

    def output_shape(self, input_size):
        output_height = (input_size[0] - self.kernel_size) // self.stride + 1
        output_width = (input_size[1] - self.kernel_size) // self.stride + 1
        return (output_height, output_width, input_size[2])


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

    def output_shape(self, input_size: int) -> tuple[int, int]:
        return (1, self.out_features)
