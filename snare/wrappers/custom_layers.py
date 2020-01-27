from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K


class CustomConv(layers.Conv1D):

    def __init__(self, filters, kernel_size, connections, **kwargs):

        self.connections = connections

        # Initialize underlying ConvLayer
        super(CustomConv, self).__init__(filters, kernel_size, **kwargs)

    def call(self, inputs):
        # Mask kernel with connection matrix
        masked_kernel = self.kernel * self.connections

        # Apply convolution
        if self.rank == 1:
            outputs = K.conv1d(
                inputs,
                masked_kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                masked_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = K.conv3d(
                inputs,
                masked_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class CustomConnected(layers.Dense):

    def __init__(self, units, connections, **kwargs):

        self.connections = connections

        # Initialize underlying Dense-layer
        super(CustomConnected, self).__init__(units, **kwargs)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
