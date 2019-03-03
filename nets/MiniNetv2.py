import tensorflow as tf
from tensorflow.keras import layers, regularizers


def upsampling(inputs, scale):

    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * scale, tf.shape(inputs)[2] * scale],
                                    align_corners=True)


def reshape_into(inputs, input_to_copy):
    return tf.image.resize_bilinear(inputs, [input_to_copy.get_shape()[1].value,
                                             input_to_copy.get_shape()[2].value], align_corners=True)


# convolution
def conv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                         kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


# Traspose convolution
def transposeConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  kernel_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


# Depthwise convolution
def depthwiseConv(kernel_size, strides=1, depth_multiplier=1, dilation_rate=1, use_bias=False):
    return layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                                  padding='same', use_bias=use_bias, kernel_regularizer=regularizers.l2(l=0.0003),
                                  dilation_rate=dilation_rate)


# Depthwise convolution
def separableConv(filters, kernel_size, strides=1, dilation_rate=1, use_bias=False):
    return layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=use_bias,
                                  depthwise_regularizer=regularizers.l2(l=0.0003),
                                  pointwise_regularizer=regularizers.l2(l=0.0003), dilation_rate=dilation_rate)


class DepthwiseConv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(DepthwiseConv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = separableConv(filters=filters, kernel_size=kernel_size, strides=strides,
                                  dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, training=None, relu=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if relu:
            x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)

        return x



class Conv_BN(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1):
        super(Conv_BN, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = conv(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, training=None, activation=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        if activation:
            x = layers.LeakyReLU(alpha=0.3)(x)

        return x



class Residual_SeparableConv(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1):
        super(Residual_SeparableConv, self).__init__()

        self.conv = DepthwiseConv_BN(filters, kernel_size, strides=strides, dilation_rate=dilation_rate)

    def call(self, inputs, training=None):

        x = self.conv(inputs, training=training, relu=False)
        x = tf.keras.activations.relu(x + inputs, alpha=0.0, max_value=None, threshold=0)

        return x


class MininetV2Module(tf.keras.Model):
    def __init__(self, filters, kernel_size, dilation_rate=1, strides=1):
        super(MininetV2Module, self).__init__()

        self.conv1 = Residual_SeparableConv(filters, kernel_size, strides=strides, dilation_rate=1)
        self.conv2 = Residual_SeparableConv(filters, kernel_size, strides=1, dilation_rate=dilation_rate)


    def call(self, inputs, training=None):

        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        return x

class MininetV2Downsample(tf.keras.Model):
    def __init__(self, filters, depthwise=True):
        super(MininetV2Downsample, self).__init__()
        if depthwise:
            self.conv = DepthwiseConv_BN(filters, kernel_size=3, dilation_rate=1, strides=2)
        else:
            self.conv = Conv_BN(filters, kernel_size=3, dilation_rate=1, strides=2)

    def call(self, inputs, training=None):

        x = self.conv(inputs, training=training)
        return x


class MininetV2Upsample(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1):
        super(MininetV2Upsample, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv = transposeConv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.bn = layers.BatchNormalization(epsilon=1e-3, momentum=0.993)

    def call(self, inputs, training=None, last=False):
        x = self.conv(inputs)
        if not last:
            x = self.bn(x, training=training)
            x = tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)

        return x


class MiniNetv2(tf.keras.Model):
    def __init__(self, num_classes, input_shape=(None, None, 3), weights='imagenet', **kwargs):
        super(MiniNetv2, self).__init__(**kwargs)

        self.down1_2 = MininetV2Downsample(16, depthwise=False)
        self.down2_2 = MininetV2Downsample(64, depthwise=True)
        self.down1 = MininetV2Downsample(16, depthwise=False)
        self.down2 = MininetV2Downsample(64, depthwise=True)
        self.module1 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.module2 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.module3 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.module4 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.module5 = MininetV2Module(64, 3, strides=1, dilation_rate=1)
        self.down3 = MininetV2Downsample(128, depthwise=True)
        self.module6 = MininetV2Module(128, 3, strides=1, dilation_rate=1)
        self.module7 = MininetV2Module(128, 3, strides=1, dilation_rate=4)
        self.module8 = MininetV2Module(128, 3, strides=1, dilation_rate=8)
        self.module9 = MininetV2Module(128, 3, strides=1, dilation_rate=16)
        self.module10 = MininetV2Module(128, 3, strides=1, dilation_rate=1)
        self.module11 = MininetV2Module(128, 3, strides=1, dilation_rate=2)
        self.module12 = MininetV2Module(128, 3, strides=1, dilation_rate=8)
        self.module13 = MininetV2Module(128, 3, strides=1, dilation_rate=16)
        self.up1 = MininetV2Upsample(64, 3, strides=2)  # call last=False
        self.module14 = MininetV2Module(64, 3, strides=1, dilation_rate=8)
        self.module15 = MininetV2Module(64, 3, strides=1, dilation_rate=8)
        self.up_last = MininetV2Upsample(num_classes, 3, strides=2)  # call last=False

    def call(self, inputs, training=None, mask=None, aux_loss=False, upsample=1):
        branch_2 = self.down1_2(inputs, training=training)
        branch_2 = self.down2_2(branch_2, training=training)

        x = self.down1(inputs, training=training)
        x = self.down2(x, training=training)
        x = self.module1(x, training=training)
        x = self.module2(x, training=training)
        x = self.module3(x, training=training)
        x = self.module4(x, training=training)
        x = self.module5(x, training=training)
        x = self.down3(x, training=training)
        x = self.module6(x, training=training)
        x = self.module7(x, training=training)
        x = self.module8(x, training=training)
        x = self.module9(x, training=training)
        x = self.module10(x, training=training)
        x = self.module11(x, training=training)
        x = self.module12(x, training=training)
        x = self.module13(x, training=training)
        x = branch_2 + self.up1(x, training=training)
        x = self.module14(x, training=training)
        x = self.module15(x, training=training)
        x = self.up_last(x, last=True, training=training)


        if upsample > 1:
            x = upsampling(x, upsample)

        if aux_loss:
            return x, x
        else:
            return x