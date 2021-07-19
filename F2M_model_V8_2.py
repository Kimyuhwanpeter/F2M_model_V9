# -*- coding:utf-8 -*-
import tensorflow as tf

l2 = tf.keras.regularizers.l2(0.000001)

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def Depth_point_wise_conv(h, filters=786, strides=1, depth_activation=False):

    if not depth_activation:
        h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding="same",
                                        use_bias=False, depthwise_regularizer=l2)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="same",
                               use_bias=False, kernel_regularizer=l2)(h)
    h = InstanceNormalization()(h)
    if depth_activation:
        h = tf.keras.layers.ReLU()(h)

    return h

def residual_block(input, filters=786, strides=1, return_skip=False, depth_activation=False, repeat=3):

    residual_input = input
    for i in range(repeat):
        residual_input = Depth_point_wise_conv(residual_input, filters=filters, strides=strides, depth_activation=depth_activation)

        if i  == 1:
            skip = residual_input 
    shortcut = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=strides, padding="same", use_bias=False, kernel_regularizer=l2)(input)
    shortcut = InstanceNormalization()(shortcut)
    outputs = tf.keras.layers.add([residual_input, shortcut])
    if return_skip:
        return outputs, skip
    else:
        return outputs

def adain(content, style, epsilon=1e-5):

    c_mean, c_var = tf.nn.moments(content, axes=[1,2], keepdims=True)
    s_mean, s_var = tf.nn.moments(style, axes=[1,2], keepdims=True)
    c_std, s_std = tf.sqrt(c_var + epsilon), tf.sqrt(s_var + epsilon)

    return s_std * (content - c_mean) / c_std + s_mean

def F2M_generator_V2(input_shape=(1024, 1024, 3), target_shape=(256, 256, 3)):

    h = inputs = tf.keras.Input(input_shape)
    t = targets = tf.keras.Input(target_shape)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=12, kernel_size=7, strides=2, padding="valid", use_bias=False, kernel_regularizer=l2)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [512, 512, 12]

    t = tf.keras.layers.ZeroPadding2D((3,3))(t)
    t = tf.keras.layers.Conv2D(filters=48, kernel_size=7, strides=1, padding="same", use_bias=False, kernel_regularizer=l2)(t)
    # [256, 256, 48]

    h = tf.keras.layers.Conv2D(filters=48, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2)(h)
    h = adain(h, t)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 48]

    h, skip2 = residual_block(h, filters=48, strides=1, return_skip=True, depth_activation=False)
    h = tf.keras.layers.ReLU()(h)

    t = InstanceNormalization()(t)
    t = tf.keras.layers.ReLU()(t)
    t = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2)(t)

    h = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2)(h)
    h = adain(h, t)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 192]

    h = residual_block(h, filters=192, strides=1, return_skip=False, depth_activation=False)
    h, skip = residual_block(h, filters=192, strides=1, return_skip=True, depth_activation=False)
    h = residual_block(h, filters=192, strides=2, return_skip=False, depth_activation=False, repeat=1)
    h = tf.keras.layers.ReLU()(h)

    t = InstanceNormalization()(t)
    t = tf.keras.layers.ReLU()(t)
    t = tf.keras.layers.Conv2D(filters=768, kernel_size=3, strides=2, padding="same", use_bias=False, kernel_regularizer=l2)(t)

    h = tf.keras.layers.Conv2D(filters=768, kernel_size=3, strides=1, padding="same", use_bias=False, kernel_regularizer=l2,
                               dilation_rate=2)(h)
    h = adain(h, t)
    h = tf.keras.layers.ReLU()(h)  # [64, 64, 768]

    conv_1_1 = tf.keras.layers.Conv2D(filters=576, kernel_size=1, strides=1, padding="same", use_bias=False, kernel_regularizer=l2)(h)
    conv_1_1 = InstanceNormalization()(conv_1_1)
    conv_1_1 = tf.keras.layers.ReLU()(conv_1_1)  # [64, 64, 576]
    conv_3_3_r6 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=False, depthwise_regularizer=l2,
                                         dilation_rate=6)(conv_1_1)
    conv_3_3_r6 = InstanceNormalization()(conv_3_3_r6)
    conv_3_3_r6 = tf.keras.layers.ReLU()(conv_3_3_r6)  # [64, 64, 576]
    conv_3_3_r12 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=False, depthwise_regularizer=l2,
                                         dilation_rate=12)(conv_3_3_r6)
    conv_3_3_r12 = InstanceNormalization()(conv_3_3_r12)
    conv_3_3_r12 = tf.keras.layers.ReLU()(conv_3_3_r12)  # [64, 64, 576]
    conv_3_3_r18 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=False, depthwise_regularizer=l2,
                                         dilation_rate=12)(conv_3_3_r12)
    conv_3_3_r18 = InstanceNormalization()(conv_3_3_r18)
    conv_3_3_r18 = tf.keras.layers.ReLU()(conv_3_3_r18)  # [64, 64, 576]

    pooling = tf.keras.layers.AveragePooling2D((3,3), strides=2, padding="same")(skip)  # [64, 64, 192]

    h = tf.concat([conv_3_3_r18, pooling], -1)
    h = tf.keras.layers.Conv2D(filters=208, kernel_size=1, strides=1, padding="same",
                               use_bias=False, kernel_regularizer=l2)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    upsample_4_h = tf.keras.layers.UpSampling2D((4, 4), interpolation="bilinear")(h)    # [256, 256, 208]

    h = tf.concat([skip2, upsample_4_h], -1)    # [256, 256, 256]

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1, padding="valid")(h)
    h = tf.keras.layers.UpSampling2D((4,4), interpolation="bilinear")(h)
    h = tf.nn.tanh(h)

    #h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", use_bias=False, kernel_regularizer=l2,
    #                           dilation_rate=8)(h)
    #h = InstanceNormalization()(h)
    #h = tf.keras.layers.ReLU()(h)   # [128, 128, 512]

    return tf.keras.Model(inputs=[inputs, targets], outputs=h)

def ConvDiscriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):
    dim_ = dim
    #Norm = BatchNorm(axis=3,momentum=BATCH_NORM_DECAY,epsilon=BATCH_NORM_EPSILON)

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)


    return tf.keras.Model(inputs=inputs, outputs=h)
