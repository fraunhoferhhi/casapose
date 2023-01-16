import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils


def compile_jit_functions():
    return os.getenv("CASAPOSE_INFERENCE", "False").lower() == "true"


class ConditionalInstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(ConditionalInstanceNormalization, self).__init__()
        self.num_classes = num_classes
        self.gamma_initializer = "ones"  # tf.random_normal_initializer(1, 0.02)
        self.beta_initializer = "zeros"

    def build(self, input_shape):
        self.num_channels = input_shape[0][-1]
        self.beta = self.add_weight(
            name=self.name + "_beta",
            shape=(self.num_classes, 1, 1, self.num_channels),
            initializer=self.beta_initializer,
        )
        self.gamma = self.add_weight(
            name=self.name + "_gamma",
            shape=(self.num_classes, 1, 1, self.num_channels),
            initializer=self.gamma_initializer,
        )

    def call(self, inputs, **kwargs):
        x, idx_sigmas = inputs
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)

        gamma1 = tf.gather(self.gamma, idx_sigmas)
        beta1 = tf.gather(self.beta, idx_sigmas)

        x = tf.nn.batch_normalization(x, mean, var, beta1, gamma1, 2e-5)
        return x


class ClassAdaptiveNormalization(tf.keras.layers.Layer):
    def __init__(self, name, num_classes):
        super(ClassAdaptiveNormalization, self).__init__(name=name)
        self.num_classes = num_classes
        self.gamma_initializer = "ones"  # tf.random_normal_initializer(1, 0.02)
        self.beta_initializer = "zeros"

    def build(self, input_shape):
        self.num_channels = input_shape[0][-1]
        self.beta = self.add_weight(
            name=self.name + "_beta",
            shape=(self.num_classes, self.num_channels),
            initializer=self.beta_initializer,
            dtype=tf.float32,
        )
        self.gamma = self.add_weight(
            name=self.name + "_gamma",
            shape=(self.num_classes, self.num_channels),
            initializer=self.gamma_initializer,
            dtype=tf.float32,
        )
        self.bn = tf.keras.layers.experimental.SyncBatchNormalization(center=False, scale=False, epsilon=2e-5)

    def call(self, inputs, **kwargs):
        x, seg_mask = inputs

        beta1 = tf.gather_nd(
            self.beta,
            seg_mask,
            name=self.name + "_gather_nd_beta_dim_" + str(self.num_classes),
        )
        gamma1 = tf.gather_nd(
            self.gamma,
            seg_mask,
            name=self.name + "_gather_nd_dim_" + str(self.num_classes),
        )

        x = self.bn(x)
        x = gamma1 * x + beta1
        return x


class ClassAdaptiveWeightedNormalization(tf.keras.layers.Layer):
    def __init__(self, name, num_classes):
        super(ClassAdaptiveWeightedNormalization, self).__init__(name=name)
        self.num_classes = num_classes
        self.gamma_initializer = "ones"  # tf.random_normal_initializer(1, 0.02)
        self.beta_initializer = "zeros"

    def build(self, input_shape):
        self.num_channels = input_shape[0][-1]
        self.height = input_shape[0][1]
        self.width = input_shape[0][2]
        self.beta = self.add_weight(
            name=self.name + "_beta",
            shape=(self.num_classes, self.num_channels),
            initializer=self.beta_initializer,
            dtype=tf.float32,
        )
        self.gamma = self.add_weight(
            name=self.name + "_gamma",
            shape=(self.num_classes, self.num_channels),
            initializer=self.gamma_initializer,
            dtype=tf.float32,
        )
        self.bn = tf.keras.layers.experimental.SyncBatchNormalization(center=False, scale=False, epsilon=2e-5)

    def call(self, inputs, **kwargs):
        if compile_jit_functions():
            return self.calc_jit(inputs)
        return self.calc(inputs)

    @tf.function(jit_compile=True)
    def calc_jit(self, inputs):
        return self.calc(inputs)

    def calc(self, inputs):
        x, seg_softmax = inputs

        # beta1 = tf.linalg.matmul( seg_softmax, self.beta) #BatchMatmulV2 not supported by opencv
        # gamma1 = tf.linalg.matmul( seg_softmax, self.gamma)

        # tensorflow
        beta1 = tf.tensordot(
            seg_softmax,
            self.beta,
            [3, 0],
            self.name + "_tensordot_dim_" + str(self.num_classes) + "_" + str(self.num_channels),
        )
        gamma1 = tf.tensordot(
            seg_softmax,
            self.gamma,
            [3, 0],
            self.name + "_tensordot_dim_" + str(self.num_classes) + "_" + str(self.num_channels),
        )
        x = self.bn(x)
        x = gamma1 * x + beta1

        # OpenCV (before 4.5.1)
        # seg_softmax = tf.reshape(seg_softmax, [-1, self.num_classes]) # causing a transformation in opencv
        # beta1 = tf.matmul( seg_softmax, self.beta)
        # gamma1 = tf.matmul( seg_softmax, self.gamma)
        # beta1 = tf.transpose(beta1, perm=[1,0]) # only needed for opencv
        # gamma1 = tf.transpose(gamma1, perm=[1,0]) # only needed for opencv
        # x = self.bn(x)
        # x = gamma1 * x + beta1
        # x = tf.transpose(x, perm=[0,1,2,3]) # this is needed for opencv to enforce the right channel order

        # OpenCV (after 4.5.1)
        # seg_softmax = tf.reshape(seg_softmax, [-1, self.num_classes]) # causing a transformation in opencv
        # beta1 = tf.matmul( seg_softmax, self.beta)
        # gamma1 = tf.matmul( seg_softmax, self.gamma)
        # beta1 = tf.transpose(beta1, perm=[1,0])
        # gamma1 = tf.transpose(gamma1, perm=[1,0])
        # beta1 = tf.reshape(beta1, [-1, self.num_channels, self.height, self.width]) # width and height can not be determined dynamically by opencv
        # gamma1 = tf.reshape(gamma1, [-1, self.num_channels, self.height, self.width])
        # x = self.bn(x)
        # x = tf.transpose(x, perm=[0,3,1,2]) # this is needed for opencv to enforce the right channel order
        # x = gamma1 * x + beta1
        # x = tf.transpose(x, perm=[0,2,3,1]) # this is needed for opencv to enforce the right channel order

        return x


class ClassAdaptiveWeightedNormalizationWithInput(tf.keras.layers.Layer):
    def __init__(self, name, num_classes):
        super(ClassAdaptiveWeightedNormalizationWithInput, self).__init__(name=name)
        self.num_classes = num_classes

    def build(self, input_shape):
        self.num_channels = input_shape[0][-1]
        self.height = input_shape[0][1]
        self.width = input_shape[0][2]

        self.bn = tf.keras.layers.experimental.SyncBatchNormalization(center=False, scale=False, epsilon=2e-5)

    def call(self, inputs, **kwargs):
        x, seg_softmax, gamma, beta = inputs

        beta1 = tf.keras.backend.batch_dot(seg_softmax, beta, [3, 1])
        gamma1 = tf.keras.backend.batch_dot(seg_softmax, gamma, [3, 1])

        # alternative implementation
        # beta1 = tf.map_fn(
        #     lambda xy: tf.tensordot(xy[0], xy[1], [[-1], [-2]], self.name + '_tensordot_dim_' + str(self.num_classes) + '_' +  str(self.num_channels) ),
        #     elems=(seg_softmax, beta), dtype=seg_softmax.dtype)
        # gamma1 = tf.map_fn(
        #     lambda xy: tf.tensordot(xy[0], xy[1], [[-1], [-2]], self.name + '_tensordot_dim_' + str(self.num_classes) + '_' +  str(self.num_channels) ),
        #     elems=(seg_softmax, gamma), dtype=seg_softmax.dtype)

        x = self.bn(x)
        x = gamma1 * x + beta1
        x = tf.transpose(x, perm=[0, 1, 2, 3])  # this is needed for opencv to enforce the right channel order
        return x


class ClassAdaptiveWeightedNormalizationWithInputAndLearnedParameters(tf.keras.layers.Layer):
    def __init__(self, name, num_classes):
        super(ClassAdaptiveWeightedNormalizationWithInputAndLearnedParameters, self).__init__(name=name)
        self.num_classes = num_classes
        self.gamma_initializer = "ones"  # tf.random_normal_initializer(1, 0.02)
        self.beta_initializer = "zeros"
        self.alpha_initializer = tf.keras.initializers.Constant(0.5)

    def build(self, input_shape):
        self.num_channels = input_shape[0][-1]
        self.height = input_shape[0][1]
        self.width = input_shape[0][2]
        self.beta = self.add_weight(
            name=self.name + "_beta",
            shape=(self.num_classes, self.num_channels),
            initializer=self.beta_initializer,
            dtype=tf.float32,
        )
        self.gamma = self.add_weight(
            name=self.name + "_gamma",
            shape=(self.num_classes, self.num_channels),
            initializer=self.gamma_initializer,
            dtype=tf.float32,
        )
        self.alpha_1 = self.add_weight(
            name=self.name + "_alpha_1",
            shape=(1),
            initializer=self.alpha_initializer,
            constraint=tf.keras.constraints.MinMaxNorm(),
            dtype=tf.float32,
        )
        self.alpha_2 = self.add_weight(
            name=self.name + "_alpha_2",
            shape=(1),
            initializer=self.alpha_initializer,
            constraint=tf.keras.constraints.MinMaxNorm(),
            dtype=tf.float32,
        )

        self.bn = tf.keras.layers.experimental.SyncBatchNormalization(center=False, scale=False, epsilon=2e-5)

    def call(self, inputs, **kwargs):
        x, seg_softmax, gamma, beta = inputs

        beta1 = tf.keras.backend.batch_dot(seg_softmax, beta, [3, 1])
        gamma1 = tf.keras.backend.batch_dot(seg_softmax, gamma, [3, 1])

        beta2 = tf.tensordot(seg_softmax, self.beta, [3, 0])
        gamma2 = tf.tensordot(seg_softmax, self.gamma, [3, 0])

        x = self.bn(x)
        w1 = (self.alpha_1 * gamma1) + ((1 - self.alpha_1) * gamma2)
        w2 = (self.alpha_2 * beta1) + ((1 - self.alpha_2) * beta2)

        x = w1 * x + w2
        x = tf.transpose(x, perm=[0, 1, 2, 3])  # this is needed for opencv to enforce the right channel order
        return x


class HalfSize(tf.keras.layers.Layer):
    def __init__(self, name, depth=1, trainable=True, data_format=None):
        super(HalfSize, self).__init__(name=name, trainable=trainable)
        self.input_spec = tf.keras.layers.InputSpec(ndim=4)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.depth = depth
        self.trainable_conv = trainable

    def build(self, input_shape):
        super(HalfSize, self).build(input_shape)
        self.compute_output_shape(input_shape)
        self.init = tf.expand_dims(tf.expand_dims(tf.eye(self.depth), 0), 0)

        def costom_init(shape, dtype=None):
            return tf.expand_dims(tf.expand_dims(tf.eye(self.depth), 0), 0)

        self.conv = tf.keras.layers.Conv2D(
            self.depth,
            (1, 1),
            strides=(2, 2),
            padding="valid",
            use_bias=False,
            kernel_initializer=costom_init,
        )

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_first":
            self.height = int(0.5 * input_shape[2])
            self.width = int(0.5 * input_shape[3])
            return tf.TensorShape([input_shape[0], input_shape[1], self.height, self.width])
        else:
            self.height = int(0.5 * input_shape[1])
            self.width = int(0.5 * input_shape[2])
            return tf.TensorShape([input_shape[0], self.height, self.width, input_shape[3]])

    def call(self, x, **kwargs):
        if self.trainable_conv:
            x = self.conv(x)
        else:
            x = tf.nn.conv2d(x, self.init, strides=[2, 2], padding="VALID")
        return x


class PartialConvolution(tf.keras.layers.Layer):
    def __init__(self, name, dim, num_classes, conv_name=None, skip=False):
        # def __init__(self, name):
        super(PartialConvolution, self).__init__(name=name)
        self.dim = dim
        self.skip = skip
        self.num_classes = num_classes

    def build(self, input_shape):
        self.height = input_shape[0][1]
        self.width = input_shape[0][2]
        self.ch = input_shape[0][3]
        self.conv_w = self.add_weight(
            name=self.name + "_weights",
            shape=(self.ch, 3, 3, self.dim),
            initializer=tf.keras.initializers.he_uniform(),
            dtype=tf.float32,
        )
        filter_value = np.eye(9).reshape((3, 3, 1, 9))
        self.extract_image_patch_filter = tf.constant(filter_value)
        self.extract_image_patch_filter = tf.cast(self.extract_image_patch_filter, tf.float32)
        self.extract_image_patch_filter = tf.repeat(self.extract_image_patch_filter, self.ch, axis=2)

    @tf.function(jit_compile=compile_jit_functions())
    def calc(self, inputs, weights):
        if len(inputs) == 1 or self.skip:
            x = inputs[0]
            # w = tf.reshape(weights, [self.dim,3,3,self.ch])
            weights = tf.transpose(weights, [1, 2, 0, 3])
            x = tf.nn.conv2d(x, weights, strides=[1, 1], padding="SAME")
        else:
            x, seg_mask = inputs
            s_patch = tf.image.extract_patches(
                images=seg_mask,
                sizes=[1, 3, 3, 1],
                strides=[1, 1, 1, 1],
                rates=[1, 1, 1, 1],
                padding="SAME",
            )
            s_patch = tf.reshape(s_patch, [-1, self.height, self.width, 9, self.num_classes])
            seg_mask = tf.expand_dims(seg_mask, 3)

            s_max = tf.reduce_max(seg_mask, -1, keepdims=True)
            seg_mask = tf.where(seg_mask == s_max, s_patch, 0.0)

            seg_mask = tf.reduce_sum(seg_mask, -1)
            # norm = tf.math.divide_no_nan(9.0, tf.reduce_sum(seg_mask, -1, keepdims=True))  # for 3*3 // division by zero?
            norm = tf.math.divide_no_nan(
                9.0,
                tf.math.count_nonzero(seg_mask, -1, keepdims=True, dtype=tf.dtypes.float32),
            )  # for 3*3 // division by zero?

            # seg_mask = tf.expand_dims(seg_mask, -1)  # the extract_patches method is allocating huge amount of memory during inference
            # x = tf.image.extract_patches(images=x, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1],rates=[1, 1, 1, 1], padding='SAME')
            # x = tf.reshape(x, [-1, self.height, self.width, 9, self.ch])
            # x = x * seg_mask
            # x = tf.reshape(x, [-1, self.height * self.width, 9 * self.ch ])
            # weights = tf.transpose(weights, [1,2,0,3])
            # weights = tf.reshape(weights, [9 * self.ch , self.dim])
            # x = tf.einsum('ikl,lm->ikm',x,weights)
            # x = tf.reshape(x, [-1, self.height, self.width, self.dim]) * norm

            seg_mask = tf.expand_dims(seg_mask, -2)
            x = tf.nn.depthwise_conv2d(
                x, self.extract_image_patch_filter, strides=[1, 1, 1, 1], padding="SAME"
            )  # extract image patches
            x = tf.reshape(x, [-1, self.height, self.width, self.ch, 9])
            x = x * seg_mask
            weights = tf.reshape(weights, [self.ch, 9, self.dim])
            x = tf.einsum("iklmn,mno->iklo", x, weights) * norm

        return x

    def call(self, inputs, **kwargs):
        x = self.calc(inputs, self.conv_w)  # , _checkpoint=False)
        return x


# old version
class PartialConvolution2(tf.keras.layers.Layer):
    def __init__(self, name, dim, num_classes, conv_name=None):
        # def __init__(self, name):
        super(PartialConvolution2, self).__init__(name=name)
        # self.conv = tf.keras.layers.Conv2D(dim, (3, 3), strides=(3,3), name=conv_name)
        # elf.up = tf.compat.v1.keras.layers.UpSampling2D(size=(3, 3), data_format="channels_last", interpolation="nearest")
        self.dim = dim
        # self.skip = skip
        self.num_classes = num_classes

    def build(self, input_shape):
        # if input_shape[1] is None:
        #    self.skip = True
        # else:
        #    self.num_classes = input_shape[1][-1]
        self.height = input_shape[0][1]
        self.width = input_shape[0][2]
        self.ch = input_shape[0][3]
        self.conv_w = self.add_weight(
            name=self.name + "_weights",
            shape=(1, self.dim, 1, 9 * self.ch),
            initializer=tf.keras.initializers.he_uniform(),
            dtype=tf.float32,
        )

    def call(self, inputs, **kwargs):

        if len(inputs) == 1:
            x = inputs[0]
            w = tf.reshape(self.conv_w, [self.dim, 3, 3, self.ch])
            w = tf.transpose(w, [1, 2, 3, 0])
            x = tf.nn.conv2d(x, w, strides=[1, 1], padding="SAME")
        else:
            x, seg_mask = inputs
            s_patch = tf.image.extract_patches(
                images=seg_mask,
                sizes=[1, 3, 3, 1],
                strides=[1, 1, 1, 1],
                rates=[1, 1, 1, 1],
                padding="SAME",
            )
            s_patch = tf.reshape(s_patch, [-1, self.height, self.width, 9, self.num_classes])
            seg_mask = tf.expand_dims(seg_mask, 3)

            s_max = tf.reduce_max(seg_mask, -1, keepdims=True)
            seg_mask = tf.where(seg_mask == s_max, s_patch, 0.0)

            seg_mask = tf.reduce_sum(seg_mask, -1)
            # norm = tf.math.divide_no_nan(9.0, tf.reduce_sum(seg_mask, -1, keepdims=True))  # for 3*3 // division by zero?
            norm = tf.math.divide_no_nan(
                9.0,
                tf.math.count_nonzero(seg_mask, -1, keepdims=True, dtype=tf.dtypes.float32),
            )  # for 3*3 // division by zero?

            x = tf.image.extract_patches(
                images=x,
                sizes=[1, 3, 3, 1],
                strides=[1, 1, 1, 1],
                rates=[1, 1, 1, 1],
                padding="SAME",
            )
            x = tf.reshape(x, [-1, self.height, self.width, 9, self.ch])

            # if not self.skip:
            x = x * tf.expand_dims(seg_mask, -1)

            # manual convolution
            # x_test = tf.reshape(x, [-1, self.height, self.width, 9 * self.ch])
            # x_test = tf.nn.depth_to_space(x_test, 3)
            # k_test = tf.reshape(self.conv_w, [self.dim,3,3,self.ch])
            # k_test = tf.transpose(k_test, [1,2,3,0])
            # x_test = tf.nn.conv2d(x_test, k_test, strides=[3, 3], padding='VALID') * norm

            # ____AVOID CONVOLUTION
            x = tf.reshape(x, [-1, 1, self.height * self.width, 9 * self.ch])
            x = tf.tensordot(x, self.conv_w, [3, 3])
            # if not self.skip:
            x = tf.reshape(x, [-1, self.height, self.width, self.dim]) * norm
            # else:
            #    x = tf.reshape(x, [-1, self.height, self.width, self.dim])
            # OLD VERSION __ BUGGY?
            # s_max = tf.nn.depth_to_space(s_max, 3)
            # s_max = tf.reshape(s_max, [-1, self.height, 3, self.width, 3, 1])
            # x = tf.reshape(x, [-1, self.height, 1, self.width, 1, self.ch])
            # x = x * s_max
            # x = tf.reshape(x, [-1, self.height * 3, self.width * 3, self.ch])

            # ____DO CONVOLUTION
            # x = tf.reshape(x, [-1, self.height, self.width, 9 * self.ch])
            # x = tf.nn.depth_to_space(x, 3)
            # x = self.conv(x) * norm
        return x


class GuidedUpsampling(tf.keras.layers.Layer):
    def __init__(self, name):
        super(GuidedUpsampling, self).__init__(name=name)
        # self.conv = tf.keras.layers.Conv2D(num_channels_out, (3, 3), strides=(3,3), name=conv_name)

    def build(self, input_shape):
        self.h = input_shape[2][1]  # output height
        self.w = input_shape[2][2]  # output width
        self.h2 = input_shape[1][1]
        self.w2 = input_shape[1][2]
        self.c = input_shape[0][3]
        self.r_up = tf.range(1, input_shape[2][3] + 1, dtype=tf.float32)  # indices from 1 to #classes+1
        self.r_down = tf.reverse(tf.range(1, 4 + 1, dtype=tf.float32), axis=[0])  # indices from 5 to 1
        self.f = tf.reshape(
            tf.cast(tf.one_hot(0, depth=4), dtype=tf.bool), [1, 1, 1, 1, 1, 4]
        )  # one hot vector selecting first out of four

        filter_value = np.eye(4).reshape((2, 2, 1, 4))
        self.extract_image_patch_filter_seg = tf.constant(filter_value)
        self.extract_image_patch_filter_seg = tf.cast(self.extract_image_patch_filter_seg, tf.float32)

        c_x, c_y = tf.meshgrid(tf.range(0, self.w2, 1), tf.range(0, self.h2, 1))
        self.coords = tf.expand_dims(tf.stack([c_y, c_x, tf.zeros_like(c_x)], 2), 0)
        self.coords = tf.image.extract_patches(
            images=self.coords,
            sizes=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        self.coords = tf.reshape(self.coords, [-1, self.h2, 1, self.w2, 1, 4, 3])

    @tf.function(jit_compile=True)  # compile_jit_functions())
    def call(self, inputs, **kwargs):
        x, seg_d, seg_u = inputs
        # 1. get maximum in each segmentation mask
        # 2. select maximum and multiply with increasing index to differentiate between classes
        seg_d = tf.reduce_sum(
            tf.where(
                tf.equal(seg_d, tf.reduce_max(seg_d, axis=-1, keepdims=True)),
                seg_d,
                0.0,
            )
            * self.r_up,
            axis=-1,
            keepdims=True,
        )
        seg_u = tf.reduce_sum(
            tf.where(
                tf.equal(seg_u, tf.reduce_max(seg_u, axis=-1, keepdims=True)),
                seg_u,
                0.0,
            )
            * self.r_up,
            axis=-1,
            keepdims=True,
        )
        seg_u = tf.reshape(seg_u, [-1, self.h2, 2, self.w2, 2, 1])
        # 3. select 4 by 4 patches and upsample to target resolution
        seg_d = tf.nn.conv2d(
            seg_d,
            self.extract_image_patch_filter_seg,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        # seg_d = tf.image.extract_patches(images=seg_d,sizes=[1, 2, 2, 1], strides=[1, 1, 1, 1],rates=[1, 1, 1, 1], padding='SAME')
        seg_d = tf.reshape(seg_d, [-1, self.h2, 1, self.w2, 1, 4])
        # 4. compare upscaled and original map and multiply with increasing index to priorize first indexs in patches by reduce_min operation
        cond = tf.cast(tf.equal(seg_d, seg_u), tf.float32) * self.r_down  # [b,h2,2,w2,2,4]
        cond = tf.equal(cond, tf.reduce_max(cond, axis=-1, keepdims=True))  # can this give more than one true value

        cond = tf.where(
            tf.greater(tf.reduce_sum(tf.cast(cond, tf.float32), -1, keepdims=True), 1),
            self.f,
            cond,
        )  # use nn upsampling for all locations where new labels appear
        cond = tf.reshape(cond, [-1, self.h2, 2, self.w2, 2, 4, 1])

        coords = tf.reshape(
            tf.reduce_sum(tf.where(cond, self.coords, 0), axis=5),
            [-1, self.h * self.w, 3],
        )
        x = tf.gather_nd(tf.expand_dims(x, 3), coords, batch_dims=1)

        # 5. select 4 in output by 4 patches and upsample to target resolution
        # x = tf.image.extract_patches(images=x,sizes=[1, 2, 2, 1], strides=[1, 1, 1, 1],rates=[1, 1, 1, 1], padding='SAME')
        # x = tf.reshape(x, [-1,self.h2,1,self.w2,1,4, self.c])
        # 6. copy matching positions and reduce to sum to get rid of zeros
        # x = tf.reduce_sum(tf.where(cond, x, 0), axis=5)
        x = tf.reshape(x, [-1, self.h, self.w, self.c])
        return x


class GuidedBilinearUpsampling(tf.keras.layers.Layer):
    def __init__(self, name):
        super(GuidedBilinearUpsampling, self).__init__(name=name)

    def build(self, input_shape):
        self.h = input_shape[2][1]  # output height
        self.w = input_shape[2][2]  # output width
        self.h2 = input_shape[1][1]
        self.w2 = input_shape[1][2]
        self.c = input_shape[0][3]
        self.r_up = tf.range(1, input_shape[2][3] + 1, dtype=tf.float32)  # indices from 1 to #classes+1

        filter_value = np.eye(4).reshape((2, 2, 1, 4))
        self.extract_image_patch_filter_seg = tf.constant(filter_value)
        self.extract_image_patch_filter_seg = tf.cast(self.extract_image_patch_filter_seg, tf.float32)

        c_x, c_y = tf.meshgrid(tf.range(0, self.w2, 1), tf.range(0, self.h2, 1))
        self.coords = tf.expand_dims(tf.stack([c_y, c_x, tf.zeros_like(c_x)], 2), 0)
        self.coords = tf.image.extract_patches(
            images=self.coords,
            sizes=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        self.coords = tf.reshape(self.coords, [-1, self.h2, 1, self.w2, 1, 4, 3])

        self.interp = tf.constant(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.0],
                [0.25, 0.25, 0.25, 0.25],
            ]
        )
        self.interp = tf.reshape(self.interp, [1, 1, 2, 1, 2, 4, 1])

    # @tf.function(experimental_compile=False)
    def call(self, inputs, **kwargs):
        x, seg_d, seg_u = inputs
        # 1. get maximum in each segmentation mask
        # 2. select maximum and multiply with increasing index to differentiate between classes
        seg_d = tf.reduce_sum(
            tf.where(
                tf.equal(seg_d, tf.reduce_max(seg_d, axis=-1, keepdims=True)),
                seg_d,
                0.0,
            )
            * self.r_up,
            axis=-1,
            keepdims=True,
        )
        seg_u = tf.reduce_sum(
            tf.where(
                tf.equal(seg_u, tf.reduce_max(seg_u, axis=-1, keepdims=True)),
                seg_u,
                0.0,
            )
            * self.r_up,
            axis=-1,
            keepdims=True,
        )
        seg_u = tf.reshape(seg_u, [-1, self.h2, 2, self.w2, 2, 1])
        # 3. select 4 by 4 patches and upsample to target resolution
        seg_d = tf.nn.conv2d(
            seg_d,
            self.extract_image_patch_filter_seg,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )

        seg_d = tf.reshape(seg_d, [-1, self.h2, 1, self.w2, 1, 4])

        # 4. compare upscaled and original map and multiply with increasing index to priorize first indexs in patches by reduce_min operation
        cond = tf.equal(seg_d, seg_u)  # [b,h2,2,w2,2,4]

        cond = tf.reshape(cond, [-1, self.h2, 2, self.w2, 2, 4, 1])
        norm = tf.reduce_sum(tf.cast(cond, tf.float32), axis=5, keepdims=True)
        x = tf.image.extract_patches(
            images=x,
            sizes=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        x = tf.reshape(x, [-1, self.h2, 1, self.w2, 1, 4, self.c])
        x = tf.where(cond, x, 0)
        x = tf.where(
            cond,
            x,
            tf.math.divide_no_nan(tf.reduce_sum(x, axis=5, keepdims=True), norm),
        )
        x = x * self.interp
        x = tf.reduce_sum(x, axis=5, keepdims=True)
        x = tf.reshape(x, [-1, self.h, self.w, self.c])
        return x
