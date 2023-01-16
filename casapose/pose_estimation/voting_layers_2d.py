import tensorflow as tf
import tensorflow_addons as tfa


class CoordLSVotingWeighted(tf.keras.layers.Layer):
    def __init__(
        self,
        name,
        num_classes,
        num_points=9,
        sigmoid_weights=False,
        filter_estimates=False,
        output_second_largest_component=False,
    ):
        super(CoordLSVotingWeighted, self).__init__(name=name)
        self.num_classes = num_classes
        self.num_points = num_points
        self.sigmoid_weights = sigmoid_weights
        self.filter_estimates = filter_estimates
        self.sigmoid_scale = 1.0
        self.output_second_largest_component = output_second_largest_component

    def build(self, input_shape):
        self.width = input_shape[0][2]
        self.height = input_shape[0][1]

    # Gewichte werden geschätzt auch itertive reweighted least squares könnte genutzt werden
    def call(self, inp, **kwargs):

        seg, direct, w = inp

        if self.sigmoid_weights:
            w = tf.nn.sigmoid(self.sigmoid_scale * w)
        else:
            w = tf.math.softplus(w)  # avoid negative weights (seems to be numerically stable)

        seg = tf.stop_gradient(seg)
        beta = tf.cast(1e6, dtype=seg.dtype)
        hot_seg = tf.expand_dims(tf.expand_dims(tf.nn.softmax(seg * beta), -1), -1)[
            :, :, :, 1:, :, :
        ]  # [bs, w, h, oc, 1, 1]

        if self.filter_estimates:
            hot_seg_int = tf.cast(hot_seg + 0.1, tf.int32)
            hot_seg_int = tf.transpose(hot_seg_int, perm=[0, 3, 1, 2, 4, 5])

            b, oc, ht, wt, _, _ = hot_seg_int.shape

            hot_seg_int = tf.reshape(hot_seg_int, [-1, ht, wt])

            @tf.function
            def connected_components(img):
                return tfa.image.connected_components(img)

            fn = lambda x: connected_components(x)
            components = tf.map_fn(fn, elems=(hot_seg_int), dtype=tf.int32)

            if self.output_second_largest_component:  # just for testing
                bins = 3
            else:
                bins = 2

            components = tf.reshape(components, [-1, ht * wt])
            bincount = tf.math.bincount(components, axis=-1, minlength=bins)  # [:,1:]

            bincount = tf.where(bincount < 50, 0, bincount)
            values, indices = tf.math.top_k(bincount, k=bins)
            indices = tf.expand_dims(indices, 1)
            components = tf.expand_dims(components, -1)
            # Assumption: background class is the largest
            if self.output_second_largest_component:  # just for testing
                copy_components_2 = tf.where(components == indices[:, :, 2:3], 1.0, 0.0)
                copy_components = tf.reshape(copy_components_2, [b, oc, ht, wt])
            else:
                copy_components_1 = tf.where(components == indices[:, :, 1:2], 1.0, 0.0)
                copy_components = tf.reshape(copy_components_1, [b, oc, ht, wt])

            copy_components = tf.transpose(copy_components, perm=[0, 2, 3, 1])
            hot_seg = tf.expand_dims(tf.expand_dims(copy_components, -1), -1) * hot_seg

        return self.calc(direct, w, hot_seg)

    @tf.function(jit_compile=True)
    def calc(self, direct, w, hot_seg):
        n = tf.reshape(direct, [-1, self.height, self.width, self.num_points, 2])
        # n = tf.reverse(n, axis=[-2])

        # normalize
        norm = tf.norm(n, ord="euclidean", axis=-1, keepdims=True)
        n = tf.expand_dims(tf.math.divide_no_nan(n, norm), axis=-1)

        n_nt = tf.matmul(n, tf.reshape(n, [-1, self.height, self.width, self.num_points, 1, 2]))
        R_full = tf.reshape(tf.eye(2), [1, 1, 1, 1, 2, 2]) - n_nt  # [bs, w, h, kc, 2, 2]
        R_full = R_full * tf.expand_dims(tf.expand_dims(w, -1), -1)
        w_grid, h_grid = tf.meshgrid(tf.range(self.width), tf.range(self.height))
        w_grid = (tf.cast(w_grid, dtype=tf.float32) + 0.5) / self.height
        h_grid = (tf.cast(h_grid, dtype=tf.float32) + 0.5) / self.height

        coords = tf.expand_dims(tf.stack([h_grid, w_grid], axis=-1), axis=-0)
        coords = tf.expand_dims(
            tf.expand_dims(coords, 3), -1
        )  # [bs, w, h, 1, 2, 1]        #q_full = tf.squeeze(tf.matmul(R_full, coords),-1)  # [bs, w, h, kc, 2]  ## is this line super slow ???
        q_full = (R_full[:, :, :, :, :, 0] * coords[:, :, :, :, 0]) + (
            R_full[:, :, :, :, :, 1] * coords[:, :, :, :, 1]
        )  # rolling out matmult is much faster

        q = tf.math.multiply_no_nan(tf.expand_dims(q_full, 3), hot_seg)  # [bs, w, h, oc, kc, 2]
        R = tf.math.multiply_no_nan(tf.expand_dims(R_full, 3), tf.expand_dims(hot_seg, -1))  # [bs, w, h, oc, kc, 2, 2]
        tf.Assert(tf.reduce_all(tf.math.is_finite(R)), ["R nan"])
        tf.Assert(tf.reduce_all(tf.math.is_finite(q)), ["q nan"])

        # this part causes nan with float32 operations
        R_comb = tf.reduce_sum(tf.cast(R, tf.float64), axis=[1, 2])  # [bs, oc, kc, 2, 2]
        q_comb = tf.expand_dims(tf.reduce_sum(tf.cast(q, tf.float64), axis=[1, 2]), -1)  # [bs, oc, kc, 2, 1]
        R_shape = tf.shape(R_comb)
        R_pinv = tf.linalg.pinv(tf.reshape(R_comb, [-1, 2, 2]))  # [bs * oc * kc, 2, 2]
        tf.Assert(tf.reduce_all(tf.math.is_finite(R_pinv)), ["R_pinv nan"])

        R_pinv = tf.reshape(R_pinv, R_shape)
        p = tf.squeeze(tf.matmul(R_pinv, q_comb), -1)  # [bs, oc,kc, 2]
        tf.Assert(tf.reduce_all(tf.math.is_finite(p)), ["p nan"])
        return tf.cast(p, tf.float32) * tf.constant([[[[self.height, self.height]]]], dtype=tf.float32)
