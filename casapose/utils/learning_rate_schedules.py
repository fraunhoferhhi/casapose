import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


class ExponentialDecayLateStart(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        decay_steps_start,
        decay_rate,
        staircase=False,
        name=None,
    ):
        """Applies exponential decay to the learning rate, performes the first decay after decay_steps_start"""
        super(ExponentialDecayLateStart, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_steps_start = decay_steps_start
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "ExponentialDecayLateStart") as name:
            initial_learning_rate = ops.convert_to_tensor_v2_with_dispatch(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = math_ops.cast(self.decay_steps, dtype)
            decay_steps_start = math_ops.cast(self.decay_steps_start, dtype)
            decay_rate = math_ops.cast(self.decay_rate, dtype)

            global_step_recomp = math_ops.cast(step, dtype)
            offset = tf.cond(
                decay_steps_start == 0,
                lambda: tf.constant(0.0),
                lambda: tf.constant(1.0),
            )
            p = tf.cond(
                global_step_recomp < decay_steps_start,
                lambda: tf.constant(0.0),
                lambda: offset + ((global_step_recomp - decay_steps_start) / decay_steps),
            )

            if self.staircase:
                p = math_ops.floor(p)
            return math_ops.multiply(initial_learning_rate, math_ops.pow(decay_rate, p), name=name)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_steps_start": self.decay_steps_start,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "name": self.name,
        }


class LossWeightHandler:
    def __init__(
        self,
        mask_loss_weight=1.0,
        vertex_loss_weight=1.0,
        proxy_loss_weight=0.01,
        kp_loss_weight=1.0,
        mask_loss_factor=1.0,
        vertex_loss_factor=1.0,
        proxy_loss_factor=1.0,
        kp_loss_factor=1.0,
        mask_loss_borders=(0.0, 2.5),
        vertex_loss_borders=(0.000, 10.0),
        proxy_loss_borders=(0.000, 0.025),
        kp_loss_borders=(0.0, 2.5),
        filter_vertex_with_segmentation=False,
        filter_high_proxy_errors=False,
    ):
        ###################
        self.mask_loss_weight = mask_loss_weight
        self.vertex_loss_weight = vertex_loss_weight
        self.proxy_loss_weight = proxy_loss_weight
        self.kp_loss_weight = kp_loss_weight
        self.mask_loss_factor = mask_loss_factor
        self.vertex_loss_factor = vertex_loss_factor
        self.proxy_loss_factor = proxy_loss_factor
        self.kp_loss_factor = kp_loss_factor
        self.mask_loss_borders = mask_loss_borders
        self.vertex_loss_borders = vertex_loss_borders
        self.proxy_loss_borders = proxy_loss_borders
        self.kp_loss_borders = kp_loss_borders
        self.filter_vertex_with_segmentation = filter_vertex_with_segmentation
        self.filter_high_proxy_errors = filter_high_proxy_errors

    def clamp(self, n, min_max):
        return max(min_max[0], min(n, min_max[1]))

    def update(self):
        self.mask_loss_weight = self.clamp(self.mask_loss_weight * self.mask_loss_factor, self.mask_loss_borders)
        self.vertex_loss_weight = self.clamp(
            self.vertex_loss_weight * self.vertex_loss_factor, self.vertex_loss_borders
        )
        self.proxy_loss_weight = self.clamp(self.proxy_loss_weight * self.proxy_loss_factor, self.proxy_loss_borders)
        self.kp_loss_weight = self.clamp(self.kp_loss_weight * self.kp_loss_factor, self.kp_loss_borders)

    def print(self):
        tf.print(
            "==Mask loss weight: {} , vertex loss weight: {} , proxy loss weight: {} , keypoint loss weight: {}==".format(
                self.mask_loss_weight,
                self.vertex_loss_weight,
                self.proxy_loss_weight,
                self.kp_loss_weight,
            )
        )
