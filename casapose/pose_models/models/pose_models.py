import tensorflow as tf

from .. import get_submodules_from_kwargs
from ._normalization_layers import HalfSize, PartialConvolution
from .casapose import casa_layer
from .resnet import ResNet, ResNet18, get_conv_params

# -------------------------------------------------------------------------
#  Pose Models
# -------------------------------------------------------------------------


# BMVC (C), separate decoder, no partial conv, skip connections
def CASAPoseConditional1(
    ver_dim,
    seg_dim,
    fcdim=256,
    s8dim=128,
    s4dim=64,
    s2dim=32,
    raw_dim=32,
    input_shape=None,
    input_segmentation_shape=None,
    input_tensor=None,
    weights=None,
    base_model="resnet18",
    backbone=None,
    output_lablemap=False,
    **kwargs
):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if backbone is None:
        if base_model == "resnet18":
            backbone = ResNet18(
                input_shape=input_shape, input_tensor=input_tensor, weights=weights, include_top=False, **kwargs
            )
        else:
            raise TypeError("Undefined base model type")

    [x2s, x4s, x8s, x16s, x32s] = backbone(backbone.inputs[0])

    x = casa_layer(x32s, "1", fcdim)
    x = casa_layer(layers.concatenate([x, x8s], 3), "2", s8dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x4s], 3), "3", s4dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x2s], 3), "4", s2dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, backbone.inputs[0]], 3), "5", raw_dim, leaky=True)
    x = layers.Conv2D(seg_dim, (1, 1), strides=(1, 1), name="pv_final_conv_segmentation", **get_conv_params())(x)

    # beta = tf.cast(1e6, dtype=x.dtype)  # if this value is 1e6 problems with 16 bit precision can arise in opencv
    beta = tf.cast(1e6, dtype=x.dtype)  # 10
    if input_segmentation_shape is None:
        x_mask = layers.Activation("softmax")(x * beta)
    else:
        segmentation_input = layers.Input(
            shape=input_segmentation_shape, name="data_segmentation", dtype=x.dtype
        )  # segmentation input will not work for this dasign
        x_mask = layers.Activation("softmax")(segmentation_input * beta)

    x_mask2s = HalfSize(name="segmentation_half_size", depth=seg_dim)(x_mask)
    x_mask4s = HalfSize(name="segmentation_quater_size", depth=seg_dim)(x_mask2s)
    x_mask8s = HalfSize(name="segmentation_eighth_size", depth=seg_dim)(x_mask4s)
    y = casa_layer(
        x32s,
        "6",
        fcdim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        weighted_clade=True,
    )
    y = casa_layer(
        layers.concatenate([y, x8s], 3),
        "7",
        s8dim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
    )
    y = casa_layer(
        layers.concatenate([y, x4s], 3),
        "8",
        s4dim,
        seg_mask=x_mask4s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
    )
    y = casa_layer(
        layers.concatenate([y, x2s], 3),
        "9",
        s2dim,
        seg_mask=x_mask2s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
    )
    y = casa_layer(
        layers.concatenate([y, backbone.inputs[0]], 3),
        "10",
        raw_dim,
        seg_mask=x_mask,
        num_classes=seg_dim,
        leaky=True,
        weighted_clade=True,
    )
    y = layers.Conv2D(ver_dim, (1, 1), strides=(1, 1), name="pv_final_conv_vertex", **get_conv_params())(y)

    # add softargmax to output
    if output_lablemap:
        x_range = tf.range(seg_dim, dtype=x.dtype)
        if input_segmentation_shape is None:
            x_mask_out = x_mask * x_range
        else:
            x_mask_out = layers.Activation("softmax")(x * beta) * x_range
        x_mask_out = tf.add_n(tf.split(x_mask_out, seg_dim, axis=3))
        x = layers.concatenate([x_mask_out, y], 3)
    else:
        x = layers.concatenate([x, y], 3, name="pv_final_concatenation")  # add x_mask here to output perfect mask

    model_input = [backbone.inputs[0]]
    if input_segmentation_shape is not None:
        model_input.append(segmentation_input)
    model = models.Model(model_input, [x])

    return model


# BMVC (C/GU: 0x partial conv, guided us), separate decoder, no partial conv, skip connections, guided upsampling
def CASAPoseConditional2(
    ver_dim,
    seg_dim,
    fcdim=256,
    s8dim=128,
    s4dim=64,
    s2dim=32,
    raw_dim=32,
    input_shape=None,
    input_segmentation_shape=None,
    input_tensor=None,
    weights=None,
    base_model="resnet18",
    backbone=None,
    output_lablemap=False,
    **kwargs
):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if backbone is None:
        if base_model == "resnet18":
            backbone = ResNet18(
                input_shape=input_shape, input_tensor=input_tensor, weights=weights, include_top=False, **kwargs
            )
        else:
            raise TypeError("Undefined base model type")

    [x2s, x4s, x8s, x16s, x32s] = backbone(backbone.inputs[0])

    x = casa_layer(x32s, "1", fcdim)
    x = casa_layer(layers.concatenate([x, x8s], 3), "2", s8dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x4s], 3), "3", s4dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x2s], 3), "4", s2dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, backbone.inputs[0]], 3), "5", raw_dim, leaky=True)
    x = layers.Conv2D(seg_dim, (1, 1), strides=(1, 1), name="pv_final_conv_segmentation", **get_conv_params())(x)

    beta = tf.cast(1e6, dtype=x.dtype)  # 10
    if input_segmentation_shape is None:
        x_mask = layers.Activation("softmax")(x * beta)
    else:
        segmentation_input = layers.Input(
            shape=input_segmentation_shape, name="data_segmentation", dtype=x.dtype
        )  # segmentation input will not work for this dasign
        x_mask = layers.Activation("softmax")(segmentation_input * beta)

    x_mask = tf.stop_gradient(x_mask)
    x_mask2s = HalfSize(name="segmentation_half_size", depth=seg_dim, trainable=False)(x_mask)
    x_mask4s = HalfSize(name="segmentation_quater_size", depth=seg_dim, trainable=False)(x_mask2s)
    x_mask8s = HalfSize(name="segmentation_eighth_size", depth=seg_dim, trainable=False)(x_mask4s)

    y = casa_layer(
        x32s,
        "6",
        fcdim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        weighted_clade=True,
        partial_conv=False,
    )
    y = casa_layer(
        layers.concatenate([y, x8s], 3),
        "7",
        s8dim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=False,
        seg_mask_guide=x_mask4s,
    )
    y = casa_layer(
        layers.concatenate([y, x4s], 3),
        "8",
        s4dim,
        seg_mask=x_mask4s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=False,
        seg_mask_guide=x_mask2s,
    )
    y = casa_layer(
        layers.concatenate([y, x2s], 3),
        "9",
        s2dim,
        seg_mask=x_mask2s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=False,
        seg_mask_guide=x_mask,
    )
    y = casa_layer(
        layers.concatenate([y, backbone.inputs[0]], 3),
        "10",
        raw_dim,
        seg_mask=x_mask,
        num_classes=seg_dim,
        leaky=True,
        weighted_clade=True,
        partial_conv=False,
    )
    y = layers.Conv2D(ver_dim, (1, 1), strides=(1, 1), name="pv_final_conv_vertex", **get_conv_params())(y)

    # add softargmax to output
    if output_lablemap:
        x_range = tf.range(seg_dim, dtype=x.dtype)
        if input_segmentation_shape is None:
            x_mask_out = x_mask * x_range
        else:
            x_mask_out = layers.Activation("softmax")(x * beta) * x_range
        x_mask_out = tf.add_n(tf.split(x_mask_out, seg_dim, axis=3))
        x = layers.concatenate([x_mask_out, y], 3)
    else:
        x = layers.concatenate([x, y], 3, name="pv_final_concatenation")  # add x_mask here to output perfect mask

    model_input = [backbone.inputs[0]]
    if input_segmentation_shape is not None:
        model_input.append(segmentation_input)
    model = models.Model(model_input, [x])

    return model


# BMVC  (C/GCU3: 3x partial conv, guided us), separate decoder, partial conv in first three blocks, skip connections, guided upsampling
def CASAPoseConditional3(
    ver_dim,
    seg_dim,
    fcdim=256,
    s8dim=128,
    s4dim=64,
    s2dim=32,
    raw_dim=32,
    input_shape=None,
    input_segmentation_shape=None,
    input_tensor=None,
    weights=None,
    base_model="resnet18",
    backbone=None,
    output_lablemap=False,
    **kwargs
):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if backbone is None:
        if base_model == "resnet18":
            backbone = ResNet18(
                input_shape=input_shape, input_tensor=input_tensor, weights=weights, include_top=False, **kwargs
            )
        else:
            raise TypeError("Undefined base model type")

    [x2s, x4s, x8s, x16s, x32s] = backbone(backbone.inputs[0])

    x = casa_layer(x32s, "1", fcdim)
    x = casa_layer(layers.concatenate([x, x8s], 3), "2", s8dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x4s], 3), "3", s4dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x2s], 3), "4", s2dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, backbone.inputs[0]], 3), "5", raw_dim, leaky=True)
    x = layers.Conv2D(seg_dim, (1, 1), strides=(1, 1), name="pv_final_conv_segmentation", **get_conv_params())(x)

    beta = tf.cast(1e6, dtype=x.dtype)  # 10
    if input_segmentation_shape is None:
        x_mask = layers.Activation("softmax")(x * beta)
    else:
        segmentation_input = layers.Input(
            shape=input_segmentation_shape, name="data_segmentation", dtype=x.dtype
        )  # segmentation input will not work for this dasign
        x_mask = layers.Activation("softmax")(segmentation_input * beta)

    x_mask = tf.stop_gradient(x_mask)
    x_mask2s = HalfSize(name="segmentation_half_size", depth=seg_dim, trainable=False)(x_mask)
    x_mask4s = HalfSize(name="segmentation_quater_size", depth=seg_dim, trainable=False)(x_mask2s)
    x_mask8s = HalfSize(name="segmentation_eighth_size", depth=seg_dim, trainable=False)(x_mask4s)

    y = casa_layer(
        x32s,
        "6",
        fcdim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        weighted_clade=True,
        partial_conv=True,
    )
    y = casa_layer(
        layers.concatenate([y, x8s], 3),
        "7",
        s8dim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask4s,
    )
    y = casa_layer(
        layers.concatenate([y, x4s], 3),
        "8",
        s4dim,
        seg_mask=x_mask4s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask2s,
    )
    y = casa_layer(
        layers.concatenate([y, x2s], 3),
        "9",
        s2dim,
        seg_mask=x_mask2s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=False,
        seg_mask_guide=x_mask,
    )
    y = casa_layer(
        layers.concatenate([y, backbone.inputs[0]], 3),
        "10",
        raw_dim,
        seg_mask=x_mask,
        num_classes=seg_dim,
        leaky=True,
        weighted_clade=True,
        partial_conv=False,
    )
    y = layers.Conv2D(ver_dim, (1, 1), strides=(1, 1), name="pv_final_conv_vertex", **get_conv_params())(y)

    # add softargmax to output
    if output_lablemap:
        x_range = tf.range(seg_dim, dtype=x.dtype)
        if input_segmentation_shape is None:
            x_mask_out = x_mask * x_range
        else:
            x_mask_out = layers.Activation("softmax")(x * beta) * x_range
        x_mask_out = tf.add_n(tf.split(x_mask_out, seg_dim, axis=3))
        x = layers.concatenate([x_mask_out, y], 3)
    else:
        x = layers.concatenate([x, y], 3, name="pv_final_concatenation")  # add x_mask here to output perfect mask

    model_input = [backbone.inputs[0]]
    if input_segmentation_shape is not None:
        model_input.append(segmentation_input)
    model = models.Model(model_input, [x])

    return model


# BMVC (C/GCU3: 4x partial conv, guided us), separate decoder, partial conv in first four blocks, skip connections, guided upsampling
def CASAPoseConditional4(
    ver_dim,
    seg_dim,
    fcdim=256,
    s8dim=128,
    s4dim=64,
    s2dim=32,
    raw_dim=32,
    input_shape=None,
    input_segmentation_shape=None,
    input_tensor=None,
    weights=None,
    base_model="resnet18",
    backbone=None,
    output_lablemap=False,
    **kwargs
):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if backbone is None:
        if base_model == "resnet18":
            backbone = ResNet18(
                input_shape=input_shape, input_tensor=input_tensor, weights=weights, include_top=False, **kwargs
            )
        else:
            raise TypeError("Undefined base model type")

    [x2s, x4s, x8s, x16s, x32s] = backbone(backbone.inputs[0])

    x = casa_layer(x32s, "1", fcdim)
    x = casa_layer(layers.concatenate([x, x8s], 3), "2", s8dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x4s], 3), "3", s4dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x2s], 3), "4", s2dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, backbone.inputs[0]], 3), "5", raw_dim, leaky=True)
    x = layers.Conv2D(seg_dim, (1, 1), strides=(1, 1), name="pv_final_conv_segmentation", **get_conv_params())(x)

    beta = tf.cast(1e6, dtype=x.dtype)  # 10
    if input_segmentation_shape is None:
        x_mask = layers.Activation("softmax")(x * beta)
    else:
        segmentation_input = layers.Input(
            shape=input_segmentation_shape, name="data_segmentation", dtype=x.dtype
        )  # segmentation input will not work for this dasign
        x_mask = layers.Activation("softmax")(segmentation_input * beta)

    x_mask = tf.stop_gradient(x_mask)
    x_mask2s = HalfSize(name="segmentation_half_size", depth=seg_dim, trainable=False)(x_mask)
    x_mask4s = HalfSize(name="segmentation_quater_size", depth=seg_dim, trainable=False)(x_mask2s)
    x_mask8s = HalfSize(name="segmentation_eighth_size", depth=seg_dim, trainable=False)(x_mask4s)

    y = casa_layer(
        x32s,
        "6",
        fcdim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        weighted_clade=True,
        partial_conv=True,
    )
    y = casa_layer(
        layers.concatenate([y, x8s], 3),
        "7",
        s8dim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask4s,
    )
    y = casa_layer(
        layers.concatenate([y, x4s], 3),
        "8",
        s4dim,
        seg_mask=x_mask4s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask2s,
    )
    y = casa_layer(
        layers.concatenate([y, x2s], 3),
        "9",
        s2dim,
        seg_mask=x_mask2s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask,
    )
    y = casa_layer(
        layers.concatenate([y, backbone.inputs[0]], 3),
        "10",
        raw_dim,
        seg_mask=x_mask,
        num_classes=seg_dim,
        leaky=True,
        weighted_clade=True,
        partial_conv=False,
    )
    y = layers.Conv2D(ver_dim, (1, 1), strides=(1, 1), name="pv_final_conv_vertex", **get_conv_params())(y)

    # add softargmax to output
    if output_lablemap:
        x_range = tf.range(seg_dim, dtype=x.dtype)
        if input_segmentation_shape is None:
            x_mask_out = x_mask * x_range
        else:
            x_mask_out = layers.Activation("softmax")(x * beta) * x_range
        x_mask_out = tf.add_n(tf.split(x_mask_out, seg_dim, axis=3))
        x = layers.concatenate([x_mask_out, y], 3)
    else:
        x = layers.concatenate([x, y], 3, name="pv_final_concatenation")  # add x_mask here to output perfect mask

    model_input = [backbone.inputs[0]]
    if input_segmentation_shape is not None:
        model_input.append(segmentation_input)
    model = models.Model(model_input, [x])

    return model


# BMVC  (C/GCU5: 5x partial conv, guided us), separate decoder, partial conv in five blocks layers, skip connections, guided upsampling
def CASAPoseConditional5(
    ver_dim,
    seg_dim,
    fcdim=256,
    s8dim=128,
    s4dim=64,
    s2dim=32,
    raw_dim=32,
    input_shape=None,
    input_segmentation_shape=None,
    input_tensor=None,
    weights=None,
    base_model="resnet18",
    backbone=None,
    output_lablemap=False,
    **kwargs
):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if backbone is None:
        if base_model == "resnet18":
            backbone = ResNet18(
                input_shape=input_shape, input_tensor=input_tensor, weights=weights, include_top=False, **kwargs
            )
        else:
            raise TypeError("Undefined base model type")

    [x2s, x4s, x8s, x16s, x32s] = backbone(backbone.inputs[0])

    x = casa_layer(x32s, "1", fcdim)
    x = casa_layer(layers.concatenate([x, x8s], 3), "2", s8dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x4s], 3), "3", s4dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x2s], 3), "4", s2dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, backbone.inputs[0]], 3), "5", raw_dim, leaky=True)
    x = layers.Conv2D(seg_dim, (1, 1), strides=(1, 1), name="pv_final_conv_segmentation", **get_conv_params())(x)

    beta = tf.cast(1e6, dtype=x.dtype)  # 10
    if input_segmentation_shape is None:
        x_mask = layers.Activation("softmax")(x * beta)
    else:
        segmentation_input = layers.Input(
            shape=input_segmentation_shape, name="data_segmentation", dtype=x.dtype
        )  # segmentation input will not work for this dasign
        x_mask = layers.Activation("softmax")(segmentation_input * beta)

    x_mask = tf.stop_gradient(x_mask)
    x_mask2s = HalfSize(name="segmentation_half_size", depth=seg_dim, trainable=False)(x_mask)
    x_mask4s = HalfSize(name="segmentation_quater_size", depth=seg_dim, trainable=False)(x_mask2s)
    x_mask8s = HalfSize(name="segmentation_eighth_size", depth=seg_dim, trainable=False)(x_mask4s)

    y = casa_layer(
        x32s,
        "6",
        fcdim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        weighted_clade=True,
        partial_conv=True,
    )
    y = casa_layer(
        layers.concatenate([y, x8s], 3),
        "7",
        s8dim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask4s,
    )
    y = casa_layer(
        layers.concatenate([y, x4s], 3),
        "8",
        s4dim,
        seg_mask=x_mask4s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask2s,
    )
    y = casa_layer(
        layers.concatenate([y, x2s], 3),
        "9",
        s2dim,
        seg_mask=x_mask2s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask,
    )
    y = casa_layer(
        layers.concatenate([y, backbone.inputs[0]], 3),
        "10",
        raw_dim,
        seg_mask=x_mask,
        num_classes=seg_dim,
        leaky=True,
        weighted_clade=True,
        partial_conv=True,
    )
    y = layers.Conv2D(ver_dim, (1, 1), strides=(1, 1), name="pv_final_conv_vertex", **get_conv_params())(y)

    # add softargmax to output
    if output_lablemap:
        x_range = tf.range(seg_dim, dtype=x.dtype)
        if input_segmentation_shape is None:
            x_mask_out = x_mask * x_range
        else:
            x_mask_out = layers.Activation("softmax")(x * beta) * x_range
        x_mask_out = tf.add_n(tf.split(x_mask_out, seg_dim, axis=3))
        x = layers.concatenate([x_mask_out, y], 3)
    else:
        x = layers.concatenate([x, y], 3, name="pv_final_concatenation")  # add x_mask here to output perfect mask

    model_input = [backbone.inputs[0]]
    if input_segmentation_shape is not None:
        model_input.append(segmentation_input)
    model = models.Model(model_input, [x])

    return model


"""
Our implementation of PVNet is based on code for paper "PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation Sida"
by Sida Peng et al., ZJU-SenseTime Joint Lab of 3D Vision (https://github.com/zju3dv/pvnet).
"""


# BMVC (Base)
def PVNet(
    ver_dim,
    seg_dim,
    fcdim=256,
    s8dim=128,
    s4dim=64,
    s2dim=32,
    raw_dim=32,
    input_shape=None,
    input_tensor=None,
    weights=None,
    base_model="resnet18",
    backbone=None,
    output_lablemap=False,
    **kwargs
):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if backbone is None:
        if base_model == "resnet18":
            backbone = ResNet18(
                input_shape=input_shape, input_tensor=input_tensor, weights=weights, include_top=False, **kwargs
            )
        else:
            raise TypeError("Undefined base model type")

    [x2s, x4s, x8s, x16s, x32s] = backbone(backbone.inputs[0])

    x = casa_layer(x32s, "1", fcdim)
    x = casa_layer(layers.concatenate([x, x8s], 3), "2", s8dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x4s], 3), "3", s4dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x2s], 3), "4", s2dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, backbone.inputs[0]], 3), "5", raw_dim, leaky=True)

    x = layers.Conv2D(seg_dim + ver_dim, (1, 1), strides=(1, 1), name="pv_final_conv", **get_conv_params())(x)

    # add softargmax to output

    if output_lablemap:
        # beta = 1e5
        y = x[:, :, :, seg_dim : seg_dim + ver_dim]
        x_argmax = x[:, :, :, 0:seg_dim]
        x_range = tf.range(seg_dim, dtype=x.dtype)
        beta = tf.cast(1e6, dtype=x.dtype)
        x_argmax = layers.Activation("softmax")(x_argmax * beta) * x_range
        x_argmax = tf.add_n(tf.split(x_argmax, seg_dim, axis=3))
        # x_argmax = tf.squeeze(tf.keras.backend.one_hot(tf.keras.backend.cast(x_argmax, 'uint8'), num_classes=seg_dim), axis=3)
        # x_argmax = tf.keras.backend.mean(x_argmax, axis=3, keepdims=True ) * x_range # reduce_sum (compatibility with opencv)
        x = layers.concatenate([x_argmax, y], 3)

    model = models.Model(backbone.inputs, [x])
    return model


# uses the same convolution weights in both decoders # casapose_cond_weighted_combined
def CASAPoseConditional6(
    ver_dim,
    seg_dim,
    fcdim=256,
    s8dim=128,
    s4dim=64,
    s2dim=32,
    raw_dim=32,
    input_shape=None,
    input_segmentation_shape=None,
    input_tensor=None,
    weights=None,
    base_model="resnet18",
    backbone=None,
    output_lablemap=False,
    **kwargs
):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if backbone is None:
        if base_model == "resnet18":
            backbone = ResNet18(
                input_shape=input_shape, input_tensor=input_tensor, weights=weights, include_top=False, **kwargs
            )
        else:
            raise TypeError("Undefined base model type")

    [x2s, x4s, x8s, x16s, x32s] = backbone(backbone.inputs[0])

    c1 = PartialConvolution(name="pv_block_1_6_conv2d", dim=fcdim, num_classes=seg_dim)
    c2 = PartialConvolution(name="pv_block_2_7_conv2d", dim=s8dim, num_classes=seg_dim)
    c3 = PartialConvolution(name="pv_block_3_8_conv2d", dim=s4dim, num_classes=seg_dim)
    c4 = PartialConvolution(name="pv_block_4_9_conv2d", dim=s2dim, num_classes=seg_dim)
    c5 = PartialConvolution(name="pv_block_5_10_conv2d", dim=raw_dim, num_classes=seg_dim)

    y = c1([x32s])
    x = casa_layer(y, "1", fcdim, skip_conv=True)
    x = c2([layers.concatenate([x, x8s], 3)])
    x = casa_layer(x, "2", s8dim, leaky=True, upsampling=True, skip_conv=True)
    x = c3([layers.concatenate([x, x4s], 3)])
    x = casa_layer(x, "3", s4dim, leaky=True, upsampling=True, skip_conv=True)
    x = c4([layers.concatenate([x, x2s], 3)])
    x = casa_layer(x, "4", s2dim, leaky=True, upsampling=True, skip_conv=True)
    x = c5([layers.concatenate([x, backbone.inputs[0]], 3)])
    x = casa_layer(x, "5", raw_dim, leaky=True, skip_conv=True)
    x = layers.Conv2D(seg_dim, (1, 1), strides=(1, 1), name="pv_final_conv_segmentation", **get_conv_params())(x)

    beta = tf.cast(1e6, dtype=x.dtype)  # 10
    if input_segmentation_shape is None:
        x_mask = layers.Activation("softmax")(x * beta)
    else:
        segmentation_input = layers.Input(
            shape=input_segmentation_shape, name="data_segmentation", dtype=x.dtype
        )  # segmentation input will not work for this dasign
        x_mask = layers.Activation("softmax")(segmentation_input * beta)

    x_mask = tf.stop_gradient(x_mask)
    x_mask2s = HalfSize(name="segmentation_half_size", depth=seg_dim, trainable=False)(x_mask)
    x_mask4s = HalfSize(name="segmentation_quater_size", depth=seg_dim, trainable=False)(x_mask2s)
    x_mask8s = HalfSize(name="segmentation_eighth_size", depth=seg_dim, trainable=False)(x_mask4s)

    # y = c1([x32s, x_mask8s])
    y = casa_layer(
        y,
        "6",
        fcdim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        weighted_clade=True,
        skip_conv=True,
    )
    y = c2([layers.concatenate([y, x8s], 3), x_mask8s])
    y = casa_layer(
        y,
        "7",
        s8dim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        skip_conv=True,
        seg_mask_guide=x_mask4s,
    )
    y = c3([layers.concatenate([y, x4s], 3), x_mask4s])
    y = casa_layer(
        y,
        "8",
        s4dim,
        seg_mask=x_mask4s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        skip_conv=True,
        seg_mask_guide=x_mask2s,
    )
    y = c4([layers.concatenate([y, x2s], 3), x_mask2s])
    y = casa_layer(
        y,
        "9",
        s2dim,
        seg_mask=x_mask2s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        skip_conv=True,
        seg_mask_guide=x_mask,
    )
    y = c5([layers.concatenate([y, backbone.inputs[0]], 3), x_mask])
    y = casa_layer(
        y,
        "10",
        raw_dim,
        seg_mask=x_mask,
        num_classes=seg_dim,
        leaky=True,
        weighted_clade=True,
        skip_conv=True,
    )
    y = layers.Conv2D(ver_dim, (1, 1), strides=(1, 1), name="pv_final_conv_vertex", **get_conv_params())(y)

    # add softargmax to output
    if output_lablemap:
        x_range = tf.range(seg_dim, dtype=x.dtype)
        if input_segmentation_shape is None:
            x_mask_out = x_mask * x_range
        else:
            x_mask_out = layers.Activation("softmax")(x * beta) * x_range
        x_mask_out = tf.add_n(tf.split(x_mask_out, seg_dim, axis=3))
        x = layers.concatenate([x_mask_out, y], 3)
    else:
        x = layers.concatenate([x, y], 3, name="pv_final_concatenation")  # add x_mask here to output perfect mask

    model_input = [backbone.inputs[0]]
    if input_segmentation_shape is not None:
        model_input.append(segmentation_input)
    model = models.Model(model_input, [x])

    return model


# both decoders share initial convolution, 4x partial conv
def CASAPoseConditional7(
    ver_dim,
    seg_dim,
    fcdim=256,
    s8dim=128,
    s4dim=64,
    s2dim=32,
    raw_dim=32,
    input_shape=None,
    input_segmentation_shape=None,
    input_tensor=None,
    weights=None,
    base_model="resnet18",
    backbone=None,
    output_lablemap=False,
    **kwargs
):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if backbone is None:
        if base_model == "resnet18":
            backbone = ResNet18(
                input_shape=input_shape, input_tensor=input_tensor, weights=weights, include_top=False, **kwargs
            )
        else:
            raise TypeError("Undefined base model type")

    [x2s, x4s, x8s, x16s, x32s] = backbone(backbone.inputs[0])

    c1 = PartialConvolution(name="pv_block_1_6_conv2d", dim=fcdim, num_classes=seg_dim)

    y = c1([x32s])
    x = casa_layer(y, "1", fcdim, skip_conv=True)
    x = casa_layer(layers.concatenate([x, x8s], 3), "2", s8dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x4s], 3), "3", s4dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x2s], 3), "4", s2dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, backbone.inputs[0]], 3), "5", raw_dim, leaky=True)
    x = layers.Conv2D(seg_dim, (1, 1), strides=(1, 1), name="pv_final_conv_segmentation", **get_conv_params())(x)

    beta = tf.cast(1e6, dtype=x.dtype)  # 10
    if input_segmentation_shape is None:
        x_mask = layers.Activation("softmax")(x * beta)
    else:
        segmentation_input = layers.Input(
            shape=input_segmentation_shape, name="data_segmentation", dtype=x.dtype
        )  # segmentation input will not work for this dasign
        x_mask = layers.Activation("softmax")(segmentation_input * beta)

    x_mask = tf.stop_gradient(x_mask)
    x_mask2s = HalfSize(name="segmentation_half_size", depth=seg_dim, trainable=False)(x_mask)
    x_mask4s = HalfSize(name="segmentation_quater_size", depth=seg_dim, trainable=False)(x_mask2s)
    x_mask8s = HalfSize(name="segmentation_eighth_size", depth=seg_dim, trainable=False)(x_mask4s)

    y = casa_layer(
        y,
        "6",
        fcdim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        weighted_clade=True,
        skip_conv=True,
    )
    y = casa_layer(
        layers.concatenate([y, x8s], 3),
        "7",
        s8dim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask4s,
    )
    y = casa_layer(
        layers.concatenate([y, x4s], 3),
        "8",
        s4dim,
        seg_mask=x_mask4s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask2s,
    )
    y = casa_layer(
        layers.concatenate([y, x2s], 3),
        "9",
        s2dim,
        seg_mask=x_mask2s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask,
    )
    y = casa_layer(
        layers.concatenate([y, backbone.inputs[0]], 3),
        "10",
        raw_dim,
        seg_mask=x_mask,
        num_classes=seg_dim,
        leaky=True,
        weighted_clade=True,
        partial_conv=True,
    )
    y = layers.Conv2D(ver_dim, (1, 1), strides=(1, 1), name="pv_final_conv_vertex", **get_conv_params())(y)

    # add softargmax to output
    if output_lablemap:
        x_range = tf.range(seg_dim, dtype=x.dtype)
        if input_segmentation_shape is None:
            x_mask_out = x_mask * x_range
        else:
            x_mask_out = layers.Activation("softmax")(x * beta) * x_range
        x_mask_out = tf.add_n(tf.split(x_mask_out, seg_dim, axis=3))
        x = layers.concatenate([x_mask_out, y], 3)
    else:
        x = layers.concatenate([x, y], 3, name="pv_final_concatenation")  # add x_mask here to output perfect mask

    model_input = [backbone.inputs[0]]
    if input_segmentation_shape is not None:
        model_input.append(segmentation_input)
    model = models.Model(model_input, [x])

    return model


# both decoders share initial convolution, 5x partial conv
def CASAPoseConditional8(
    ver_dim,
    seg_dim,
    fcdim=256,
    s8dim=128,
    s4dim=64,
    s2dim=32,
    raw_dim=32,
    input_shape=None,
    input_segmentation_shape=None,
    input_tensor=None,
    weights=None,
    base_model="resnet18",
    backbone=None,
    output_lablemap=False,
    **kwargs
):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if backbone is None:
        if base_model == "resnet18":
            backbone = ResNet18(
                input_shape=input_shape, input_tensor=input_tensor, weights=weights, include_top=False, **kwargs
            )
        else:
            raise TypeError("Undefined base model type")

    [x2s, x4s, x8s, x16s, x32s] = backbone(backbone.inputs[0])

    c1 = PartialConvolution(name="pv_block_1_6_conv2d", dim=fcdim, num_classes=seg_dim)

    y = c1([x32s])
    x = casa_layer(y, "1", fcdim, skip_conv=True)
    x = casa_layer(layers.concatenate([x, x8s], 3), "2", s8dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x4s], 3), "3", s4dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x2s], 3), "4", s2dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, backbone.inputs[0]], 3), "5", raw_dim, leaky=True)
    x = layers.Conv2D(seg_dim, (1, 1), strides=(1, 1), name="pv_final_conv_segmentation", **get_conv_params())(x)

    beta = tf.cast(1e6, dtype=x.dtype)  # 10
    if input_segmentation_shape is None:
        x_mask = layers.Activation("softmax")(x * beta)
    else:
        segmentation_input = layers.Input(
            shape=input_segmentation_shape, name="data_segmentation", dtype=x.dtype
        )  # segmentation input will not work for this dasign
        x_mask = layers.Activation("softmax")(segmentation_input * beta)

    x_mask = tf.stop_gradient(x_mask)
    x_mask2s = HalfSize(name="segmentation_half_size", depth=seg_dim, trainable=False)(x_mask)
    x_mask4s = HalfSize(name="segmentation_quater_size", depth=seg_dim, trainable=False)(x_mask2s)
    x_mask8s = HalfSize(name="segmentation_eighth_size", depth=seg_dim, trainable=False)(x_mask4s)

    y = casa_layer(
        y,
        "6",
        fcdim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        weighted_clade=True,
        skip_conv=True,
    )
    y = casa_layer(
        y,
        "7",
        s8dim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask4s,
    )
    y = casa_layer(
        y,
        "8",
        s4dim,
        seg_mask=x_mask4s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask2s,
    )
    y = casa_layer(
        y,
        "9",
        s2dim,
        seg_mask=x_mask2s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask,
    )
    y = casa_layer(
        y,
        "10",
        raw_dim,
        seg_mask=x_mask,
        num_classes=seg_dim,
        leaky=True,
        weighted_clade=True,
        partial_conv=True,
    )
    y = layers.Conv2D(ver_dim, (1, 1), strides=(1, 1), name="pv_final_conv_vertex", **get_conv_params())(y)

    # add softargmax to output
    if output_lablemap:
        x_range = tf.range(seg_dim, dtype=x.dtype)
        if input_segmentation_shape is None:
            x_mask_out = x_mask * x_range
        else:
            x_mask_out = layers.Activation("softmax")(x * beta) * x_range
        x_mask_out = tf.add_n(tf.split(x_mask_out, seg_dim, axis=3))
        x = layers.concatenate([x_mask_out, y], 3)
    else:
        x = layers.concatenate([x, y], 3, name="pv_final_concatenation")  # add x_mask here to output perfect mask

    model_input = [backbone.inputs[0]]
    if input_segmentation_shape is not None:
        model_input.append(segmentation_input)
    model = models.Model(model_input, [x])

    return model


# use guided bilateral upsampling, (4x partial conv, guided us)
def CASAPoseConditional9(
    ver_dim,
    seg_dim,
    fcdim=256,
    s8dim=128,
    s4dim=64,
    s2dim=32,
    raw_dim=32,
    input_shape=None,
    input_segmentation_shape=None,
    input_tensor=None,
    weights=None,
    base_model="resnet18",
    backbone=None,
    output_lablemap=False,
    **kwargs
):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if backbone is None:
        if base_model == "resnet18":
            backbone = ResNet18(
                input_shape=input_shape, input_tensor=input_tensor, weights=weights, include_top=False, **kwargs
            )
        else:
            raise TypeError("Undefined base model type")

    [x2s, x4s, x8s, x16s, x32s] = backbone(backbone.inputs[0])

    x = casa_layer(x32s, "1", fcdim)
    x = casa_layer(layers.concatenate([x, x8s], 3), "2", s8dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x4s], 3), "3", s4dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x2s], 3), "4", s2dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, backbone.inputs[0]], 3), "5", raw_dim, leaky=True)
    x = layers.Conv2D(seg_dim, (1, 1), strides=(1, 1), name="pv_final_conv_segmentation", **get_conv_params())(x)

    beta = tf.cast(1e6, dtype=x.dtype)  # 10
    if input_segmentation_shape is None:
        x_mask = layers.Activation("softmax")(x * beta)
    else:
        segmentation_input = layers.Input(
            shape=input_segmentation_shape, name="data_segmentation", dtype=x.dtype
        )  # segmentation input will not work for this dasign
        x_mask = layers.Activation("softmax")(segmentation_input * beta)

    x_mask = tf.stop_gradient(x_mask)
    x_mask2s = HalfSize(name="segmentation_half_size", depth=seg_dim, trainable=False)(x_mask)
    x_mask4s = HalfSize(name="segmentation_quater_size", depth=seg_dim, trainable=False)(x_mask2s)
    x_mask8s = HalfSize(name="segmentation_eighth_size", depth=seg_dim, trainable=False)(x_mask4s)

    y = casa_layer(
        x32s,
        "6",
        fcdim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        weighted_clade=True,
        partial_conv=True,
    )
    y = casa_layer(
        layers.concatenate([y, x8s], 3),
        "7",
        s8dim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        upsampling_bilinear=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask4s,
    )
    y = casa_layer(
        layers.concatenate([y, x4s], 3),
        "8",
        s4dim,
        seg_mask=x_mask4s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        upsampling_bilinear=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask2s,
    )
    y = casa_layer(
        layers.concatenate([y, x2s], 3),
        "9",
        s2dim,
        seg_mask=x_mask2s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        upsampling_bilinear=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask,
    )
    y = casa_layer(
        layers.concatenate([y, backbone.inputs[0]], 3),
        "10",
        raw_dim,
        seg_mask=x_mask,
        num_classes=seg_dim,
        leaky=True,
        weighted_clade=True,
        partial_conv=False,
    )
    y = layers.Conv2D(ver_dim, (1, 1), strides=(1, 1), name="pv_final_conv_vertex", **get_conv_params())(y)

    # add softargmax to output
    if output_lablemap:
        x_range = tf.range(seg_dim, dtype=x.dtype)
        if input_segmentation_shape is None:
            x_mask_out = x_mask * x_range
        else:
            x_mask_out = layers.Activation("softmax")(x * beta) * x_range
        x_mask_out = tf.add_n(tf.split(x_mask_out, seg_dim, axis=3))
        x = layers.concatenate([x_mask_out, y], 3)
    else:
        x = layers.concatenate([x, y], 3, name="pv_final_concatenation")  # add x_mask here to output perfect mask

    model_input = [backbone.inputs[0]]
    if input_segmentation_shape is not None:
        model_input.append(segmentation_input)
    model = models.Model(model_input, [x])

    return model


# shared decoder for first two blocks decoder (4x partial conv, guided us)
def CASAPoseConditional10(
    ver_dim,
    seg_dim,
    fcdim=256,
    s8dim=128,
    s4dim=64,
    s2dim=32,
    raw_dim=32,
    input_shape=None,
    input_segmentation_shape=None,
    input_tensor=None,
    weights=None,
    base_model="resnet18",
    backbone=None,
    output_lablemap=False,
    **kwargs
):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if backbone is None:
        if base_model == "resnet18":
            backbone = ResNet18(
                input_shape=input_shape, input_tensor=input_tensor, weights=weights, include_top=False, **kwargs
            )
        else:
            raise TypeError("Undefined base model type")

    [x2s, x4s, x8s, x16s, x32s] = backbone(backbone.inputs[0])

    c1 = PartialConvolution(name="pv_block_1_6_conv2d", dim=fcdim, num_classes=seg_dim)
    c2 = PartialConvolution(name="pv_block_2_7_conv2d", dim=s8dim, num_classes=seg_dim)

    x = casa_layer(c1([x32s]), "1", fcdim, skip_conv=True)
    x = casa_layer(
        c2([layers.concatenate([x, x8s], 3)]),
        "2",
        s8dim,
        leaky=True,
        upsampling=True,
        skip_conv=True,
    )
    x = casa_layer(layers.concatenate([x, x4s], 3), "3", s4dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, x2s], 3), "4", s2dim, leaky=True, upsampling=True)
    x = casa_layer(layers.concatenate([x, backbone.inputs[0]], 3), "5", raw_dim, leaky=True)
    x = layers.Conv2D(seg_dim, (1, 1), strides=(1, 1), name="pv_final_conv_segmentation", **get_conv_params())(x)

    beta = tf.cast(1e6, dtype=x.dtype)  # 10
    if input_segmentation_shape is None:
        x_mask = layers.Activation("softmax")(x * beta)
    else:
        segmentation_input = layers.Input(
            shape=input_segmentation_shape, name="data_segmentation", dtype=x.dtype
        )  # segmentation input will not work for this dasign
        x_mask = layers.Activation("softmax")(segmentation_input * beta)

    x_mask = tf.stop_gradient(x_mask)
    x_mask2s = HalfSize(name="segmentation_half_size", depth=seg_dim, trainable=False)(x_mask)
    x_mask4s = HalfSize(name="segmentation_quater_size", depth=seg_dim, trainable=False)(x_mask2s)
    x_mask8s = HalfSize(name="segmentation_eighth_size", depth=seg_dim, trainable=False)(x_mask4s)

    y = casa_layer(
        c1([x32s, x_mask8s]),
        "6",
        fcdim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        weighted_clade=True,
        skip_conv=True,
    )
    y = casa_layer(
        c2([layers.concatenate([y, x8s], 3), x_mask8s]),
        "7",
        s8dim,
        seg_mask=x_mask8s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        skip_conv=True,
        seg_mask_guide=x_mask4s,
    )
    y = casa_layer(
        layers.concatenate([y, x4s], 3),
        "8",
        s4dim,
        seg_mask=x_mask4s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask2s,
    )
    y = casa_layer(
        layers.concatenate([y, x2s], 3),
        "9",
        s2dim,
        seg_mask=x_mask2s,
        num_classes=seg_dim,
        leaky=True,
        upsampling=True,
        weighted_clade=True,
        partial_conv=True,
        seg_mask_guide=x_mask,
    )
    y = casa_layer(
        layers.concatenate([y, backbone.inputs[0]], 3),
        "10",
        raw_dim,
        seg_mask=x_mask,
        num_classes=seg_dim,
        leaky=True,
        weighted_clade=True,
        partial_conv=False,
    )
    y = layers.Conv2D(ver_dim, (1, 1), strides=(1, 1), name="pv_final_conv_vertex", **get_conv_params())(y)

    # add softargmax to output
    if output_lablemap:
        x_range = tf.range(seg_dim, dtype=x.dtype)
        if input_segmentation_shape is None:
            x_mask_out = x_mask * x_range
        else:
            x_mask_out = layers.Activation("softmax")(x * beta) * x_range
        x_mask_out = tf.add_n(tf.split(x_mask_out, seg_dim, axis=3))
        x = layers.concatenate([x_mask_out, y], 3)
    else:
        x = layers.concatenate([x, y], 3, name="pv_final_concatenation")  # add x_mask here to output perfect mask

    model_input = [backbone.inputs[0]]
    if input_segmentation_shape is not None:
        model_input.append(segmentation_input)
    model = models.Model(model_input, [x])

    return model


setattr(ResNet18, "__doc__", ResNet.__doc__)
# setattr(ResNet34, "__doc__", ResNet.__doc__)
# setattr(ResNet50, '__doc__', ResNet.__doc__)
# setattr(ResNet101, '__doc__', ResNet.__doc__)
# setattr(ResNet152, '__doc__', ResNet.__doc__)
setattr(PVNet, "__doc__", ResNet.__doc__)
setattr(CASAPoseConditional1, "__doc__", ResNet.__doc__)
setattr(CASAPoseConditional2, "__doc__", ResNet.__doc__)
setattr(CASAPoseConditional3, "__doc__", ResNet.__doc__)
setattr(CASAPoseConditional4, "__doc__", ResNet.__doc__)
setattr(CASAPoseConditional5, "__doc__", ResNet.__doc__)
setattr(CASAPoseConditional6, "__doc__", ResNet.__doc__)
setattr(CASAPoseConditional7, "__doc__", ResNet.__doc__)
setattr(CASAPoseConditional8, "__doc__", ResNet.__doc__)
setattr(CASAPoseConditional9, "__doc__", ResNet.__doc__)
setattr(CASAPoseConditional10, "__doc__", ResNet.__doc__)
