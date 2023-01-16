import collections

import tensorflow as tf

from .. import get_submodules_from_kwargs
from ._normalization_layers import (
    ClassAdaptiveNormalization,
    ClassAdaptiveWeightedNormalization,
    ClassAdaptiveWeightedNormalizationWithInput,
    ClassAdaptiveWeightedNormalizationWithInputAndLearnedParameters,
    GuidedBilinearUpsampling,
    GuidedUpsampling,
    HalfSize,
    PartialConvolution,
)
from .resnet import ResNet, ResNet18, get_bn_params, get_conv_params

DecoderParams = collections.namedtuple(
    "DecoderParams",
    ["weighted_clade", "partial_conv", "guided_upsampling", "bilinear_upsampling", "reuse_conv"],
)

# -------------------------------------------------------------------------
#   CASAPose
# -------------------------------------------------------------------------

CASAPOSE_PARAMS = {
    "clade": [
        DecoderParams(True, True, False, False, False),  # layer "5"
        DecoderParams(True, True, True, False, False),  # layer "6"
        DecoderParams(True, True, True, False, False),  # layer "7"
        DecoderParams(True, True, True, False, False),  # layer "8"
        DecoderParams(True, True, False, False, False),  # layer "9"
    ],
}


def CASAPoseConditional(*args, **kwargs):
    return CASAPose(CASAPOSE_PARAMS["clade"], *args, **kwargs, learn_upsampling=False)


def casa_layer(
    x,
    idx,
    dim,
    seg_mask=None,
    num_classes=1,
    leaky=False,
    upsampling=False,
    upsampling_bilinear=False,
    weighted_clade=False,
    skip_conv=False,
    denormalization_weights=None,
    weighted_clade_plus_parameter=False,
    partial_conv=False,
    seg_mask_guide=None,
):
    conv_params = get_conv_params()
    bn_params = get_bn_params()

    if skip_conv is False:
        if partial_conv and seg_mask is not None:
            x = PartialConvolution(
                name="pv_block_" + idx + "_prepare_conv2d",
                dim=dim,
                num_classes=num_classes,
                conv_name="pv_block_" + idx + "_conv2d",
            )([x, seg_mask])
        else:
            # x = PartialConvolution(name="pv_block_"+idx+"_prepare_conv2d", dim=dim, num_classes = num_classes, conv_name="pv_block_"+idx+"_conv2d")([x])
            x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
            x = tf.keras.layers.Conv2D(dim, (3, 3), strides=(1, 1), name="pv_block_" + idx + "_conv2d", **conv_params)(
                x
            )

    if seg_mask is None:
        x = tf.keras.layers.experimental.SyncBatchNormalization(name="pv_block_" + idx + "_bn", **bn_params)(x)
    elif weighted_clade is True:
        if denormalization_weights is None:
            x = ClassAdaptiveWeightedNormalization(name="pv_block_" + idx + "_clade", num_classes=num_classes)(
                [x, seg_mask]
            )
        else:
            if weighted_clade_plus_parameter is True:
                x = ClassAdaptiveWeightedNormalizationWithInputAndLearnedParameters(
                    name="pv_block_" + idx + "_clade_mesh_input",
                    num_classes=num_classes,
                )([x, seg_mask, denormalization_weights[0], denormalization_weights[1]])
            else:
                x = ClassAdaptiveWeightedNormalizationWithInput(
                    name="pv_block_" + idx + "_clade_mesh_input",
                    num_classes=num_classes,
                )([x, seg_mask, denormalization_weights[0], denormalization_weights[1]])

    else:
        x = ClassAdaptiveNormalization(name="pv_block_" + idx + "_clade", num_classes=num_classes)([x, seg_mask])

    if leaky is True:
        # replace leaky relu layer with normal relu combination
        x = tf.keras.layers.Subtract()(
            [
                tf.keras.layers.Activation("relu", name="pv_block_" + idx + "_relu1")(x),
                tf.keras.layers.Activation("relu", name="pv_block_" + idx + "_relu2")(-0.1 * x),
            ]
        )
    else:
        x = tf.keras.layers.Activation("relu", name="pv_block_" + idx + "_relu")(x)

    if upsampling is True:
        if weighted_clade is True:
            if seg_mask_guide is not None:
                # x = tf.compat.v1.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last", interpolation="nearest", name="pv_block_"+idx+"_upsampling")(x)
                if upsampling_bilinear is True:
                    x = GuidedBilinearUpsampling(name="pv_block_" + idx + "_guided_upsamling")(
                        [x, seg_mask, seg_mask_guide]
                    )
                else:
                    x = GuidedUpsampling(name="pv_block_" + idx + "_guided_upsamling")([x, seg_mask, seg_mask_guide])
            else:
                if upsampling_bilinear is True:
                    x = tf.compat.v1.keras.layers.UpSampling2D(
                        size=(2, 2),
                        data_format="channels_last",
                        interpolation="bilinear",
                        name="pv_block_" + idx + "_upsampling",
                    )(x)
                else:
                    x = tf.compat.v1.keras.layers.UpSampling2D(
                        size=(2, 2),
                        data_format="channels_last",
                        interpolation="nearest",
                        name="pv_block_" + idx + "_upsampling",
                    )(x)
        else:
            x = tf.compat.v1.keras.layers.UpSampling2D(
                size=(2, 2),
                data_format="channels_last",
                interpolation="bilinear",
                name="pv_block_" + idx + "_upsampling",
            )(x)

    return x


def CASAPose(
    layer_params,
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
    learn_upsampling=False,
    **kwargs,
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

    [x2s, x4s, x8s, _, x32s] = backbone(backbone.inputs[0])
    backbone_features = [x32s, x8s, x4s, x2s, backbone.inputs[0]]
    layer_dims = [fcdim, s8dim, s4dim, s2dim, raw_dim]
    num_layers = 5
    # Build fist decoder with five layers
    reuse_convs = [None] * num_layers
    x = None
    for i in range(num_layers):
        name = str(i + 1)  # layer name
        reuse_conv = layer_params[i].reuse_conv
        upsample = 0 < i < 4  # do not upsample in last layer
        inp = backbone_features[i]
        if i > 0:
            inp = layers.concatenate([x, inp], 3)
        if reuse_conv:
            p_name = "pv_block_" + name + "_" + str(i + 1 + num_layers) + "_conv2d"
            partial = PartialConvolution(name=p_name, dim=layer_dims[i], num_classes=seg_dim)
            inp = partial([inp])

            # if the convolution weight of the first layer are reused they do not have to be recalculated later
            if i == 0:
                y = inp
            reuse_convs[i] = partial

        x = casa_layer(inp, name, layer_dims[i], leaky=(i > 0), upsampling=upsample, skip_conv=reuse_conv)

    x = layers.Conv2D(seg_dim, (1, 1), strides=(1, 1), name="pv_final_conv_segmentation", **get_conv_params())(x)

    beta = tf.cast(1e6, dtype=x.dtype)  # 10
    if input_segmentation_shape is None:
        x_mask = layers.Activation("softmax")(x * beta)
    else:
        segmentation_input = layers.Input(
            shape=input_segmentation_shape, name="data_segmentation", dtype=x.dtype
        )  # segmentation input will not work for this dasign
        x_mask = layers.Activation("softmax")(segmentation_input * beta)

    if not learn_upsampling:
        x_mask = tf.stop_gradient(x_mask)

    x_mask2s = HalfSize(name="segmentation_half_size", depth=seg_dim, trainable=learn_upsampling)(x_mask)
    x_mask4s = HalfSize(name="segmentation_quater_size", depth=seg_dim, trainable=learn_upsampling)(x_mask2s)
    x_mask8s = HalfSize(name="segmentation_eighth_size", depth=seg_dim, trainable=learn_upsampling)(x_mask4s)

    seg_masks = [x_mask8s, x_mask8s, x_mask4s, x_mask2s, x_mask, None]

    # Build second decoder
    for i in range(num_layers):
        name = str(i + 1 + num_layers)  # layer name
        upsample = 0 < i < 4  # do not upsample in last layer

        # use the next larger mask to guide upsampling
        guided_upsampling_mask = seg_masks[i + 1] if layer_params[i].guided_upsampling else None
        seg_mask = seg_masks[i] if layer_params[i].weighted_clade else None
        inp = backbone_features[i]
        if i > 0:
            inp = layers.concatenate([y, inp], 3)
        if layer_params[i].reuse_conv:
            inp = y if i == 0 else reuse_convs[i]([inp])

        print("___")
        print(name)
        print(layer_dims[i])
        print("seg_masks: {}".format(seg_masks[i].shape))
        print("upsample: {}".format(upsample))
        print("weighted_clade: {}".format(layer_params[i].weighted_clade))
        print("partial_conv: {}".format(layer_params[i].partial_conv))
        if guided_upsampling_mask is not None:
            print("guided_upsampling_mask: {}".format(guided_upsampling_mask.shape))

        y = casa_layer(
            inp,
            name,
            layer_dims[i],
            seg_mask=seg_mask,
            num_classes=seg_dim,
            leaky=(i > 0),
            upsampling=upsample,
            upsampling_bilinear=layer_params[i].bilinear_upsampling,
            weighted_clade=layer_params[i].weighted_clade,
            skip_conv=layer_params[i].reuse_conv,
            partial_conv=layer_params[i].partial_conv and not layer_params[i].reuse_conv,
            seg_mask_guide=guided_upsampling_mask,
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


setattr(CASAPoseConditional, "__doc__", ResNet.__doc__)
