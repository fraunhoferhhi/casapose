import tensorflow as tf
from casapose.pose_estimation.bpnp_layers import BPNP_fast  # noqa: E402
from casapose.pose_estimation.ransac_voting import (  # noqa: E402
    project_tf_batch,
    transform_points_back_tf_batch,
)
from casapose.utils.geometry_utils import rodrigues_batch  # noqa: E402

##################################################
# LOSS FUNCTIONS
##################################################


def smooth_l1_loss(
    vertex_pred,
    vertex_targets,
    vertex_weights,
    ignore_weights=False,
    invert_weights=False,
    normalize=True,
    reduce=True,
):
    b, h, w, ver_dim = vertex_pred.shape
    if ignore_weights:
        vertex_weights = tf.ones_like(vertex_weights)
    elif invert_weights:
        vertex_weights = tf.math.abs(tf.ones_like(vertex_weights, dtype=vertex_weights.dtype) - vertex_weights)

    vertex_diff = tf.subtract(vertex_pred, vertex_targets)
    vertex_diff = tf.multiply(vertex_weights, vertex_diff)

    vertex_diff = tf.abs(vertex_diff)
    smoothL1_sign = tf.less(vertex_diff, 1.0)
    in_loss = tf.where(smoothL1_sign, tf.square(vertex_diff) * 0.5, (vertex_diff - 0.5))

    if normalize:
        in_loss = tf.reduce_sum(tf.reshape(in_loss, [b, -1]), 1) / (
            ver_dim * tf.reduce_sum(tf.reshape(vertex_weights, [b, -1]), 1) + 1e-3
        )  # calculate sum per batch

    if reduce:
        in_loss = tf.reduce_mean(input_tensor=in_loss)

    return in_loss


def proxy_voting_dist(
    vertex_pred,
    keypoint_targets,
    vertex_one_hot_weights,
    vertex_weights,
    invert_weights=False,
    min_object_pixel=20,
):
    b, h, w, ver_dim = vertex_pred.shape
    _, _, _, object_count = vertex_one_hot_weights.shape
    _, _, _, keypoint_count, _ = keypoint_targets.shape

    if object_count > 1 and ver_dim == object_count * keypoint_count * 2:
        vertex_pred = tf.reshape(
            vertex_pred,
            [
                vertex_pred.shape[0],
                vertex_pred.shape[1],
                vertex_pred.shape[2],
                object_count,
                keypoint_count,
                2,
            ],
        )
        argmax_segmentation = tf.math.argmax(vertex_one_hot_weights, axis=3)
        vertex_pred = tf.gather(vertex_pred, argmax_segmentation, batch_dims=3)
        vertex_pred = tf.where(tf.expand_dims(vertex_weights > 0, -1), 0.0, vertex_pred)
        vertex_pred = tf.reshape(
            vertex_pred,
            [
                vertex_pred.shape[0],
                vertex_pred.shape[1],
                vertex_pred.shape[2],
                keypoint_count * 2,
            ],
        )
        ver_dim = vertex_pred.shape[-1]

    if invert_weights:
        vertex_weights = tf.math.abs(
            tf.ones_like(vertex_weights, dtype=vertex_weights.dtype) - vertex_weights
        )  # vertex weights need to be inverted

    vertex_weights_argmax = tf.expand_dims(tf.argmax(vertex_one_hot_weights, axis=-1), axis=-1)
    keypoint_targets = tf.gather_nd(keypoint_targets, vertex_weights_argmax, batch_dims=1)
    k_x, k_y = tf.split(keypoint_targets, 2, 5)
    keypoint_targets = tf.concat([k_y, -k_x], 5)  # [b, h, w, 1, vn, 2]

    vertex_pred = tf.reshape(vertex_pred, [b, h, w, tf.cast(ver_dim / 2, tf.int32), 2])
    vertex_norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(vertex_pred), axis=4)), 3)

    c_x, c_y = tf.meshgrid(tf.range(0, w, 1), tf.range(0, h, 1))  # c_x counts in x direction
    c_x = tf.cast(c_x, dtype=tf.float32) + 0.5
    c_y = tf.cast(c_y, dtype=tf.float32) + 0.5
    coords = tf.expand_dims(
        tf.expand_dims(tf.stack([-c_x, c_y], 2), 2), 0
    )  # cy_first and negative c_x for substraction [1, h, w, 1, 2]
    d_2 = tf.expand_dims(tf.reduce_sum(tf.multiply(vertex_pred, coords), 4), 3)

    vertex_pred = tf.expand_dims(vertex_pred, 3)
    d_1 = tf.reduce_sum(tf.multiply(vertex_pred, keypoint_targets), 5)

    dist = tf.math.reduce_min(
        tf.math.divide_no_nan(tf.abs(d_1 + d_2), vertex_norm), axis=3
    )  # [b, h, w, vn] # perp_foot_dist
    dist = tf.multiply(vertex_weights, dist)
    dist = tf.abs(dist)

    mask_sum = tf.reduce_sum(tf.reduce_sum(vertex_one_hot_weights, axis=1), axis=1)  # [b, h, w, object_count]
    valid = tf.where(tf.greater_equal(mask_sum, min_object_pixel), 1.0, 0.0)
    in_loss = tf.reduce_sum(
        tf.where(tf.less(dist, 1.0), tf.square(dist) * 0.5, (dist - 0.5)), axis=-1
    )  # [b, h, w, vn]
    in_loss = tf.map_fn(
        lambda x: tf.math.unsorted_segment_sum(x[0], x[1], object_count),
        elems=(in_loss, tf.squeeze(vertex_weights_argmax, -1)),
        dtype=tf.float32,
    )
    in_loss = tf.math.divide_no_nan(
        valid * in_loss, (ver_dim / 2) * mask_sum + 1e-3
    )  # correct normaization is used here !

    return dist, in_loss


def proxy_voting_loss_v2(
    vertex_pred,
    keypoint_targets,
    vertex_one_hot_weights,
    vertex_weights,
    invert_weights=False,
    normalize=True,
    reduce=True,
    loss_per_object=False,
    min_object_pixel=20,
):

    b, h, w, ver_dim = vertex_pred.shape
    _, _, _, object_count = vertex_one_hot_weights.shape
    if invert_weights:
        vertex_weights = tf.math.abs(
            tf.ones_like(vertex_weights, dtype=vertex_weights.dtype) - vertex_weights
        )  # vertex weights need to be inverted

    vertex_weights_argmax = tf.expand_dims(tf.argmax(vertex_one_hot_weights, axis=-1), axis=-1)
    keypoint_targets = tf.gather_nd(keypoint_targets, vertex_weights_argmax, batch_dims=1)
    k_x, k_y = tf.split(keypoint_targets, 2, 5)
    keypoint_targets = tf.concat([k_y, -k_x], 5)  # [b, h, w, 1, vn, 2]

    vertex_pred = tf.reshape(vertex_pred, [b, h, w, tf.cast(ver_dim / 2, tf.int32), 2])
    vertex_norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(vertex_pred), axis=4)), 3)

    c_x, c_y = tf.meshgrid(tf.range(0, w, 1), tf.range(0, h, 1))  # c_x counts in x direction
    c_x = tf.cast(c_x, dtype=tf.float32) + 0.5
    c_y = tf.cast(c_y, dtype=tf.float32) + 0.5
    coords = tf.expand_dims(
        tf.expand_dims(tf.stack([-c_x, c_y], 2), 2), 0
    )  # cy_first and negative c_x for substraction [1, h, w, 1, 2]
    d_2 = tf.expand_dims(tf.reduce_sum(tf.multiply(vertex_pred, coords), 4), 3)

    vertex_pred = tf.expand_dims(vertex_pred, 3)
    d_1 = tf.reduce_sum(tf.multiply(vertex_pred, keypoint_targets), 5)

    dist = tf.math.reduce_min(
        tf.math.divide_no_nan(tf.abs(d_1 + d_2), vertex_norm), axis=3
    )  # [b, h, w, vn] # perp_foot_dist
    dist = tf.multiply(vertex_weights, dist)
    dist = tf.abs(dist)

    if loss_per_object and normalize:
        mask_sum = tf.reduce_sum(tf.reduce_sum(vertex_one_hot_weights, axis=1), axis=1)  # [b, h, w, object_count]
        valid = tf.where(tf.greater_equal(mask_sum, min_object_pixel), 1.0, 0.0)
        count = tf.math.count_nonzero(valid, axis=1)
        in_loss = tf.reduce_sum(
            tf.where(tf.less(dist, 1.0), tf.square(dist) * 0.5, (dist - 0.5)), axis=-1
        )  # [b, h, w, vn]
        in_loss = tf.map_fn(
            lambda x: tf.math.unsorted_segment_sum(x[0], x[1], object_count),
            elems=(in_loss, tf.squeeze(vertex_weights_argmax, -1)),
            dtype=tf.float32,
        )
        in_loss = tf.math.divide_no_nan(
            valid * in_loss, (ver_dim * mask_sum + 1e-3)
        )  # actually a division by ver_dim/2 is needed here, since the errors are given for each keypoint
        in_loss = tf.math.divide_no_nan(tf.reduce_sum(tf.reshape(in_loss, [b, -1]), 1), tf.cast(count, tf.float32))

    else:
        # reminder: because of smooth l1 loss the output is not the same as the mean perp foot dist
        in_loss = tf.where(tf.less(dist, 1.0), tf.square(dist) * 0.5, (dist - 0.5))  # [b, h, w, vn]
        if normalize:
            in_loss = tf.reduce_sum(tf.reshape(in_loss, [b, -1]), 1) / (
                ver_dim * tf.reduce_sum(tf.reshape(vertex_weights, [b, -1]), 1) + 1e-3
            )  # calculate average of non zero entrys per batch
    if reduce:
        in_loss = tf.reduce_mean(input_tensor=in_loss)

    return in_loss


# @tf.function
def keypoint_reprojection_loss(
    points_estimated,
    seg_estimated,
    poses_gt,
    object_points_3d,
    target_seg,
    camera_data,
    offsets,
    confidence,
    max_pixel_error=25.0,
    confidence_regularization=False,
    points_gt=None,
    min_num=20,
    min_num_gt=-1,
    use_bpnp_reprojection_loss=False,
    estimate_poses=False,
    filter_with_gt=True,
):
    b, h, w, c = target_seg.shape
    b, oc, ic, _, _ = poses_gt.shape
    _, _, _, vc, _ = object_points_3d.shape

    offsets = tf.broadcast_to(tf.expand_dims(offsets, 1), [b, oc, 10])

    points_estimated = tf.reshape(points_estimated, [-1, vc, 2])
    offsets = tf.reshape(offsets, [-1, 10])
    object_points_3d = tf.reshape(object_points_3d, [-1, vc, 3])
    poses_gt = tf.reshape(poses_gt, [-1, 3, 4])
    points_estimated = tf.reverse(points_estimated, axis=[2])
    seg_estimated = tf.stop_gradient(seg_estimated)
    beta = tf.cast(1e6, dtype=seg_estimated.dtype)
    hot_seg = tf.expand_dims(tf.expand_dims(tf.nn.softmax(seg_estimated * beta), -1), -1)[
        :, :, :, 1:, :, :
    ]  # [bs, w, h, oc, 1, 1]

    objects_pixel_count_gt = tf.math.count_nonzero(
        tf.reshape(target_seg[:, :, :, 1:], [b, h * w, -1]), 1
    )  # [b, object_count]
    objects_pixel_count_est = tf.math.count_nonzero(
        tf.cast(tf.greater(tf.reshape(hot_seg, [b, h * w, -1]), 0.1), tf.float32), 1
    )  # [b, object_count]

    objects_available = tf.where(objects_pixel_count_est > min_num, 1, 0)
    objects_available = tf.reshape(objects_available, [-1, 1])

    if filter_with_gt:
        if min_num_gt < 0:
            min_num_gt = min_num
        objects_available_2 = tf.where(objects_pixel_count_gt > min_num_gt, 1, 0)
        objects_available_2 = tf.reshape(objects_available_2, [-1, 1])
        objects_available = objects_available * objects_available_2

    if confidence_regularization:
        confidence = tf.math.softplus(confidence)
        mask = tf.abs(target_seg[:, :, :, 0:1] - 1.0)
        hot_conf = confidence * mask
        confidence_sum = tf.reduce_sum(hot_conf, axis=[1, 2], keepdims=True)
        mask_sum = tf.reduce_sum(mask, axis=[1, 2, 3], keepdims=True)
        confidence_loss = tf.math.divide_no_nan(confidence_sum, mask_sum)
        confidence_loss = tf.reduce_mean(tf.abs(confidence_loss - 0.7))

    points_estimated = transform_points_back_tf_batch(
        points_estimated,
        offsets[:, 0:1],
        offsets[:, 1:2],
        offsets[:, 8:9],
        offsets[:, 9:10],
        offsets[:, 4:5],
        offsets[:, 5:6],
        offsets[:, 6:7],
        offsets[:, 7:8],
    )

    object_points_3d = tf.stop_gradient(object_points_3d)
    camera_data = tf.stop_gradient(camera_data)
    objects_available = tf.cast(tf.expand_dims(objects_available, 1), tf.float32)

    poses_est = None

    ############################
    if use_bpnp_reprojection_loss or estimate_poses:
        poses_est = BPNP_fast(name="BPNP")(
            [points_estimated, object_points_3d, camera_data[0]]
        )  # offsets are neeeded!

        is_finite = tf.reduce_all(tf.math.is_finite(poses_est))
        if not is_finite:
            tf.print("Tensor had inf or nan values:")
            tf.print(poses_est, summarize=-1)
            tf.print(points_estimated, summarize=-1)
            tf.Assert(is_finite, ["end:", is_finite])

        R_out = rodrigues_batch(poses_est[:, 0:3])  # [bs, 3,3]
        T_out = tf.expand_dims(poses_est[:, 3:6], -1)  # bs, 3, 1]

        poses_est = tf.concat([R_out, T_out], axis=-1)
        poses_est = tf.where(T_out[:, 2:3, :] < 0, -poses_est, poses_est)

        reproj_est, _ = project_tf_batch(object_points_3d, camera_data[0], poses_est)
        reproj_est = reproj_est * objects_available
        poses_est = poses_est * objects_available
        poses_est = tf.reshape(poses_est, [b, oc, ic, 3, 4])
    ##############################

    reproj_gt, _ = project_tf_batch(object_points_3d, camera_data[0], poses_gt)
    reproj_gt = reproj_gt * objects_available
    points_estimated = points_estimated * objects_available

    if use_bpnp_reprojection_loss:
        l1 = tf.abs(reproj_est - points_estimated)
        l1 = tf.norm(l1, ord="euclidean", axis=-1, keepdims=True)
        l2 = tf.abs(reproj_gt - reproj_est)
        l2 = tf.norm(l2, ord="euclidean", axis=-1, keepdims=True)
        loss = (l1 + l2) / 2.0
    else:
        loss = tf.abs(reproj_gt - points_estimated)
        loss = tf.norm(loss, ord="euclidean", axis=-1, keepdims=True)

    smoothL1_sign = tf.less(loss, 1.0)

    loss = tf.where(smoothL1_sign, tf.square(loss) * 0.5, (loss - 0.5))

    loss = tf.where(
        tf.greater(loss, max_pixel_error),
        max_pixel_error + ((loss - max_pixel_error) * 0.01),
        loss,
    )
    loss = loss * objects_available
    loss = tf.reduce_mean(loss, axis=[1, 2])

    loss = tf.math.divide_no_nan(tf.reduce_sum(loss), tf.reduce_sum(objects_available))

    if confidence_regularization:
        loss = loss + confidence_loss

    points_estimated = tf.reshape(points_estimated, [b, oc, vc, 2])

    return loss, poses_est, points_estimated
