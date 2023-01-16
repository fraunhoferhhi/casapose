import tensorflow as tf
from casapose.pose_estimation.bpnp_layers import BPNP_fast  # noqa: E402
from casapose.pose_estimation.ransac_voting import (  # noqa: E402
    estimate_poses,
    evaluate_poses,
    ransac_voting_layer_all_masks,
)
from casapose.utils.geometry_utils import rodrigues_batch  # noqa: E402


def estimate_and_evaluate_poses(
    output_seg,
    target_seg,
    output_vertex,
    poses_gt,
    object_points_3d,
    camera_data,
    diameters,
    offsets,
    evaluation_points=None,
    object_points_3d_count=None,
    points_estimated=None,
    min_num=20,
):
    b, h, w, c = target_seg.shape
    _, oc, ic, _, _ = poses_gt.shape
    _, _, _, vc, _ = object_points_3d.shape

    # check in which images the object is visible in gt data
    objects_available = tf.math.count_nonzero(
        tf.reshape(target_seg[:, :, :, 1:], [b, h * w, -1]), 1
    )  # [b, object_count]
    objects_available = tf.where(objects_available > min_num, 1, 0)

    # create one hot vector representation
    argmax_segmentation = tf.math.argmax(output_seg, axis=3)
    mask_one_hot = tf.one_hot(argmax_segmentation, c)
    if oc > 1 and output_vertex.shape[-1] == vc * oc * 2:
        output_vertex = tf.reshape(output_vertex, [b, h, w, oc, vc, 2])
        output_vertex = tf.gather(output_vertex, tf.nn.relu(argmax_segmentation - 1), batch_dims=3)
        output_vertex = tf.where(
            tf.expand_dims(tf.expand_dims(argmax_segmentation == 0, -1), -1),
            0.0,
            output_vertex,
        )

    output_vertex = tf.reshape(output_vertex, [b, h, w, vc, 2])

    if points_estimated is None:
        points_estimated = ransac_voting_layer_all_masks(
            mask_one_hot[:, :, :, 1:],
            output_vertex,
            512,
            inlier_thresh=0.99,
            max_iter=20,
            min_num=min_num,
            max_num=30000,
        )  # [b,oc,vn,2]
    else:
        points_estimated = tf.math.multiply(points_estimated, tf.constant([[[[h, w]]]], dtype=tf.float32))

    # filtering happens inside
    poses, false_positive_mask = estimate_poses(
        points_estimated, object_points_3d, camera_data, objects_available, offsets
    )

    if evaluation_points is not None and object_points_3d_count is not None:
        evaluation_points = tf.expand_dims(tf.expand_dims(evaluation_points, 0), 2)
        evaluation_points = tf.tile(evaluation_points, [b, 1, ic, 1, 1])
        object_points_3d = evaluation_points
        object_points_3d_count = tf.expand_dims(object_points_3d_count, 0)  # [1, oc, 1]
        object_points_3d_count = tf.tile(object_points_3d_count, [b, 1, ic])
    else:
        object_points_3d_count = tf.ones([b, oc, ic], dtype=tf.int32) * 9

    (err_2d, err_3d, valid_2d, valid_3d, missing_object, valid_pose_count, false_positive_pose,) = evaluate_poses(
        poses,
        poses_gt,
        points_estimated,
        object_points_3d,
        object_points_3d_count,
        camera_data,
        diameters,
        objects_available,
        5.0,
    )

    return (
        [
            valid_2d,
            valid_3d,
            valid_pose_count,
            false_positive_mask,
            err_2d,
            err_3d,
            missing_object,
            false_positive_pose,
        ],
        poses,
        points_estimated,
    )


def evaluate_pose_estimates(
    points_estimated,
    poses,
    poses_gt,
    target_seg,
    object_points_3d,
    camera_data,
    diameters,
    evaluation_points=None,
    object_points_3d_count=None,
    min_num=20,
):

    b, h, w, c = target_seg.shape
    _, oc, ic, _, _ = poses_gt.shape

    # check in which images the object is visible in gt data
    objects_available = tf.math.count_nonzero(
        tf.reshape(target_seg[:, :, :, 1:], [b, h * w, -1]), 1
    )  # [b, object_count]
    objects_available = tf.where(objects_available > min_num, 1, 0)

    if evaluation_points is not None and object_points_3d_count is not None:
        evaluation_points = tf.expand_dims(tf.expand_dims(evaluation_points, 0), 2)
        evaluation_points = tf.tile(evaluation_points, [b, 1, ic, 1, 1])
        object_points_3d = evaluation_points
        object_points_3d_count = tf.expand_dims(object_points_3d_count, 0)  # [1, oc, 1]
        object_points_3d_count = tf.tile(object_points_3d_count, [b, 1, ic])
    else:
        object_points_3d_count = tf.ones([b, oc, ic], dtype=tf.int32) * 9

    (err_2d, err_3d, valid_2d, valid_3d, missing_object, valid_pose_count, false_positive_pose,) = evaluate_poses(
        poses,
        poses_gt,
        points_estimated,
        object_points_3d,
        object_points_3d_count,
        camera_data,
        diameters,
        objects_available,
        5.0,
    )

    return (
        [
            valid_2d,
            valid_3d,
            valid_pose_count,
            tf.zeros_like(valid_2d),
            err_2d,
            err_3d,
            missing_object,
            false_positive_pose,
        ],
        poses,
        points_estimated,
    )


@tf.function
def poses_pnp(
    points_estimated,
    seg_estimated,
    object_points_3d,
    camera_data,
    no_objects,
    min_num=20,
):
    b, h, w, _ = seg_estimated.shape
    oc = no_objects
    ic = 1
    _, _, _, vc, _ = object_points_3d.shape

    points_estimated = tf.reshape(points_estimated, [-1, vc, 2])
    object_points_3d = tf.reshape(object_points_3d, [-1, vc, 3])
    points_estimated = tf.reverse(points_estimated, axis=[2])

    seg_estimated = tf.stop_gradient(seg_estimated)
    beta = tf.cast(1e6, dtype=seg_estimated.dtype)
    hot_seg = tf.expand_dims(tf.expand_dims(tf.nn.softmax(seg_estimated * beta), -1), -1)[
        :, :, :, 1:, :, :
    ]  # [bs, w, h, oc, 1, 1]
    objects_pixel_count_est = tf.math.count_nonzero(
        tf.cast(tf.greater(tf.reshape(hot_seg, [b, h * w, -1]), 0.1), tf.float32), 1
    )  # [b, object_count]

    objects_available = tf.where(objects_pixel_count_est > min_num, 1, 0)
    objects_available = tf.reshape(objects_available, [-1, 1])

    object_points_3d = tf.stop_gradient(object_points_3d)
    camera_data = tf.stop_gradient(camera_data)
    points_estimated = tf.stop_gradient(points_estimated)

    objects_available = tf.cast(tf.expand_dims(objects_available, 1), tf.float32)

    poses_est = BPNP_fast(name="BPNP")([points_estimated, object_points_3d, camera_data[0]])  # offsets are neeeded!

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
    poses_est = poses_est * objects_available
    poses_est = tf.reshape(poses_est, [b, oc, ic, 3, 4])
    ##############################

    return poses_est


# unused
@tf.function
def pose_estimation(
    output_seg,
    target_seg,
    output_vertex,
    poses_gt,
    object_points_3d,
    camera_data,
    offsets,
    points_estimated=None,
    min_num=20,
):

    b, h, w, c = target_seg.shape
    _, oc, ic, _, _ = poses_gt.shape
    _, _, _, vc, _ = object_points_3d.shape

    # check in which images the object is visible in gt data
    objects_available = tf.ones([b, oc])
    # create one hot vector representation
    argmax_segmentation = tf.math.argmax(output_seg, axis=3)
    mask_one_hot = tf.one_hot(argmax_segmentation, c)
    if oc > 1 and output_vertex.shape[-1] == vc * oc * 2:
        output_vertex = tf.reshape(output_vertex, [b, h, w, oc, vc, 2])
        output_vertex = tf.gather(output_vertex, tf.nn.relu(argmax_segmentation - 1), batch_dims=3)
        output_vertex = tf.where(
            tf.expand_dims(tf.expand_dims(argmax_segmentation == 0, -1), -1),
            0.0,
            output_vertex,
        )

    output_vertex = tf.reshape(output_vertex, [b, h, w, vc, 2])

    if points_estimated is None:
        points_estimated = ransac_voting_layer_all_masks(
            mask_one_hot[:, :, :, 1:],
            output_vertex,
            512,
            inlier_thresh=0.99,
            max_iter=20,
            min_num=min_num,
            max_num=30000,
        )  # [b,oc,vn,2]
    else:
        points_estimated = tf.math.multiply(points_estimated, tf.constant([[[[h, w]]]], dtype=tf.float32))

    poses, _ = estimate_poses(points_estimated, object_points_3d, camera_data, objects_available, offsets)

    return poses
