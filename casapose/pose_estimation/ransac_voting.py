import math

import cv2
import numpy as np
import tensorflow as tf

"""
Our implementation of ransac voting is based on code for paper "PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation Sida"
by Sida Peng et al., ZJU-SenseTime Joint Lab of 3D Vision (https://github.com/zju3dv/pvnet).
"""


def pnp(points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_EPNP):  # cv2.SOLVEPNP_ITERATIVE

    assert points_3d.shape[0] == points_2d.shape[0], "points 3D and points 2D must have same number of vertices"

    if np.abs(np.sum(points_2d)) < 1e-4:
        return np.zeros([3, 4]).astype(np.float32)

    if method == cv2.SOLVEPNP_EPNP:
        points_3d = np.expand_dims(points_3d, 0)
        points_2d = np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))

    camera_matrix = camera_matrix.astype(np.float64)
    _, rvec0, T0, _ = cv2.solvePnPRansac(
        points_3d,
        points_2d,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_EPNP,
        confidence=0.9999,
        reprojectionError=12,
    )
    ret, R_exp, t = cv2.solvePnP(
        points_3d,
        points_2d,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        useExtrinsicGuess=True,
        rvec=rvec0,
        tvec=T0,
    )

    if ret is False or np.isnan(np.sum(t)):
        return np.zeros([3, 4]).astype(np.float32)

    R, _ = cv2.Rodrigues(R_exp)

    if t[2] < 0:
        t *= -1
        R *= -1

    return np.concatenate([R, t], axis=-1).astype(np.float32)


def get_rotation_matrix_2D(center, angle):
    angle_rad = angle * (np.pi / 180)
    a = np.cos(angle_rad)
    b = np.sin(angle_rad)
    c = (1 - a) * center[0] - b * center[1]
    d = b * center[0] + (1 - a) * center[1]

    rot_mat = np.float32([[a, b, c], [-b, a, d]])
    return rot_mat


def transform_points_back(points, w_crop, h_crop, sx, sy, dx, dy, angle, scale):
    proj_points = np.array(points)

    proj_points /= scale
    tm = np.float32([[1, 0, -dx], [0, 1, -dy]])
    rm = get_rotation_matrix_2D((sx / 2, sy / 2), -angle)

    rmat = np.identity(3)
    rmat[0:2] = rm
    tmat = np.identity(3)
    tmat[0:2] = tm

    proj_points += [w_crop, h_crop]
    new_points = proj_points[0:2].T
    new_points = np.matmul(tmat, np.vstack((proj_points.T, np.ones(len(points)))))
    new_points = np.matmul(rmat, new_points)
    new_points = new_points[0:2].T

    return new_points.astype(np.float32)


def transform_points_back_tf(points, h_crop, w_crop, sx, sy, dx, dy, angle, scale):

    proj_points = points
    proj_points = tf.math.divide(proj_points, scale)
    tm = tf.stack(
        [
            tf.stack([1.0, 0.0, -dx]),
            tf.stack([0.0, 1.0, -dy]),
            tf.stack([0.0, 0.0, 1.0]),
        ]
    )

    center = tf.stack([sx / 2.0, sy / 2.0])
    angle_rad = -angle * (math.pi / 180)
    a = tf.math.cos(angle_rad)
    b = tf.math.sin(angle_rad)
    c = (1.0 - a) * center[0] - b * center[1]
    d = b * center[0] + (1.0 - a) * center[1]
    rm = tf.stack([tf.stack([a, b, c]), tf.stack([-1.0 * b, a, d]), tf.stack([0.0, 0.0, 1.0])])

    proj_points += tf.stack([w_crop, h_crop])
    new_points = tf.transpose(proj_points)
    new_points = tf.matmul(
        tm,
        tf.concat([new_points, tf.ones([1, points.shape[0]], dtype=tf.float32)], axis=0),
    )
    new_points = tf.matmul(rm, new_points)
    new_points = tf.transpose(new_points[0:2])

    return tf.cast(new_points, dtype=tf.float32)


def transform_points_back_tf_batch(points, h_crop, w_crop, sx, sy, dx, dy, angle, scale):
    bs = tf.shape(points)[0]
    vc = tf.shape(points)[1]
    proj_points = points
    proj_points = tf.math.divide(proj_points, tf.expand_dims(scale, axis=-1))
    z = tf.zeros_like(dx)
    o = tf.ones_like(dx)
    tm = tf.stack(
        [
            tf.concat([o, z, -dx], axis=1),
            tf.concat([z, o, -dy], axis=1),
            tf.concat([z, z, o], axis=1),
        ],
        axis=1,
    )
    center = tf.stack([sx / 2.0, sy / 2.0], axis=1)
    angle_rad = -angle * (math.pi / 180)
    a = tf.math.cos(angle_rad)
    b = tf.math.sin(angle_rad)
    c = (1.0 - a) * center[:, 0] - b * center[:, 1]
    d = b * center[:, 0] + (1.0 - a) * center[:, 1]
    rm = tf.stack(
        [
            tf.concat([a, b, c], axis=1),
            tf.concat([-1.0 * b, a, d], axis=1),
            tf.concat([z, z, o], axis=1),
        ],
        axis=1,
    )
    proj_points += tf.stack([w_crop, h_crop], axis=2)
    new_points = tf.transpose(proj_points, perm=[0, 2, 1])
    new_points = tf.matmul(tm, tf.concat([new_points, tf.ones([bs, 1, vc], dtype=tf.float32)], axis=1))
    new_points = tf.matmul(rm, new_points)
    new_points = tf.transpose(new_points[:, 0:2], perm=[0, 2, 1])
    return tf.cast(new_points, dtype=tf.float32)


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz_proj = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz_proj, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy.astype(np.float32), xyz_proj.astype(np.float32)


def project_tf(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz_proj = tf.matmul(xyz, tf.transpose(RT[:, :3])) + tf.transpose(RT[:, 3:])
    xyz = tf.matmul(xyz_proj, tf.transpose(K))
    xy = xyz[:, :2] / xyz[:, 2:]
    return tf.cast(xy, dtype=tf.float32), tf.cast(xyz_proj, dtype=tf.float32)


def project_tf_batch(xyz, K, RT):
    """
    xyz: [B, N, 3]
    K: [3, 3]
    RT: [B, 3, 4]
    """
    xyz_proj = tf.matmul(xyz, tf.transpose(RT[:, :, :3], perm=[0, 2, 1])) + tf.transpose(RT[:, :, 3:], perm=[0, 2, 1])
    xyz = tf.matmul(xyz_proj, tf.expand_dims(tf.transpose(K), 0))
    xy = tf.math.divide_no_nan(xyz[:, :, :2], xyz[:, :, 2:])
    return tf.cast(xy, dtype=tf.float32), tf.cast(xyz_proj, dtype=tf.float32)


def generate_hypothesis(direct, coords, idxs):  # index can be two times the same
    """
    :param direct:      [tn,vn,2]
    :param coords:      [tn,2]
    :param idxs:        [hn,vn,2]
    :return: [hn,vn,2]
    """
    hn = tf.shape(idxs)[0]
    vn = tf.shape(idxs)[1]
    idxs_help = tf.expand_dims(
        tf.stack(
            [
                tf.broadcast_to(tf.range(vn), [hn, vn]),
                tf.broadcast_to(tf.range(vn), [hn, vn]),
            ],
            axis=2,
        ),
        3,
    )
    c_s = tf.gather(coords, idxs)
    d_s = tf.gather_nd(direct, tf.concat([tf.expand_dims(idxs, axis=3), idxs_help], axis=3))

    det = d_s[:, :, 1, 0] * d_s[:, :, 0, 1] - d_s[:, :, 1, 1] * d_s[:, :, 0, 0]

    u = (
        (c_s[:, :, 1, 1] - c_s[:, :, 0, 1]) * d_s[:, :, 1, 0] - (c_s[:, :, 1, 0] - c_s[:, :, 0, 0]) * d_s[:, :, 1, 1]
    ) / det

    hypo_pts = c_s[:, :, 0] + (d_s[:, :, 0] * tf.expand_dims(u, axis=2))
    hypo_pts = tf.where(tf.expand_dims((tf.abs(det) > 1e-6), 2), hypo_pts, [0, 0])
    return hypo_pts


def voting_for_hypothesis(direct, coords, cur_hyp_pts, inlier_thresh):
    # cautin cur_hyp_pts can contain zero
    coords = tf.expand_dims(tf.expand_dims(coords, 1), 0)
    direct = tf.expand_dims(direct, 0)

    cur_hyp_pts = tf.expand_dims(cur_hyp_pts, 1)
    hypo_dirs = cur_hyp_pts - coords  # connect the coordinates to the hypothesises

    norm_dir = tf.norm(direct, axis=-1)
    norm_hyp = tf.norm(hypo_dirs, axis=-1)
    valid_norm = tf.math.logical_and(tf.greater(norm_dir, 1e-6), tf.greater(norm_hyp, 1e-6))
    valid_norm = tf.math.logical_and(
        valid_norm, tf.abs(tf.reduce_sum(cur_hyp_pts, axis=-1)) > 1e-6
    )  # filter invalid points (necessary?)

    angle_dist = tf.reduce_sum((direct * hypo_dirs), axis=-1) / (norm_dir * norm_hyp)  # calculation is ok

    cur_inlier = tf.where(tf.math.logical_and(valid_norm, tf.greater(angle_dist, inlier_thresh)), 1, 0)

    return cur_inlier


# https://stackoverflow.com/questions/57073381/how-to-check-if-a-matrix-is-invertible-in-tensorflow
# Based on np.linalg.cond(x, p=None)
@tf.function
def tf_cond(x):
    x = tf.convert_to_tensor(x)
    s = tf.linalg.svd(x, compute_uv=False)
    r = s[..., 0] / s[..., -1]
    # Replace NaNs in r with infinite unless there were NaNs before
    x_nan = tf.reduce_any(tf.math.is_nan(x), axis=(-2, -1))
    r_nan = tf.math.is_nan(r)
    r_inf = tf.fill(tf.shape(r), tf.constant(math.inf, r.dtype))
    tf.where(x_nan, r, tf.where(r_nan, r_inf, r))
    return r


@tf.function
def is_invertible(x, epsilon=1e-6):  # Epsilon may be smaller with tf.float64
    x = tf.convert_to_tensor(x)
    eps_inv = tf.cast(1 / epsilon, x.dtype)
    x_cond = tf_cond(x)
    return tf.math.is_finite(x_cond) & (x_cond < eps_inv)


@tf.function
def ransac_voting_batch(
    cur_mask,
    cur_vertex,
    inlier_thresh,
    confidence,
    max_iter,
    min_num,
    max_num,
    round_hyp_num,
    vn,
):
    foreground_num = tf.reduce_sum(cur_mask)

    # if too few points, just skip it
    if tf.less(foreground_num, min_num):
        win_pts = tf.zeros([vn, 2], dtype=tf.float32)
        return win_pts

    # if too many inliers, we randomly down sample it
    if tf.greater(foreground_num, max_num):
        selection = tf.random.uniform(cur_mask.shape, dtype=tf.float32)
        selected_mask = tf.cast(
            (selection < (max_num / tf.cast(foreground_num, dtype=tf.float32))),
            dtype=tf.float32,
        )
        cur_mask *= selected_mask

    coords = tf.reverse(
        tf.cast(tf.where(tf.not_equal(cur_mask, 0.0)), dtype=tf.float32), axis=[1]
    )  # tf.where is slow in tf_func since the size of the arra is unknown
    coords += 0.5
    cur_mask = tf.cast(cur_mask, dtype=tf.bool)
    direct = tf.reverse(tf.boolean_mask(cur_vertex, cur_mask), axis=[2])

    tn = tf.shape(coords)[0]
    all_win_ratio = tf.zeros([vn], dtype=tf.float32)
    all_win_pts = tf.zeros([vn, 2], dtype=tf.float32)
    cur_iter = 0
    hyp_num = 0.0

    break_loop = False

    while tf.equal(break_loop, False):
        idxs = tf.random.uniform(
            [round_hyp_num, vn, 2], minval=0, maxval=tn, dtype=tf.int32
        )  # idx can be two times the same
        cur_hyp_pts = generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

        cur_inlier = voting_for_hypothesis(direct, coords, cur_hyp_pts, inlier_thresh)  # [hn,tn, vn]

        # find max
        cur_inlier_counts = tf.reduce_sum(cur_inlier, 1)  # [hn,vn]
        cur_win_idx = tf.cast(tf.argmax(cur_inlier_counts, 0), dtype=tf.int32)  # [vn]

        cur_win_counts = tf.reduce_max(cur_inlier_counts, 0)  # [vn]

        cur_win_pts = tf.gather_nd(cur_hyp_pts, tf.stack([cur_win_idx, tf.range(vn)], axis=1))
        cur_win_ratio = tf.cast(cur_win_counts, tf.float32) / tf.cast(tn, tf.float32)

        # update best point
        larger_mask = all_win_ratio < cur_win_ratio
        all_win_pts = tf.where(tf.expand_dims(larger_mask, 1), cur_win_pts, all_win_pts)
        all_win_ratio = tf.where(larger_mask, cur_win_ratio, all_win_ratio)
        # check confidence
        hyp_num += round_hyp_num
        cur_iter += 1
        cur_min_ratio = tf.reduce_min(all_win_ratio)

        if (
            1 - (1 - cur_min_ratio**2.0) ** hyp_num
        ) > confidence or cur_iter >= max_iter:  # this is almost always true in first iteration
            break_loop = True

    normal = tf.reverse(direct * tf.constant([1, -1], dtype=tf.float32), axis=[2])  # [tn,vn,2]

    all_win_pts = tf.expand_dims(all_win_pts, 0)  # [1,vn,2]

    all_inlier = voting_for_hypothesis(direct, coords, all_win_pts, inlier_thresh)  # [1,tn,vn]
    all_inlier = tf.cast(tf.squeeze(all_inlier, 0), tf.float32)  # [vn,tn]

    normal = normal * tf.expand_dims(all_inlier, 2)  # [tn,vn,2] outlier is all zero
    normal = tf.transpose(normal, perm=[1, 0, 2])

    b = tf.reduce_sum(normal * tf.expand_dims(coords, 0), 2)  # [vn, tn]

    ATA = tf.matmul(tf.transpose(normal, perm=[0, 2, 1]), normal)  # [vn,2,2]
    ATb = tf.reduce_sum(normal * tf.expand_dims(b, 2), 1)  # [vn,2]

    if tf.reduce_min(tf.cast(is_invertible(ATA), tf.int32)) == 0:
        return tf.transpose(all_win_pts, perm=[1, 2, 0])[:, :, 0]
    else:
        all_win_pts = tf.matmul(tf.linalg.inv(ATA), tf.expand_dims(ATb, 2))  # [vn,2,1]
    return all_win_pts[:, :, 0]


def ransac_voting_layer(
    mask,
    vertex,
    round_hyp_num,
    inlier_thresh=0.99,
    confidence=0.99,
    max_iter=20,
    min_num=5,
    max_num=30000,
):
    """
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2]
    """
    _, _, _, vn, _ = tf.shape(vertex)
    inlier_thresh = tf.convert_to_tensor(inlier_thresh)
    confidence = tf.convert_to_tensor(confidence)
    max_iter = tf.convert_to_tensor(max_iter, dtype=tf.int32)
    min_num = tf.convert_to_tensor(min_num, dtype=tf.float32)
    max_num = tf.convert_to_tensor(max_num, dtype=tf.float32)
    fn = lambda x: ransac_voting_batch(
        x[0],
        x[1],
        inlier_thresh,
        confidence,
        max_iter,
        min_num,
        max_num,
        round_hyp_num,
        vn,
    )
    elems = (mask, vertex)
    batch_win_pts = tf.map_fn(fn, elems=elems, dtype=tf.float32)
    return batch_win_pts


def ransac_voting_layer_single_mask(
    mask,
    vertex,
    inlier_thresh,
    confidence,
    max_iter,
    min_num,
    max_num,
    round_hyp_num,
    vn,
):
    """
    :param mask:      [h,w,oc]
    :param vertex:    [h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [oc,vn,2]
    """

    mask = tf.transpose(mask, perm=[2, 0, 1])
    fn = lambda x: ransac_voting_batch(
        x,
        vertex,
        inlier_thresh,
        confidence,
        max_iter,
        min_num,
        max_num,
        round_hyp_num,
        vn,
    )
    elems = mask
    batch_win_pts = tf.map_fn(fn, elems=elems, dtype=tf.float32)
    return batch_win_pts


@tf.function
def ransac_voting_layer_all_masks(
    mask,
    vertex,
    round_hyp_num,
    inlier_thresh=0.99,
    confidence=0.99,
    max_iter=20,
    min_num=5,
    max_num=30000,
):
    """
    :param mask:      [b,h,w,oc]
    :param vertex:    [b,h,w,vn*2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,oc,vn,2]
    """
    _, _, _, vn, _ = vertex.shape

    inlier_thresh = tf.convert_to_tensor(inlier_thresh)
    confidence = tf.convert_to_tensor(confidence)
    max_iter = tf.convert_to_tensor(max_iter, dtype=tf.int32)
    min_num = tf.convert_to_tensor(min_num, dtype=tf.float32)
    max_num = tf.convert_to_tensor(max_num, dtype=tf.float32)
    fn = lambda x: ransac_voting_layer_single_mask(
        x[0],
        x[1],
        inlier_thresh,
        confidence,
        max_iter,
        min_num,
        max_num,
        round_hyp_num,
        vn,
    )
    elems = (mask, vertex)
    batch_win_pts = tf.map_fn(fn, elems=elems, dtype=tf.float32)
    return batch_win_pts


@tf.function
def map_offsets(points, valid_points_filter, offsets):
    if tf.math.abs(tf.reduce_sum(points)) < 0.01:  # valid_points_filter == 0:
        return tf.zeros(points.shape)
    else:
        points_shape = points.shape

        points = transform_points_back_tf(
            points,
            offsets[0],
            offsets[1],
            offsets[8],
            offsets[9],
            offsets[4],
            offsets[5],
            offsets[6],
            offsets[7],
        )
        return tf.reshape(points, points_shape)


@tf.function
def map_pnp(points, keypoints, camera_matrix, valid_points_filter):
    if tf.math.abs(tf.reduce_sum(points)) < 0.01:  # valid_points_filter == 0:
        return tf.zeros([3, 4])
    else:
        result = tf.numpy_function(pnp, inp=[keypoints[0], points, camera_matrix], Tout=tf.float32)
        return tf.reshape(result, [3, 4])


@tf.function
def map_false_positive(points, valid_points_filter):
    if valid_points_filter == 0 and tf.reduce_sum(points) > 0:
        return 1.0
    else:
        return 0.0


def estimate_poses(points, keypoints, camera_matrixes, valid_points_filter, offsets):
    """
    :param points:             [b,oc,vn, 2]
    :param keypoints:          [b,oc,ic,vn,2]
    :param camera_matrixes:    [b,3,3]
    :param filter:             [b,oc]
    """

    b, oc, ic, vn, _ = keypoints.shape
    camera_matrixes = tf.broadcast_to(tf.expand_dims(camera_matrixes, 1), [b, oc, 3, 3])
    offsets = tf.broadcast_to(tf.expand_dims(offsets, 1), [b, oc, 10])

    points = tf.reshape(points, [-1, vn, 2])
    keypoints = tf.reshape(keypoints, [-1, ic, vn, 3])
    camera_matrixes = tf.reshape(camera_matrixes, [-1, 3, 3])
    offsets = tf.reshape(offsets, [-1, 10])

    valid_points_filter = tf.reshape(valid_points_filter, [-1])

    fn = lambda x: map_false_positive(x[0], x[1])
    elems = (points, valid_points_filter)

    false_positive = tf.map_fn(fn, elems=elems, dtype=tf.float32)
    false_positive = tf.reshape(false_positive, [b, oc, 1])
    false_positive = tf.cast(tf.reduce_sum(false_positive, axis=0), dtype=tf.float32)

    fn = lambda x: map_offsets(x[0], x[1], x[2])
    elems = (points, valid_points_filter, offsets)
    points = tf.map_fn(fn, elems=elems, dtype=tf.float32)
    fn = lambda x: map_pnp(x[0], x[1], x[2], x[3])
    elems = (points, keypoints, camera_matrixes, valid_points_filter)
    poses = tf.map_fn(fn, elems=elems, dtype=tf.float32)
    poses = tf.reshape(poses, [b, oc, 3, 4])
    return poses, tf.squeeze(false_positive)


@tf.function
def map_estimates(
    pose,
    pose_gt,
    points_estimated,
    object_points_3d,
    camera_matrix,
    diameter,
    valid_points_filter,
    object_points_3d_count,
    allowed_error_2d,
):
    # so far the instance count is ignored here and only the first instance of the target data is compared

    if valid_points_filter == 0:
        if tf.math.abs(tf.reduce_sum(pose)) > 0.0001:
            return tf.stack([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        return tf.stack([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if tf.math.abs(tf.reduce_sum(pose)) < 0.0001:
        return tf.stack([99.9, 999.9, 0.0, 0.0, 1.0, 0.0])  # object could not be found at all
    else:
        object_points_3d = object_points_3d[0][: object_points_3d_count[0]]
        points_2d_reproj, points_3d = project_tf(object_points_3d, camera_matrix, pose)

        points_2d_reproj = tf.reshape(points_2d_reproj, [tf.shape(object_points_3d)[0], 2])
        points_3d = tf.reshape(points_3d, tf.shape(object_points_3d))

        target_points_2d, target_points_3d = project_tf(object_points_3d, camera_matrix, pose_gt[0])

        target_points_2d = tf.reshape(target_points_2d, [tf.shape(object_points_3d)[0], 2])
        target_points_3d = tf.reshape(target_points_3d, tf.shape(object_points_3d))

        err_2d_detection = tf.reduce_mean(tf.norm(target_points_2d - points_2d_reproj, axis=1))

        # https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
        def adds_error(A, B):
            A = tf.cast(A, tf.float64)
            B = tf.cast(B, tf.float64)

            row_norms_A = tf.reduce_sum(tf.square(A), axis=1)

            row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

            row_norms_B = tf.reduce_sum(tf.square(B), axis=1)

            row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

            err = row_norms_A - 2 * tf.linalg.matmul(A, tf.transpose(B)) + row_norms_B

            return tf.cast(tf.sqrt(tf.abs(tf.reduce_min(err, axis=1)) + 1e-5), tf.float32)

        # def adds_error(target_points, estimated_points):
        #     G = tf.linalg.matmul(target_points, estimated_points, transpose_b=True)
        #     err = tf.linalg.tensor_diag_part(G) + tf.transpose(tf.linalg.tensor_diag_part(G)) - 2*G
        #     err = tf.boolean_mask(err, err > 0)
        #     return tf.math.sqrt(tf.reduce_min(err))

        if object_points_3d_count[0] == 7862 or object_points_3d_count[0] == 3417:  # glue and eggbox
            err_3d_detection = tf.reduce_mean(adds_error(target_points_3d, points_3d))
        else:
            err_3d_detection = tf.reduce_mean(tf.norm(target_points_3d - points_3d, axis=1))

        valid_3d = tf.cast(err_3d_detection < (diameter[0][0] * 0.1), tf.float32)
        valid_2d = tf.cast(err_2d_detection < allowed_error_2d, tf.float32)
        return tf.stack([err_2d_detection, err_3d_detection, valid_3d, valid_2d, 0.0, 0.0])


def evaluate_poses(
    poses,
    poses_gt,
    points_estimated,
    object_points_3d,
    object_points_3d_count,
    camera_matrixes,
    diameters,
    valid_points_filter,
    allowed_error_2d,
):
    b, oc, ic, vn, _ = object_points_3d.shape
    vn_pt = tf.shape(points_estimated)[2]

    # diameters = tf.broadcast_to(tf.expand_dims(diameters,1), [b,oc])
    camera_matrixes = tf.broadcast_to(tf.expand_dims(camera_matrixes, 1), [b, oc, 3, 3])
    poses = tf.reshape(poses, [-1, 3, 4])
    points_estimated = tf.reshape(points_estimated, [-1, vn_pt, 2])

    poses_gt = tf.reshape(poses_gt, [-1, ic, 3, 4])
    object_points_3d = tf.reshape(object_points_3d, [-1, ic, vn, 3])
    object_points_3d_count = tf.reshape(object_points_3d_count, [-1, ic])
    camera_matrixes = tf.reshape(camera_matrixes, [-1, 3, 3])
    diameters = tf.reshape(diameters, [-1, ic, 1])

    valid_points_count = tf.cast(tf.reduce_sum(valid_points_filter, axis=0), dtype=tf.float32)
    valid_points_filter = tf.reshape(valid_points_filter, [-1])

    fn = lambda x: map_estimates(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], allowed_error_2d)
    elems = (
        poses,
        poses_gt,
        points_estimated,
        object_points_3d,
        camera_matrixes,
        diameters,
        valid_points_filter,
        object_points_3d_count,
    )
    err_2d_3d = tf.map_fn(fn, elems=elems, dtype=tf.float32)
    err_2d_3d = tf.reshape(err_2d_3d, [b, oc, 6])

    err_2d = tf.cast(
        tf.reduce_sum(err_2d_3d[:, :, 0], axis=0), dtype=tf.float32
    )  # errors are added here and average is computed later
    err_3d = tf.cast(tf.reduce_sum(err_2d_3d[:, :, 1], axis=0), dtype=tf.float32)
    valid_3d = tf.cast(tf.reduce_sum(err_2d_3d[:, :, 2], axis=0), dtype=tf.float32)
    valid_2d = tf.cast(tf.reduce_sum(err_2d_3d[:, :, 3], axis=0), dtype=tf.float32)
    missing_object = tf.cast(tf.reduce_sum(err_2d_3d[:, :, 4], axis=0), dtype=tf.float32)
    false_positive_detection = tf.cast(tf.reduce_sum(err_2d_3d[:, :, 5], axis=0), dtype=tf.float32)

    return (
        err_2d,
        err_3d,
        valid_2d,
        valid_3d,
        missing_object,
        valid_points_count,
        false_positive_detection,
    )
