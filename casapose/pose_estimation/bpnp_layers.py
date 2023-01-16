import cv2
import numpy as np
import tensorflow as tf
from casapose.utils.geometry_utils import rodrigues_batch

"""
Based on Code for Paper "End-to-End Learnable Geometric Vision by Backpropagating PnP Optimization" by Bo Chen et al. (https://github.com/BoChenYS/BPnP).
"""


def batch_project(P, pts3d, K, angle_axis=True):
    # pts3d [n, 3], P [n, 6]
    n = tf.shape(pts3d)[0]  # tf.shape(pts3d)[0]

    pts3d_h = tf.concat([pts3d, tf.expand_dims(tf.ones([n]), -1)], axis=-1)
    if angle_axis:
        R_out = rodrigues_batch(P[:, 0:3])  # [bs, 3,3]
        PM = tf.concat([R_out, tf.expand_dims(P[:, 3:6], -1)], axis=-1)
    else:
        PM = P
    # pts3d_cam = tf.matmul(tf.expand_dims(tf.expand_dims(pts3d_h, 1),0), tf.expand_dims(tf.transpose(PM, [0,2,1]),1))
    # pts2d_proj = tf.squeeze(tf.matmul(pts3d_cam, tf.expand_dims(K, 0)))
    pts3d_cam = tf.matmul(tf.expand_dims(pts3d_h, 0), tf.transpose(PM, [0, 2, 1]))
    pts2d_proj = tf.matmul(pts3d_cam, tf.expand_dims(tf.transpose(K), 0))
    pts2d = pts2d_proj[:, :, 0:2] / pts2d_proj[:, :, 2:]
    return pts2d


def batch_project_(P, pts3d, K, angle_axis=True):
    # pts3d [bs, n, 3], P [bs, n, 6], K [bs, 3,3]
    bs = tf.shape(P)[0]  # self.num_points #?
    bs_P = tf.shape(pts3d)[0]
    n = tf.shape(pts3d)[1]

    P = tf.reshape(P, [bs * n, 6])
    n_ = tf.tile(tf.expand_dims(tf.expand_dims(tf.ones([n]), -1), 0), [bs_P, 1, 1])
    pts3d_h = tf.concat([pts3d, n_], axis=-1)
    if angle_axis:
        R_out = rodrigues_batch(P[:, 0:3])  # [bs, 3,3]
        PM = tf.concat([R_out, tf.expand_dims(P[:, 3:6], -1)], axis=-1)
    else:
        PM = P
    PM = tf.reshape(PM, [bs, n, 3, 4])
    # PM = tf.tile(tf.expand_dims(PM, 1), [1, n, 1, 1])
    # pts3d_cam = tf.matmul(tf.expand_dims(tf.expand_dims(pts3d_h, 1),0), tf.expand_dims(tf.transpose(PM, [0,2,1]),1))
    # pts2d_proj = tf.squeeze(tf.matmul(pts3d_cam, tf.expand_dims(K, 0)))
    pts3d_cam = tf.matmul(tf.expand_dims(pts3d_h, 1), tf.transpose(PM, [0, 1, 3, 2]))
    pts2d_proj = tf.matmul(pts3d_cam, tf.expand_dims(tf.transpose(K, perm=[0, 2, 1]), 1))
    pts2d = pts2d_proj[:, :, :, 0:2] / pts2d_proj[:, :, :, 2:]
    return pts2d


def get_coefs(P_6d, pts3d, K):
    # P_6d [1, 6], pts3d[n,3], K
    n = tf.shape(pts3d)[0]
    y = tf.tile(P_6d, [n, 1])  # [n, 6]
    vec = tf.eye(n)
    with tf.GradientTape(persistent=True) as g:
        g.watch(y)
        proj = tf.squeeze(batch_project(y, pts3d, K))
        proj_0 = proj[:, :, 0]
        proj_1 = proj[:, :, 1]
    coefs = tf.stack(
        [-2 * g.gradient(proj_0, y, [vec]), -2 * g.gradient(proj_1, y, [vec])], axis=1
    )  # no idea if vec belongs here...
    return coefs


def get_coefs_batch(P_6d, pts3d, K):
    # P_6d [bs, 6], pts3d[bs,n,3], K
    bs = tf.shape(P_6d)[0]
    n = tf.shape(pts3d)[1]
    y = tf.tile(tf.expand_dims(P_6d, 1), [1, n, 1])  # [bs, n, 6]
    vec = tf.tile(tf.expand_dims(tf.eye(n), 0), [bs, 1, 1])
    with tf.GradientTape(persistent=True) as g:
        g.watch(y)
        proj = batch_project_(y, pts3d, K)
        proj_0 = proj[:, :, :, 0]
        proj_1 = proj[:, :, :, 1]
    coefs = tf.stack(
        [-2 * g.gradient(proj_0, y, [vec]), -2 * g.gradient(proj_1, y, [vec])], axis=1
    )  # no idea if vec belongs here...
    return coefs


def pnp(points_3d, points_2d, camera_matrix, init_pose=None):
    assert points_3d.shape[0] == points_2d.shape[0], "points 3D and points 2D must have same number of vertices"

    points_2d = np.expand_dims(points_2d.astype(np.float32), 1)
    points_3d = points_3d.astype(np.float32)
    camera_matrix = camera_matrix.astype(np.float32)

    if init_pose is None:
        _, rvec0, T0, _ = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_EPNP,
            confidence=0.9999,
            reprojectionError=12,
        )
    else:
        rvec0 = np.array(init_pose[0:3]).reshape([3, 1]).astype(np.float32)
        T0 = np.array(init_pose[3:6]).reshape([3, 1]).astype(np.float32)
    _, r, t = cv2.solvePnP(
        points_3d,
        points_2d,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
        useExtrinsicGuess=True,
        rvec=rvec0,
        tvec=T0,
    )

    return np.concatenate([r, t], axis=0).astype(np.float32)


@tf.function
def map_pnp(points, keypoints, camera_matrix, init_pose=None):  # keypoints incude instance ?
    if init_pose is None:
        result = tf.numpy_function(pnp, inp=[keypoints, points, camera_matrix], Tout=tf.float32)
    else:
        result = tf.numpy_function(pnp, inp=[keypoints, points, camera_matrix, init_pose], Tout=tf.float32)
    return tf.reshape(result, [6])


def batch_pnp(points, keypoints, camera_matrix, init_pose=None):  # keypoints incude instance ?
    # start = timer()
    out = np.zeros([len(points), 6], dtype=np.float32)
    for i, pts in enumerate(points):
        out[i] = np.squeeze(pnp(keypoints[i], pts, camera_matrix))
    # print(timer() - start)
    return out


@tf.function
def pnp_gradient_efficient(grad_output, pts2d, pts3d, K, P_6d, init_pose=False, batch_3d=False, batch_cam=False):
    bs = tf.shape(pts2d)[0]
    n = tf.shape(pts2d)[1]  # self.num_points

    # batch_3d = tf.equal(tf.shape(pts3d).shape[0], tf.constant(3))
    if batch_3d:
        tf.debugging.assert_equal(bs, tf.shape(pts3d)[0])
    else:
        pts3d = tf.tile(tf.expand_dims(pts3d, 0), [bs, 1, 1])

    if batch_cam:
        tf.debugging.assert_equal(bs, tf.shape(K)[0])
    else:
        K = tf.tile(tf.expand_dims(K, 0), [bs, 1, 1])

    coefs = get_coefs_batch(P_6d, pts3d, K)
    pts2d_flat = tf.reshape(pts2d, [bs, -1])
    P_6d_flat = tf.reshape(P_6d, [bs, -1])
    pts3d_flat = tf.reshape(pts3d, [bs, -1])
    K_flat = tf.reshape(K, [bs, -1])
    with tf.GradientTape(persistent=True) as g:
        g.watch(pts2d_flat)
        g.watch(P_6d_flat)
        g.watch(pts3d_flat)
        g.watch(K_flat)
        R = rodrigues_batch(P_6d_flat[:, 0:3])
        P = tf.concat([R, tf.expand_dims(P_6d_flat[:, 3:6], -1)], axis=-1)
        KP = tf.matmul(tf.reshape(K_flat, [bs, 3, 3]), P)
        pts2d_i = tf.transpose(tf.reshape(pts2d_flat, [bs, n, 2]), perm=[0, 2, 1])
        pts3d_i = tf.transpose(
            tf.concat(
                [
                    tf.reshape(pts3d_flat, [bs, n, 3]),
                    tf.tile(tf.expand_dims(tf.expand_dims(tf.ones([n]), 0), 2), [bs, 1, 1]),
                ],
                axis=-1,
            ),
            perm=[0, 2, 1],
        )
        proj_i = tf.matmul(KP, pts3d_i)  # [bs, 3, m]
        Si = proj_i[:, 2:, :]
        r = pts2d_i * Si - proj_i[:, 0:2, :]
        fj = tf.reduce_sum(tf.reshape(coefs * tf.expand_dims(r, -1), [bs, 2 * n, 6]), axis=1)
    J_fx = g.batch_jacobian(fj, pts2d_flat)
    J_fy = g.batch_jacobian(fj, P_6d_flat)
    J_fz = g.batch_jacobian(fj, pts3d_flat)
    J_fK = g.batch_jacobian(fj, K_flat)
    # J_fy =  J_fy + (tf.expand_dims(tf.eye(6),0) * 10e-5) might be needed to make matrix invertible
    inv_J_fy = tf.linalg.pinv(J_fy)  # use pinv instead of inv to avoid creshes
    J_yx = (-1) * tf.matmul(inv_J_fy, J_fx)
    J_yz = (-1) * tf.matmul(inv_J_fy, J_fz)
    J_yK = (-1) * tf.matmul(inv_J_fy, J_fK)
    grad_x = tf.matmul(tf.expand_dims(grad_output, 1), J_yx)
    grad_z = tf.matmul(tf.expand_dims(grad_output, 1), J_yz)
    grad_K = tf.matmul(tf.expand_dims(grad_output, 1), J_yK)

    if not batch_3d:
        grad_z = tf.reduce_sum(grad_z, axis=0)
        grad_z = tf.reshape(grad_z, [n, 3])
    else:
        grad_z = tf.reshape(grad_z, [bs, n, 3])

    if not batch_cam:
        grad_K = tf.reduce_sum(grad_K, axis=0)
        grad_K = tf.reshape(grad_K, [3, 3])
    else:
        grad_K = tf.reshape(grad_K, [bs, 3, 3])

    grad_x = tf.reshape(grad_x, [bs, n, 2])

    if init_pose:
        return (grad_x, grad_z, grad_K)
    else:
        return (grad_x, grad_z, grad_K, None)


def pnp_gradient(grad_output, pts2d, pts3d, K, P_6d, init_pose=False):
    bs = tf.shape(pts2d)[0]
    n = tf.shape(pts2d)[1]  # self.num_points
    m = 6
    grad_x = tf.reshape(tf.constant([], tf.float32), [0, n * 2])
    grad_z = tf.zeros([n * 3])
    grad_K = tf.zeros_like(K)
    for i in range(bs):
        J_fy = tf.reshape(tf.constant([], tf.float32), [0, m])
        J_fx = tf.reshape(tf.constant([], tf.float32), [0, 2 * n])
        J_fz = tf.reshape(tf.constant([], tf.float32), [0, 3 * n])
        J_fK = tf.reshape(tf.constant([], tf.float32), [0, 9])
        coefs = get_coefs(tf.reshape(P_6d[i], [1, 6]), pts3d, K)
        pts2d_flat = tf.reshape(pts2d[i], [-1])
        P_6d_flat = tf.reshape(P_6d[i], [-1])
        pts3d_flat = tf.reshape(pts3d, [-1])
        K_flat = tf.reshape(K, [-1])
        for j in range(m):
            with tf.GradientTape(persistent=True) as g:
                g.watch(pts2d_flat)
                g.watch(P_6d_flat)
                g.watch(pts3d_flat)
                g.watch(K_flat)
                R = tf.squeeze(rodrigues_batch(tf.expand_dims(P_6d_flat[0 : m - 3], 0)))
                P = tf.concat([R, tf.expand_dims(P_6d_flat[3:6], -1)], axis=-1)
                KP = tf.matmul(tf.reshape(K_flat, [3, 3]), P)
                pts2d_i = tf.transpose(tf.reshape(pts2d_flat, [n, 2]))
                pts3d_i = tf.transpose(
                    tf.concat(
                        [
                            tf.reshape(pts3d_flat, [n, 3]),
                            tf.expand_dims(tf.ones([n]), -1),
                        ],
                        axis=-1,
                    )
                )
                proj_i = tf.matmul(KP, pts3d_i)
                Si = proj_i[2, :]
                r = pts2d_i * Si - proj_i[0:2, :]
                coef = tf.transpose(coefs[:, :, j])  # size: [2,n]
                fj = tf.reduce_sum(coef * r)
            grads = g.gradient(fj, [pts2d_flat, P_6d_flat, pts3d_flat, K_flat])
            J_fx = tf.concat([J_fx, tf.expand_dims(grads[0], 0)], axis=0)
            J_fy = tf.concat([J_fy, tf.expand_dims(grads[1], 0)], axis=0)
            J_fz = tf.concat([J_fz, tf.expand_dims(grads[2], 0)], axis=0)
            J_fK = tf.concat([J_fK, tf.expand_dims(grads[3], 0)], axis=0)
        inv_J_fy = tf.linalg.inv(J_fy)
        J_yx = (-1) * tf.matmul(inv_J_fy, J_fx)
        J_yz = (-1) * tf.matmul(inv_J_fy, J_fz)
        J_yK = (-1) * tf.matmul(inv_J_fy, J_fK)

        grad_x = tf.concat([grad_x, tf.matmul(tf.expand_dims(grad_output[i], 0), J_yx)], axis=0)
        grad_z += tf.matmul(tf.expand_dims(grad_output[i], 0), J_yz)
        grad_K += tf.reshape(tf.matmul(tf.expand_dims(grad_output[i], 0), J_yK), [3, 3])
    # grad_z = grad_z[1:,:]
    grad_x = tf.reshape(grad_x, [bs, n, 2])
    grad_z = tf.reshape(grad_z, [n, 3])
    if init_pose:
        return (grad_x, grad_z, grad_K)
    else:
        return (grad_x, grad_z, grad_K, None)


class BPNP_fast(tf.keras.layers.Layer):
    def __init__(self, name):
        super(BPNP_fast, self).__init__(name=name)
        self.number_of_parallel_calls = 8
        # self.pool = multiprocessing.Pool(self.number_of_parallel_calls) this crashes everything
        # self.num_classes = num_classe

    def build(self, input_shape):
        self.dims_2d = len(input_shape[0])
        self.dims_3d = len(input_shape[1])
        self.shape_2d = input_shape[0]
        tf.print(input_shape)

    #   pts2d - the 2D keypoints coordinates of size [batch_size, num_keypoints, 2]
    #   pts3d - the 3D keypoints coordinates of size [num_keypoints, 3] or [batch_size, num_keypoints, 3]
    #   K     - the camera intrinsic matrix of size [3, 3]
    #   init_pose - initial pose [batch_size, 3]
    @tf.custom_gradient
    def bpnp_func(self, pts2d, pts3d, K, init_pose):
        batch_3d = self.dims_3d > 2
        if batch_3d:
            P_6d = tf.map_fn(
                lambda x: map_pnp(x[0], x[1], K, x[2]),
                elems=(pts2d, pts3d, init_pose),
                dtype=tf.float32,
            )
        else:
            P_6d = tf.map_fn(
                lambda x: map_pnp(x[0], pts3d, K, x[1]),
                elems=(pts2d, init_pose),
                dtype=tf.float32,
            )

        def custom_grad(
            grad_output,
        ):  # grad_output /upstream is the current gradient https://www.tensorflow.org/api_docs/python/tf/custom_gradient
            return pnp_gradient_efficient(grad_output, pts2d, pts3d, K, P_6d, batch_3d=batch_3d)

        return P_6d, custom_grad

    @tf.custom_gradient
    def bpnp_func_init(self, pts2d, pts3d, K):
        batch_3d = self.dims_3d > 2
        if batch_3d:
            P_6d = tf.numpy_function(
                batch_pnp, inp=[pts2d, pts3d, K], Tout=tf.float32
            )  # seemingly this is faster than calling numpy_func for every batch
            # P_6d = tf.map_fn(lambda x: map_pnp(x[0], x[1], K), elems=(pts2d, pts3d), dtype= tf.float32)
        else:
            P_6d = tf.map_fn(lambda x: map_pnp(x, pts3d, K), elems=(pts2d), dtype=tf.float32)

        def custom_grad(grad_output):
            return pnp_gradient_efficient(grad_output, pts2d, pts3d, K, P_6d, init_pose=True, batch_3d=batch_3d)

        return P_6d, custom_grad

    def call(self, inputs, **kwargs):
        use_init_pose = len(inputs) == 4
        flatten_2d = self.dims_2d == 4
        flatten_3d = self.dims_3d == 4

        if use_init_pose:
            pts2d, pts3d, K, init_pose = inputs
        else:
            pts2d, pts3d, K = inputs

        if flatten_2d:
            pts2d = tf.reshape(pts2d, [-1, self.shape_2d[2], 2])
        if flatten_3d:
            pts3d = tf.reshape(pts3d, [-1, self.shape_2d[2], 3])

        if use_init_pose:
            if flatten_2d:
                init_pose = tf.reshape(init_pose, [-1, 6])
            output = self.bpnp_func(pts2d, pts3d, K, init_pose)
        else:
            output = self.bpnp_func_init(pts2d, pts3d, K)

        if flatten_2d:
            output = tf.reshape(output, [-1, self.shape_2d[1], 6])

        return output
