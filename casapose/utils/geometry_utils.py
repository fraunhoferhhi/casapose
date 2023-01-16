import math

import numpy as np
import tensorflow as tf


def reproject(points, tm, rm, offset):
    proj_cuboid = np.array(points)

    rmat = np.identity(3)
    rmat[0:2] = rm
    tmat = np.identity(3)
    tmat[0:2] = tm

    new_cuboid = np.matmul(rmat, np.vstack((proj_cuboid.T, np.ones(len(points)))))
    new_cuboid = np.matmul(tmat, new_cuboid)
    new_cuboid = new_cuboid[0:2].T
    new_cuboid -= offset
    return new_cuboid


def apply_offsets(points, offsets):
    w_crop = offsets[0]
    h_crop = offsets[1]
    sx = offsets[8]
    sy = offsets[9]
    dx = offsets[4]
    dy = offsets[5]
    angle = offsets[6]
    scale = offsets[7]
    tm = np.float32([[1, 0, dx], [0, 1, dy]])
    rm = get_rotation_matrix_2D((sx / 2, sy / 2), angle)

    return reproject(points, tm, rm, [w_crop, h_crop]) * scale


def get_rotation_matrix_2D(center, angle):
    angle_rad = angle * (math.pi / 180)
    a = np.cos(angle_rad)
    b = np.sin(angle_rad)
    c = (1 - a) * center[0] - b * center[1]
    d = b * center[0] + (1 - a) * center[1]

    rot_mat = np.float32([[a, b, c], [-b, a, d]])
    return rot_mat


def transform_points(points, transform):
    p = np.array(points, dtype=points.dtype, copy=True)
    n = len(points)
    p = np.transpose(np.c_[p, np.ones(n)])
    # fixed_transform = np.matmul(fixed_transform, np.diag([1, 1, 1, 1]))
    # print(fixed_transform)
    p = np.transpose(np.matmul(transform, p))
    # print(p[:, 0:3])

    return p[:, 0:3]


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz_proj = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz_proj, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy, xyz_proj


# RETURNS WXYZ in previouse Versions (08.12. changed to XYZW)
def matrix_to_quaternion(M):
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    K = (
        np.array(
            [
                [Qxx - Qyy - Qzz, 0, 0, 0],
                [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
                [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
                [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz],
            ]
        )
        / 3.0
    )
    vals, vecs = np.linalg.eigh(K)
    # q = vecs[[3, 0, 1, 2], np.argmax(vals)]

    # if q[0] < 0:
    #    q *= -1

    q = vecs[[0, 1, 2, 3], np.argmax(vals)]

    if q[3] < 0:
        q *= -1

    return q


def get_horizontal_width_angle(width, height, fx, fy):
    aspect = width / fx * (fy / height)
    return np.rad2deg(2.0 * np.arctan(aspect * (0.5 / (fy / height))))


def create_transformation_matrix(R, t):
    return np.array(
        [
            [R[0][0], R[0][1], R[0][2], t[0]],
            [R[1][0], R[1][1], R[1][2], t[1]],
            [R[2][0], R[2][1], R[2][2], t[2]],
            [0, 0, 0, 1],
        ]
    )


# Source: https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
# Copyright (c) 2006, Christoph Gohlke
# Copyright (c) 2006-2010, The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
def quaternion_matrix(quaternion_xyzw, translation=None, wxyz_input=False):
    """Return homogeneous rotation matrix from quaternion.
    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True
    """

    q = np.array(quaternion_xyzw, dtype=np.float64, copy=True)
    if wxyz_input is False:
        q = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    n = np.dot(q, q)
    if n < 0.0001:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    if translation is None:
        return np.array(
            [
                [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
            ]
        )
    else:
        t = np.array(translation, dtype=np.float64, copy=True)
        return np.array(
            [
                [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], t[0]],
                [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], t[1]],
                [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], t[2]],
            ]
        )


# Source: https://github.com/blzq/tf_rodrigues/blob/master/rodrigues.py
# MIT License

# Copyright (c) 2018 blzq

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def rodrigues_batch(rvecs):
    """
    Convert a batch of axis-angle rotations in rotation vector form shaped
    (batch, 3) to a batch of rotation matrices shaped (batch, 3, 3).
    See
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    """
    batch_size = tf.shape(rvecs)[0]
    tf.assert_equal(tf.shape(rvecs)[1], 3)
    thetas = tf.norm(rvecs, axis=1, keepdims=True)
    is_zero = tf.equal(tf.squeeze(thetas), 0.0)
    u = rvecs / thetas
    # Each K is the cross product matrix of unit axis vectors
    zero = tf.zeros([batch_size])  # for broadcasting
    Ks_1 = tf.stack([zero, -u[:, 2], u[:, 1]], axis=1)  # row 1
    Ks_2 = tf.stack([u[:, 2], zero, -u[:, 0]], axis=1)  # row 2
    Ks_3 = tf.stack([-u[:, 1], u[:, 0], zero], axis=1)  # row 3

    Ks = tf.stack([Ks_1, Ks_2, Ks_3], axis=1)  # stack rows
    Rs = (
        tf.eye(3, batch_shape=[batch_size])
        + tf.sin(thetas)[..., tf.newaxis] * Ks
        + (1 - tf.cos(thetas)[..., tf.newaxis]) * tf.matmul(Ks, Ks)
    )
    # Avoid returning NaNs where division by zero happened
    return tf.where(
        tf.expand_dims(tf.expand_dims(is_zero, -1), -1),
        tf.eye(3, batch_shape=[batch_size]),
        Rs,
    )
