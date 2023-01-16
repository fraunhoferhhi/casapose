import tensorflow as tf


def add_noise(tensor, std=0.1):
    t = tf.random.normal(
        shape=tf.shape(input=tensor),
        mean=0.0,
        stddev=tf.random.uniform([1], 0, std),
        dtype=tf.float32,
    )
    t = tf.add(tensor, t)
    t = tf.clip_by_value(t, -1, 1)  # this is expensive
    return t


@tf.function
def compute_vertex_hcoords_batch_v3(mask, coords, use_motion=False):
    # hcoords is a list with all coordinates (o-yx) (float32)
    # mask shoud be 4d array (hwc) with one channel
    # choords [b, c, o, m, 2]

    b = tf.shape(mask)[0]  # there is a big difference between t.shape(x) and x.shape
    h = tf.shape(mask)[1]  # height
    w = tf.shape(mask)[2]  # width
    o = tf.shape(coords)[2]  # number of instances
    m = tf.shape(coords)[3]  # number of points
    coords = tf.concat([tf.zeros([b, 1, o, m, 2]), coords], axis=1)
    mask = tf.cast(tf.squeeze(mask, axis=-1), dtype=tf.int32)
    grid_coords = tf.expand_dims(
        tf.cast(
            tf.stack(tf.meshgrid(tf.range(w), tf.range(h))[::-1], axis=-1),
            dtype=tf.float32,
        ),
        0,
    )  # [1, h, w, 2]
    grid_coords += 0.5
    closest_center = tf.zeros([b, h, w, 1], dtype=tf.int32)
    if o > 1:
        centers = tf.squeeze(coords[:, :, :, 0:1, :], 3)  # [b,c, o, 2]
        grid_coords_obj = tf.expand_dims(grid_coords, 3)  # [b, h, w, 1, 2]

        grid_dist = tf.linalg.norm(
            grid_coords_obj - tf.gather_nd(centers, tf.expand_dims(mask, -1), batch_dims=1),
            axis=-1,
        )
        closest_center = tf.expand_dims(
            tf.where(mask == 0, 0, tf.argmin(grid_dist, axis=-1, output_type=tf.int32)),
            -1,
        )

    coords_on_mask = tf.gather(coords, mask, batch_dims=1)

    dirs = tf.expand_dims(grid_coords, 3)  # [1, h, w, 1, 2]
    dirs = tf.repeat(tf.expand_dims(grid_coords, 3), m, axis=3)  # [b, h, w, c, 2]
    if o > 1:
        dirs = tf.gather_nd(coords_on_mask, closest_center, batch_dims=3) - dirs
    else:
        dirs = tf.squeeze(coords_on_mask, 3) - dirs
    dirs = tf.multiply(dirs, tf.cast(tf.expand_dims(tf.expand_dims(mask != 0, -1), -1), tf.float32))
    if not use_motion:
        dirs = tf.math.l2_normalize(dirs, axis=-1)  # really slow

    return tf.reshape(dirs, [b, h, w, m * 2])


def get_all_vectorfields(target_seg, target_vertex, filtered_seg, separated_vectorfields):

    if separated_vectorfields:
        target_directions = compute_vertex_hcoords_batch_v3(target_seg[:, :, :, 1:2], target_vertex[:, 0:1, :, :, :])
        for idx in range(1, target_seg.shape[3] - 1):  # iterate over objects
            direction_map = compute_vertex_hcoords_batch_v3(
                target_seg[:, :, :, idx + 1 : idx + 2],
                target_vertex[:, idx : idx + 1, :, :, :],
            )  # instance axis is squeezed!
            target_directions = tf.concat([target_directions, direction_map], 3)
    else:
        target_directions = compute_vertex_hcoords_batch_v3(filtered_seg, target_vertex)

    return target_directions
