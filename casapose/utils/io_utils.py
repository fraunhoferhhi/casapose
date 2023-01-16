import os
import re

import numpy as np
import tensorflow as tf


# from: https://stackoverflow.com/questions/10097477/python-json-array-newlines
def to_json(o, level=0, INDENT=4, SPACE=" ", NEWLINE="\n"):
    ret = ""
    if isinstance(o, dict):
        if level != 0:
            ret += NEWLINE + SPACE * INDENT * level
        ret += "{" + NEWLINE
        comma = ""
        for k, v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level + 1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o + '"'
    elif isinstance(o, list):
        is_list = False
        for e in o:
            is_list = isinstance(e, list)
        if is_list is False:
            ret += "[ " + ", ".join([to_json(e, level + 1) for e in o]) + " ]"
        else:
            ret += "[ " + NEWLINE + SPACE * INDENT * (level + 1)
            separator = ", " + NEWLINE + SPACE * INDENT * (level + 1)
            ret += separator.join([to_json(e, level + 1) for e in o])
            ret += NEWLINE + SPACE * INDENT * (level) + " ]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += "%.16g" % o
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
        ret += "[" + ",".join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
        ret += "[" + ",".join(map(lambda x: "%.16g" % x, o.flatten().tolist())) + "]"
    elif o is None:
        ret += "null"
    else:
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    return ret


def write_poses(gt_poses, estimated_poses, names, image_id, path_out, time_needed=None):
    """
    Write output poses to files.
    Stores results in bop challenge format as 'bop_evaluation.csv'
    """
    gt_poses = tf.squeeze(gt_poses, -3)
    m = re.findall(r"\d*\.*\d+", image_id[0, 0].numpy().decode("utf-8"))
    scene_id = int(m[0])
    img_id = int(m[1])

    if time_needed is None:
        time = -1.0
    else:
        time = float(time_needed.numpy())

    if not os.path.exists(path_out):
        os.mkdir(path_out)

    path_out_fp = path_out + "all_poses/"
    path_out_filtered = path_out + "filtered_poses/"

    if not os.path.exists(path_out_fp):
        os.mkdir(path_out_fp)

    if not os.path.exists(path_out_filtered):
        os.mkdir(path_out_filtered)

    def print_pose(path, pose):
        exists = os.path.isfile(path)

        with open(path, "a") as file:
            if not exists:
                file.write("#r11 r12 r13 r21 r22 r23 r31 r32 r33 tx ty tz\n")
            R = tf.reshape(pose[:, :3], [-1]).numpy()
            t = tf.reshape(pose[:, 3], [-1]).numpy()
            p_str = " ".join(map(str, R)) + " " + " ".join(map(str, t)) + "\n"
            file.write(p_str)

    def print_pose_bop(path, pose, scene_id, img_id, obj_id, time):
        exists = os.path.isfile(path)
        confidence = float(tf.math.abs(tf.reduce_sum(pose)).numpy())
        if confidence > 0:
            confidence = 1.0
        else:
            confidence = 0.0
        with open(path, "a") as file:
            if not exists:
                file.write("scene_id,im_id,obj_id,score,R,t,time\n")
            R = tf.reshape(pose[:, :3], [-1]).numpy()
            t = tf.reshape(pose[:, 3], [-1]).numpy()
            p_str = (
                str(scene_id)
                + ","
                + str(img_id)
                + ","
                + str(obj_id)
                + ","
                + str(float(confidence))
                + ","
                + " ".join(map(str, R))
                + ","
                + " ".join(map(str, t))
                + ","
                + str(time)
                + "\n"
            )
            file.write(p_str)

    for idx, name in enumerate(names):
        obj_id = int(re.findall(r"\d*\.*\d+", name)[0])
        if tf.math.abs(tf.reduce_sum(gt_poses[idx])) > 0.0001:
            print_pose_bop(
                path_out + "bop_evaluation.csv",
                estimated_poses[idx],
                scene_id,
                img_id,
                obj_id,
                time,
            )
            print_pose(path_out_filtered + "poses_gt_" + name + ".txt", gt_poses[idx])
            print_pose(path_out_filtered + "poses_init_" + name + ".txt", estimated_poses[idx])
        else:
            print_pose(path_out_filtered + "poses_gt_" + name + ".txt", tf.zeros([3, 4]))
            print_pose(path_out_filtered + "poses_init_" + name + ".txt", tf.zeros([3, 4]))
        print_pose(path_out_fp + "poses_init_" + name + ".txt", estimated_poses[idx])
