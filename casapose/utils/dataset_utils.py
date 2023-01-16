import glob
import json
import os
from os.path import exists

import cv2
import matplotlib as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from .draw_utils import (
    draw_bb,
    draw_lines,
    draw_points,
    grayscale_dist,
    pseudocolor_dir,
)
from .geometry_utils import apply_offsets, project
from .image_utils import get_all_vectorfields
from .io_utils import to_json


# dataset if tf.dataset
def save_batches(
    dataset,
    path,
    separated_vectorfields,
    no_objects,
    no_features,
    normal=[0.5, 0.5],
    no_batches=1,
):
    dataset_out = dataset.take(no_batches)
    if not exists(path):
        os.mkdir(path)

    for idx_batch, batch in enumerate(dataset_out):
        target_seg = batch[1]
        target_directions = get_all_vectorfields(target_seg, batch[3], batch[8], separated_vectorfields)

        for idx, img in enumerate(batch[0]):
            file_prefix = "batch_{}_{}".format(idx_batch, idx)
            save_single_sample(
                img,
                target_seg[idx],
                target_directions[idx],
                path,
                file_prefix,
                no_objects,
                no_features,
                normal,
            )


def save_single_sample(
    img,
    mask,
    dirs,
    path,
    file_prefix,
    no_objects,
    no_features,
    normal=[0.5, 0.5],
    locations=None,
    confidence=None,
    file_postfix="",
):

    # merge direction maps in case the original pvnet is used
    if no_objects > 1 and dirs.shape[-1] == no_objects * no_features * 2:
        dirs = tf.reshape(dirs, [dirs.shape[0], dirs.shape[1], no_objects, no_features, 2])
        argmax_segmentation = tf.math.argmax(mask, axis=2)
        dirs = tf.gather(dirs, tf.nn.relu(argmax_segmentation - 1), batch_dims=2)
        dirs = tf.where(tf.expand_dims(tf.expand_dims(argmax_segmentation == 0, -1), -1), 0.0, dirs)
        dirs = tf.reshape(dirs, [dirs.shape[0], dirs.shape[1], no_features * 2])

    if not exists(path):
        os.mkdir(path)
    img = tf.cast(((img * normal[1]) + normal[0]) * 255, dtype=tf.uint8).numpy()
    img_out = Image.fromarray(img)
    img_out.save(path + "/" + file_prefix + "color.png")

    mask = mask.numpy()
    mask = np.argmax(mask, axis=2)
    mask = mask * 255 / (no_objects + 1)

    if confidence is not None:
        confidence = tf.squeeze(confidence, axis=0)  # input confidence is already normalized softplus
        confidence = confidence.numpy()
        confidence = np.transpose(confidence, (2, 0, 1))

    dirs = dirs.numpy()
    dirs = np.transpose(dirs, (2, 0, 1))

    for idx_dir in range(no_features):
        dir_test = pseudocolor_dir(dirs[idx_dir * 2], dirs[idx_dir * 2 + 1], mask.astype("uint8"))

        # if not locations is None:
        #     for idx_obj in range(no_objects):
        #         pt = locations[idx_obj][0][idx_dir]
        #         draw_points(np.array([pt]), dir_test, (255, 255, 255), 4, 2, line_type=cv2.LINE_AA)

        dir_img = Image.fromarray((dir_test).astype("uint8"))
        dir_img.save(path + "/" + file_prefix + "color_dir_{}".format(idx_dir) + file_postfix + ".png")
        if confidence is not None:
            conf = confidence[idx_dir] * 255
            conf_img = Image.fromarray((conf).astype("uint8"))
            conf_img.save(path + "/" + file_prefix + "conf_dir_{}".format(idx_dir) + file_postfix + ".png")

    mask_cpy = mask
    mask /= 255
    ones = np.full(mask.shape, 1.0, dtype="float")
    mask = np.stack((mask, ones, ones), axis=2)
    mask = plt.colors.hsv_to_rgb(mask) * 255.0
    mask_cpy = mask_cpy[..., np.newaxis]
    mask = np.where(mask_cpy == 0, img, mask)
    mask_out = Image.fromarray(mask.astype("uint8"))
    mask_out.save(path + "/" + file_prefix + "mask" + file_postfix + ".png")


def save_single_mask(mask, path, no_objects, file_prefix="", file_postfix=""):

    if not exists(path):
        os.mkdir(path)

    mask = mask.numpy()
    mask = np.argmax(mask, axis=2)
    mask = mask * 255 / (no_objects + 1)

    mask_cpy = mask
    mask /= 255
    ones = np.full(mask.shape, 1.0, dtype="float")
    mask = np.stack((mask, ones, ones), axis=2)
    mask = plt.colors.hsv_to_rgb(mask) * 255.0
    mask_cpy = mask_cpy[..., np.newaxis]
    mask = np.where(mask_cpy == 0, 0, mask)
    mask_out = Image.fromarray(mask.astype("uint8"))
    mask_out.save(path + "/" + file_prefix + "mask" + file_postfix + ".png")


def save_clamped_grayscale_single_sample(dist, mask, path, file_prefix, no_objects, no_features, clip_max=15.0):

    if not exists(path):
        os.mkdir(path)

    mask = mask.numpy()
    mask = np.argmax(mask, axis=2)
    mask = mask * 255 / (no_objects + 1)

    dist = dist.numpy()
    dist = np.transpose(dist, (2, 0, 1))
    dist = tf.clip_by_value(dist, 0, clip_max)

    for idx_dir in range(no_features):
        dist_test = grayscale_dist(dist[idx_dir], mask.astype("uint8"), clip_max)
        dist_test = Image.fromarray((dist_test).astype("uint8"))
        dist_test.save(path + "/" + file_prefix + "proxy_error_{}".format(idx_dir) + ".png")


def save_mask_by_loss_value_single_sample(proxy_voting_loss, mask, path, file_prefix, threshold=5.0):
    if not exists(path):
        os.mkdir(path)
    h, w, _ = mask.shape
    # mask = mask.numpy()
    mask_argmax = tf.argmax(mask, axis=2)
    mask_out = tf.zeros([h, w])
    critical_objects = tf.where(proxy_voting_loss > threshold) + 1
    ok_objects = tf.where(proxy_voting_loss <= threshold) + 1
    for obj_idx in critical_objects:
        mask_out = tf.where(mask_argmax == obj_idx, 125, mask_out)
    for obj_idx in ok_objects:
        mask_out = tf.where(mask_argmax == obj_idx, 255, mask_out)
    mask_out = mask_out.numpy()
    mask_out = Image.fromarray((mask_out).astype("uint8"))
    mask_out.save(path + "/" + file_prefix + "proxy_summary.png")


def save_poses_single_sample(
    img,
    estimated_poses,
    estimated_points,
    cuboids,
    keypoints,
    camera_matrix,
    path,
    file_prefix,
    normal=[0.5, 0.5],
):
    if not exists(path):
        os.mkdir(path)

    img_keypoints = tf.cast(((img * normal[1]) + normal[0]) * 255, dtype=tf.uint8).numpy()

    eps = 1e-4
    img_cuboids = img_keypoints.copy()
    tf.print(estimated_poses)
    for obj_idx, obj_pose in enumerate(estimated_poses):
        inst_idx = 0
        obj_pose_est = estimated_poses[obj_idx].numpy()
        instance_cuboids = cuboids[obj_idx][inst_idx].numpy()
        instance_keypoints = keypoints[obj_idx][inst_idx].numpy()
        valid_est = np.abs(np.sum(obj_pose_est)) > eps

        if valid_est:
            transformed_cuboid_points2d, _ = project(instance_cuboids, camera_matrix.numpy(), obj_pose_est)
            transformed_keypoints_points2d, _ = project(instance_keypoints, camera_matrix.numpy(), obj_pose_est)
            valid_est = np.abs(np.sum(transformed_keypoints_points2d)) > eps
            draw_bb(transformed_cuboid_points2d, img_cuboids, (255, 0, 0))

    img_cuboids = Image.fromarray((img_cuboids).astype("uint8"))
    img_cuboids.save(path + "/" + file_prefix + "cuboids_all.png")


def save_pose_comparison_single_sample(
    img,
    estimated_poses,
    estimated_points,
    gt_poses,
    cuboids,
    keypoints,
    camera_matrix,
    offsets,
    path,
    file_prefix,
    normal=[0.5, 0.5],
    add_correct=None,
    draw_reprojection=True,
    split_by_no_correct=False,
):
    if split_by_no_correct and add_correct is not None:
        path = (
            path
            + "/"
            + tf.as_string(tf.cast(tf.reduce_sum(add_correct), tf.int32)).numpy().decode("utf-8")
            + "_correct"
        )

    if not exists(path):
        os.mkdir(path)

    img_keypoints = tf.cast(((img * normal[1]) + normal[0]) * 255, dtype=tf.uint8).numpy()
    offsets = offsets.numpy()

    eps = 1e-4
    img_cuboids = img_keypoints.copy()
    tf.print(add_correct, summarize=-1)
    for obj_idx, obj_pose_gt in enumerate(gt_poses):
        if add_correct is not None:
            add_obj_correct = add_correct[obj_idx]
        else:
            add_obj_correct = True

        if add_obj_correct:
            est_color = (0, 255, 0)
        else:
            est_color = (255, 0, 0)

        inst_idx = 0

        instance_pose_gt = gt_poses[obj_idx][inst_idx].numpy()
        obj_pose_est = estimated_poses[obj_idx].numpy()
        instance_cuboids = cuboids[obj_idx][inst_idx].numpy()
        instance_keypoints = keypoints[obj_idx][inst_idx].numpy()
        valid_gt = np.abs(np.sum(instance_pose_gt)) > eps
        valid_est = np.abs(np.sum(obj_pose_est)) > eps
        gt_color = (0, 0, 255)

        if valid_gt:
            transformed_cuboid_points2d_gt, _ = project(instance_cuboids, camera_matrix.numpy(), instance_pose_gt)
            transformed_cuboid_points2d_gt = apply_offsets(transformed_cuboid_points2d_gt, offsets)
            transformed_keypoints_points2d_gt, _ = project(instance_keypoints, camera_matrix.numpy(), instance_pose_gt)
            transformed_keypoints_points2d_gt = apply_offsets(transformed_keypoints_points2d_gt, offsets)

        if valid_est:
            transformed_cuboid_points2d, _ = project(instance_cuboids, camera_matrix.numpy(), obj_pose_est)
            transformed_cuboid_points2d = apply_offsets(transformed_cuboid_points2d, offsets)

            transformed_keypoints_points2d, _ = project(instance_keypoints, camera_matrix.numpy(), obj_pose_est)
            transformed_keypoints_points2d = apply_offsets(transformed_keypoints_points2d, offsets)

            valid_est = np.abs(np.sum(transformed_keypoints_points2d)) > eps

        if draw_reprojection:
            if valid_gt and valid_est:
                draw_lines(
                    transformed_keypoints_points2d_gt,
                    transformed_keypoints_points2d,
                    img_keypoints,
                )
            if valid_gt:
                draw_points(transformed_keypoints_points2d_gt, img_keypoints, gt_color)

            if valid_est:
                draw_points(transformed_keypoints_points2d, img_keypoints, est_color)

        if valid_gt:
            draw_bb(
                transformed_cuboid_points2d_gt,
                img_cuboids,
                gt_color,
                gt_color,
                width=2,
                line_type=cv2.LINE_AA,
            )
        if valid_est:
            draw_bb(
                transformed_cuboid_points2d,
                img_cuboids,
                est_color,
                est_color,
                width=2,
                line_type=cv2.LINE_AA,
            )

    img_cuboids = Image.fromarray((img_cuboids).astype("uint8"))
    img_cuboids.save(path + "/" + file_prefix + "cuboids.png")

    if draw_reprojection:
        img_keypoints = Image.fromarray((img_keypoints).astype("uint8"))
        img_keypoints.save(path + "/" + file_prefix + "reprojected_keypoints.png")


def save_eval_batch(
    img_batch,
    output_seg,
    target_dirs,
    output_dirs,
    estimated_poses,
    estimated_points,
    no_objects,
    no_features,
    path_out,
    confidence=None,
    add_correct=None,
):
    """
    Stores output images with bounding box, vectorfields and confidence maps.
    Puts results for every image in a separate folder.
    """
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    img_name = str(img_batch[12][0][0].numpy().decode("utf-8"))
    path_out += "/" + img_name

    img = img_batch[0][0]
    target_vertex = img_batch[3]
    keypoints = img_batch[4][0]
    cam_mat = img_batch[5][0]
    offsets = img_batch[7][0]

    cuboid3d = img_batch[9][0]
    gt_poses = img_batch[10][0]
    gt_points = tf.reverse(target_vertex[0], axis=[-1])
    target_seg = img_batch[1]
    estimated_poses = estimated_poses[0]
    estimated_points = tf.expand_dims(estimated_points[0], 1)

    # save estimated mask and vectorfields
    save_single_sample(
        img,
        output_seg[0],
        output_dirs[0],
        path_out,
        "",
        no_objects,
        no_features,
        locations=estimated_points,
        confidence=confidence,
    )
    # save gt mask and vectorfields
    save_single_sample(
        img,
        target_seg[0],
        target_dirs[0],
        path_out,
        "",
        no_objects,
        no_features,
        locations=gt_points,
        file_postfix="_gt",
    )
    # visualize reporojection error
    save_pose_comparison_single_sample(
        img,
        estimated_poses,
        estimated_points[0],
        gt_poses,
        cuboid3d,
        keypoints,
        cam_mat,
        offsets,
        path_out,
        "",
        draw_reprojection=False,
        add_correct=add_correct,
    )

    # visualize proxy voting error
    # vector_error, object_loss_values = proxy_voting_dist(
    #     output_dirs,
    #     target_vertex,
    #     vertex_one_hot_weights=target_seg[:, :, :, 1:],
    #     vertex_weights=target_seg[:, :, :, 0:1],
    #     invert_weights=True,
    # )

    # save_clamped_grayscale_single_sample(
    #     vector_error[0],
    #     target_seg[0],
    #     path_out,
    #     "",
    #     no_objects,
    #     no_features,
    #     clip_max=5.0,
    # )
    # save_mask_by_loss_value_single_sample(object_loss_values[0], target_seg[0], path_out, "", threshold=5)


def save_eval_comparison(
    img_batch, estimated_poses, estimated_points, path_out, add_correct=None, split_by_no_correct=False
):
    """
    Stores output images with bounding box.
    Creates a separate folder for every image.
    Sorts by number of correct matches (split_by_no_correct).
    """
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    img_name = str(img_batch[12][0][0].numpy().decode("utf-8")) + "_"
    path_out += "/" + "eval_comparison"
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    img = img_batch[0][0]
    keypoints = img_batch[4][0]
    cam_mat = img_batch[5][0]
    offsets = img_batch[7][0]

    cuboid3d = img_batch[9][0]
    gt_poses = img_batch[10][0]

    estimated_poses = estimated_poses[0]
    estimated_points = tf.expand_dims(estimated_points[0], 1)
    save_pose_comparison_single_sample(
        img,
        estimated_poses,
        estimated_points[0],
        gt_poses,
        cuboid3d,
        keypoints,
        cam_mat,
        offsets,
        path_out,
        img_name,
        draw_reprojection=False,
        split_by_no_correct=split_by_no_correct,
        add_correct=add_correct,
    )


def load_split(path, ratio):
    file_path = path + "/_split_settings.json"
    if os.path.isfile(file_path):
        with open(file_path) as f:
            split_info = json.load(f)
            if split_info["split"][0]["ratio"] == ratio:
                print("reload split with ratio {} in {}".format(ratio, file_path))
                split = split_info["split"][0]["values"]
            else:
                split = write_json_split(path, ratio)
    else:
        split = write_json_split(path, ratio)

    return split


def write_json_split(path, ratio):
    files = glob.glob(path + "/*seg.png")
    print("write new split with ratio {} in {}".format(ratio, path))
    file_count = len(files)
    split = np.zeros([file_count], dtype=int)
    split[0 : int(file_count * ratio)] = 1
    np.random.shuffle(split)
    json_data = {}
    json_data["split"] = []
    split_info = {}
    split_info["ratio"] = ratio
    split_info["values"] = split
    json_data["split"].append(split_info)
    with open(path + "/_split_settings.json", "w") as outfile:
        outfile.write(to_json(json_data))
    return split


def create_split(root, ratio):
    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
        if len(folders) > 0:
            for folder in folders:
                explore(folder)
        else:
            load_split(path, ratio)

    explore(root)


np.random.seed(123)
path = r"E:\DeepLearningData\BoBChallenge2020\lmo\mini"
ratio = 0.9

create_split(path, ratio)
