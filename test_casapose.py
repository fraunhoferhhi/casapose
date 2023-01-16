from __future__ import print_function

import glob
import os

import numpy as np
import tensorflow as tf

os.environ["CASAPOSE_INFERENCE"] = "True"

from casapose.data_handler.vectorfield_dataset import VectorfieldDataset
from casapose.pose_estimation.pose_evaluation import (
    estimate_and_evaluate_poses,
    evaluate_pose_estimates,
)
from casapose.pose_estimation.voting_layers_2d import CoordLSVotingWeighted
from casapose.pose_models.tfkeras import Classifiers
from casapose.utils.config_parser import parse_config
from casapose.utils.dataset_utils import save_eval_batch  # , save_eval_comparison
from casapose.utils.image_utils import get_all_vectorfields
from casapose.utils.io_utils import write_poses
from casapose.utils.loss_functions import (
    keypoint_reprojection_loss,
    proxy_voting_dist,
    proxy_voting_loss_v2,
    smooth_l1_loss,
)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


def compute_loss(
    output_seg,
    target_seg,
    output_vertex,
    target_vertex,
    target_points,
    mask_loss_weight=1.0,
    vertex_loss_weight=1.0,
    proxy_loss_weight=1.0,
    kp_loss_weight=0.0,
    kp_loss=None,
):

    oc = np.int32(target_seg.shape[3] - 1)  # object count
    vc = target_points.shape[3] * 2  # vertex count
    # kp_loss = tf.constant(0.0, dtype=tf.float32)
    mask_loss = tf.reduce_mean(
        input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=target_seg, logits=output_seg)
    )
    separated_vectors = oc > 1 and output_vertex.shape[-1] == (oc * vc)  # original pvnet with multiple objects

    if kp_loss is None:
        kp_loss = tf.constant(0.0, dtype=tf.float32)

    if separated_vectors:  #
        vertex_loss = sum(
            smooth_l1_loss(
                output_vertex[:, :, :, i * vc : (i + 1) * vc],
                target_vertex[:, :, :, i * vc : (i + 1) * vc],
                target_seg[:, :, :, i + 1 : i + 2],
            )
            for i in range(oc)
        )
        proxy_loss = sum(
            proxy_voting_loss_v2(
                output_vertex[:, :, :, i * vc : (i + 1) * vc],
                target_points[:, i : i + 1, :, :, :],
                vertex_one_hot_weights=target_seg[:, :, :, i + 1 : i + 2],
                vertex_weights=target_seg[:, :, :, i + 1 : i + 2],
            )
            for i in range(oc)
        )
    else:
        vertex_loss = smooth_l1_loss(
            output_vertex,
            target_vertex,
            target_seg[:, :, :, 0:1],
            invert_weights=True,
        )
        proxy_loss = proxy_voting_loss_v2(
            output_vertex,
            target_points,
            vertex_one_hot_weights=target_seg[:, :, :, 1:],
            vertex_weights=target_seg[:, :, :, 0:1],
            invert_weights=True,
            loss_per_object=False,
        )

    loss = (
        tf.multiply(mask_loss, mask_loss_weight)
        + tf.multiply(proxy_loss, proxy_loss_weight)
        + tf.multiply(vertex_loss, vertex_loss_weight)
        + tf.multiply(kp_loss, kp_loss_weight)
    )
    return [loss, mask_loss, vertex_loss, proxy_loss, kp_loss]


opt = parse_config()

# tf.config.run_functions_eagerly(opt.write_poses or opt.save_eval_batches)

if not os.path.exists(opt.evalf):
    os.makedirs(opt.evalf)

checkpoint_path = opt.outf + "/" + opt.net

frozen_path = opt.outf + "/frozen_model"
img_out_path = opt.outf + "/control_output"


def create_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


create_dir(img_out_path)
create_dir(frozen_path)

# save the hyper parameters passed
with open(opt.evalf + "/header_eval.txt", "w") as file:
    file.write(str(opt))

# set the manual seed.
np.random.seed(opt.manualseed)
tf.random.set_seed(opt.manualseed)

test_dataset = None

device_ids = []
if len(opt.gpuids) == 1 and opt.gpuids[0] < 0:
    device_ids.append("/cpu:0")
else:
    device_ids.append("/gpu:{}".format(opt.gpuids[0]))
print(device_ids)

objectsofinterest = [x.strip() for x in opt.object.split(",")]
no_objects = len(objectsofinterest)
separated_vectorfields = opt.modelname == "pvnet"

testingdata = None
normal_imgs = [0.5, 0.5]

use_split = False
if opt.data == opt.datatest:
    print("split datasets with ratio {}".format(opt.train_validation_split))
    use_split = True

test_batches = 0

test_dataset = VectorfieldDataset(
    root=opt.datatest,
    path_meshes=opt.datameshes,
    path_filter_root=opt.datatest_path_filter,
    color_input=opt.color_dataset,
    no_points=opt.no_points,
    objectsofinterest=objectsofinterest,
    noise=0.00001,
    data_size=None,
    save=opt.save_debug_batch,
    normal=normal_imgs,
    contrast=0.00001,
    brightness=0.00001,
    hue=0.00001,
    saturation=0.00001,
    random_translation=(0, 0),
    random_rotation=0,
    random_crop=False,
    use_validation_split=use_split,
    use_train_split=False,
    train_validation_split=opt.train_validation_split,
    output_folder=opt.evalf,
    visibility_filter=False,
    separated_vectorfields=(opt.modelname == "pvnet"),
    wxyz_quaterion_input=opt.datatest_wxyz_quaterion,
)
print(len(test_dataset))
testingdata, test_batches = test_dataset.generate_dataset(
    1, 1, 0, opt.imagesize_test, 1.0, 1, no_objects, shuffle=False
)

mesh_vertex_array, mesh_vertex_count = test_dataset.generate_object_vertex_array()

print("testing data: {} batches".format(test_batches))

if opt.backbonename != "resnet18":
    print(opt.backbonename + " is not a supported backbone.")
    exit()


input_segmentation_shape = None
if opt.train_vectors_with_ground_truth is True:
    input_segmentation_shape = (
        opt.imagesize_test[0],
        opt.imagesize_test[1],
        1 + no_objects,
    )


height = opt.imagesize_test[0]
width = opt.imagesize_test[1]

CASAPose = Classifiers.get(opt.modelname)
ver_dim = opt.no_points * 2
if opt.modelname == "pvnet":
    ver_dim = ver_dim * no_objects

if opt.estimate_confidence:
    assert separated_vectorfields is not None, "confidence not compaitble with this model"
    ver_dim += opt.no_points

net = CASAPose(
    ver_dim=ver_dim,
    seg_dim=1 + no_objects,
    input_shape=(height, width, 3),
    input_segmentation_shape=input_segmentation_shape,
    weights="imagenet",
    base_model=opt.backbonename,
)

checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")
checkpoint = tf.train.Checkpoint(network=net)  # , optimizer=optimizer)


if opt.load_h5_weights is True:
    net_path = frozen_path + "/" + opt.load_h5_filename + ".h5"
    print(net_path)
    net.load_weights(net_path, by_name=True, skip_mismatch=True)


elif opt.net != "":
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()


for layer in net.layers:
    layer.trainable = False

net.summary()

with open(opt.evalf + "/loss_test_eval.csv", "w") as file:
    file.write(
        "batchid,loss,mask_loss,vertex_loss,proxy_loss,kp_loss,mask_loss_weight,vertex_loss_weight,proxy_loss_weight,kp_loss_weight\n"
    )

with open(opt.evalf + "/test_summary_eval.csv", "w") as file:
    s = "loss,mask_loss,vertex_loss,proxy_loss,kp_loss,time"
    for obj in objectsofinterest:
        s += ",2d_{}".format(obj)
    s += ",2d_mean"
    for obj in objectsofinterest:
        s += ",3d_{}".format(obj)
    s += ",3d_mean"
    s += "\n"
    file.write(s)

if os.path.exists(opt.evalf + "/poses_out/"):
    files = sorted(glob.glob(opt.evalf + "/poses_out/*/" + "*.txt"))
    for f in files:
        os.remove(f)


def runnetwork(loader_iterator, batches):
    @tf.function
    def tf_test_step(img_batch):
        return test_step(img_batch)

    # @tf.function
    def test_step(img_batch):
        loss = 0

        target_seg = img_batch[1]
        target_vertex = img_batch[3]
        keypoints = img_batch[4]
        cam_mat = img_batch[5]
        diameters = img_batch[6]
        offsets = img_batch[7]
        filtered_seg = img_batch[8]
        poses_gt = img_batch[10]
        confidence = None
        kp_loss = None
        no_features = target_vertex.shape[3]  # vertex count
        no_points = opt.no_points
        no_objects = target_seg.shape[3]

        separated_vectorfields = opt.modelname == "pvnet"

        target_dirs = get_all_vectorfields(
            target_seg,
            target_vertex,
            filtered_seg,
            separated_vectorfields,
        )

        net_input = [img_batch[0]]
        if opt.train_vectors_with_ground_truth:
            net_input.append(target_seg)

        start_time = tf.timestamp()
        output_net = net(net_input, training=False)  # all stages are present here

        if opt.estimate_confidence:
            output_seg, output_dirs, confidence = tf.split(output_net, [no_objects, no_points * 2, -1], 3)
        else:
            output_seg, output_dirs = tf.split(output_net, [no_objects, -1], 3)

        if opt.estimate_coords:
            if opt.train_vectors_with_ground_truth:
                coordLSV_in = [target_seg, output_dirs, confidence]
            else:
                coordLSV_in = [output_seg, output_dirs, confidence]

            coords = CoordLSVotingWeighted(
                name="coords_ls_voting",
                num_classes=no_objects,
                num_points=no_points,
                filter_estimates=opt.confidence_filter_estimates,
                output_second_largest_component=opt.confidence_choose_second,
            )(coordLSV_in)

            kp_loss, poses_est, points_est = keypoint_reprojection_loss(
                coords,
                output_seg,
                poses_gt,
                keypoints,
                target_seg,
                cam_mat,
                offsets,
                confidence,
                min_num=opt.min_object_size_test,
                min_num_gt=1,
                use_bpnp_reprojection_loss=opt.use_bpnp_reprojection_loss,
                estimate_poses=True,
                filter_with_gt=opt.filter_test_with_gt,
            )

        if opt.estimate_coords:
            pose_stats, estimated_poses, estimated_points = evaluate_pose_estimates(
                points_est,
                poses_est,
                poses_gt,
                target_seg,
                keypoints,
                cam_mat,
                diameters,
                evaluation_points=mesh_vertex_array,
                object_points_3d_count=mesh_vertex_count,
                min_num=1,
            )
            estimated_poses = tf.squeeze(estimated_poses, axis=2)
        else:
            pose_stats, estimated_poses, estimated_points = estimate_and_evaluate_poses(
                output_seg,
                target_seg,
                output_dirs,
                poses_gt,
                keypoints,
                cam_mat,
                diameters,
                offsets,
                evaluation_points=mesh_vertex_array,
                object_points_3d_count=mesh_vertex_count,
                min_num=1,
            )

        end_time = tf.timestamp()
        time_needed = end_time - start_time

        loss = compute_loss(
            output_seg,
            target_seg,
            output_dirs,
            target_dirs,
            target_vertex,
            opt.mask_loss_weight,
            opt.vertex_loss_weight,
            opt.proxy_loss_weight,
            opt.keypoint_loss_weight,
            kp_loss=kp_loss,
        )
        loss.append(pose_stats)

        _, object_loss_values = proxy_voting_dist(
            output_dirs,
            target_vertex,
            vertex_one_hot_weights=target_seg[:, :, :, 1:],
            vertex_weights=target_seg[:, :, :, 0:1],
            invert_weights=True,
        )
        loss.append(object_loss_values)
        loss.append(time_needed)

        if opt.write_poses:
            write_poses(
                tf.squeeze(poses_gt, 0),
                tf.squeeze(estimated_poses, 0),
                objectsofinterest,
                img_batch[12],
                opt.evalf + "/poses_out/",
            )

        if opt.save_eval_batches:

            beta = tf.cast(1e6, dtype=output_seg.dtype)
            hot_seg = tf.expand_dims(tf.expand_dims(tf.nn.softmax(output_seg * beta), -1), -1)[:, :, :, 1:, :, :]
            w = tf.math.softplus(confidence)
            hot_seg_ = tf.squeeze(hot_seg, axis=-1)
            w_ = hot_seg_ * tf.expand_dims(w, axis=-2)
            w_max = tf.reduce_max(w_, axis=[0, 1, 2], keepdims=True)
            w_min = tf.reduce_min(w_, axis=[0, 1, 2], keepdims=True)
            w_ = tf.math.divide_no_nan(tf.subtract(w_, w_min), tf.subtract(w_max, w_min))
            confidence = tf.reduce_sum(w_, axis=-2)

            add_correct = loss[5][1]

            # save_eval_comparison(
            #     img_batch,
            #     estimated_poses,
            #     estimated_points,
            #     path_out=opt.evalf + "/visual_batch_eval_mask",
            #     add_correct=add_correct,
            #     split_by_no_correct=False
            # )

            save_eval_batch(
                img_batch,
                output_seg,
                target_dirs,
                output_dirs,
                estimated_poses,
                estimated_points,
                no_objects - 1,
                no_features,
                path_out=opt.evalf + "/visual_batch_eval_mask",
                confidence=confidence,
                add_correct=add_correct,
            )
        return loss

    def test_pose_step(dataset_inputs):
        if opt.save_eval_batches or opt.write_poses:
            loss = test_step(dataset_inputs)
        else:
            loss = tf_test_step(dataset_inputs)

        return [
            loss[0],
            loss[1],
            loss[2],
            loss[3],
            loss[4],
            loss[5][0],
            loss[5][1],
            loss[5][2],
            loss[5][3],
            loss[5][4],
            loss[5][5],
            loss[5][6],
            loss[5][7],
            loss[6],
            loss[7],
        ]

    test_loss = tf.zeros([5], dtype=tf.float32)
    test_pose_2d_count = tf.zeros([no_objects], dtype=tf.float32)
    test_pose_3d_count = tf.zeros([no_objects], dtype=tf.float32)
    test_pose_count_gt = tf.zeros([no_objects], dtype=tf.float32)
    test_pose_count_fp = tf.zeros([no_objects], dtype=tf.float32)
    test_pose_err_2d = tf.zeros([no_objects], dtype=tf.float32)
    test_pose_err_3d = tf.zeros([no_objects], dtype=tf.float32)
    missed_object_count = tf.zeros([no_objects], dtype=tf.float32)

    for batch_idx in range(batches):

        img_batch = loader_iterator.get_next()
        loss = test_pose_step(img_batch)
        test_pose_2d_count += loss[5]
        test_pose_3d_count += loss[6]
        test_pose_count_gt += loss[7]
        test_pose_count_fp += loss[12]
        test_pose_err_2d += loss[9]
        test_pose_err_3d += loss[10]
        missed_object_count += loss[11]
        test_loss += loss[0:5]

        namefile = "/loss_test_eval.csv"

        with open(opt.evalf + namefile, "a") as file:
            s = "{},{:.15f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f}\n".format(
                batch_idx + 1, loss[0], loss[1], loss[2], loss[3], loss[4], loss[-1]
            )
            file.write(s)

            tf.print(
                "Batch idx: {}, Loss: {:.5f} --- mask: {:.5f}, vector: {:.5f}, proxy: {:.5}, kp: {:.5} -- Average Loss: {:.5f}\n".format(
                    batch_idx,
                    loss[0],
                    loss[1],
                    loss[2],
                    loss[3],
                    loss[4],
                    test_loss[0] / (batch_idx + 1),
                )
            )
            tf.print("Test GT: {}".format(loss[7]), summarize=-1)
            tf.print("Test 2D: {}".format(loss[5]), summarize=-1)
            tf.print("Test 3D: {}".format(loss[6]), summarize=-1)
            tf.print("Test Sum GT: {}".format(test_pose_count_gt), summarize=-1)
            tf.print("Test Sum 2D: {}".format(test_pose_2d_count), summarize=-1)
            tf.print("Test Sum 3D: {}".format(test_pose_3d_count), summarize=-1)
            tf.print("Misses: {}".format(missed_object_count), summarize=-1)
            tf.print("False positive: {}".format(test_pose_count_fp), summarize=-1)

            tf.print("Err 2D: {}".format(loss[9]), summarize=-1)
            tf.print("Err 3D: {}".format(loss[10]), summarize=-1)

    test_loss /= batches

    err_2d = tf.math.divide_no_nan(test_pose_2d_count, test_pose_count_gt)
    err_3d = tf.math.divide_no_nan(test_pose_3d_count, test_pose_count_gt)
    detection_count = test_pose_count_gt - missed_object_count + test_pose_count_fp
    detection_count = tf.where(test_pose_count_gt == 0.0, 0.0, detection_count)
    precision = tf.math.divide_no_nan(test_pose_3d_count, detection_count)

    tf.print("==========================")
    tf.print(
        "== TEST == Finished test with total loss: {:.7f} --- mask: {:.7f}, vector: {:.7f}, proxy: {:.7f}, kp: {:.7} ==".format(
            test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4]
        )
    )
    tf.print("2D Valid: {}".format(err_2d), summarize=-1)
    tf.print("2D Valid (mean): {}".format(tf.reduce_mean(err_2d)), summarize=-1)
    tf.print("3D Valid: {}".format(err_3d), summarize=-1)
    tf.print("3D Valid (mean): {}".format(tf.reduce_mean(err_3d)), summarize=-1)
    tf.print("3D Valid (precision): {}".format(precision), summarize=-1)
    tf.print(
        "3D Valid (average precision): {}".format(tf.reduce_mean(precision)),
        summarize=-1,
    )
    # tf.print("Err 2D: {}".format(tf.math.divide_no_nan(test_pose_err_2d, test_pose_count_gt)), summarize=-1)
    tf.print("==========================")

    namefile = "/test_summary_eval.csv"

    with open(opt.evalf + namefile, "a") as file:
        s = "{:.7f},{:.7f},{:.7f},{:.7f},{:.7f}".format(
            test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4]
        )
        for i in range(no_objects):
            s += ",{:.4f}".format(err_2d[i])
        s += ",{:.4f}".format(tf.reduce_mean(err_2d))
        for i in range(no_objects):
            s += ",{:.4f}".format(err_3d[i])
        s += ",{:.4f}".format(tf.reduce_mean(err_3d))
        s += "\n"
        file.write(s)
    return loader_iterator


print("Test Batches: {} ".format(test_batches))

testingdata_iterator = iter(testingdata)
runnetwork(testingdata_iterator, int(test_batches))
