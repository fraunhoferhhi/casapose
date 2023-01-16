from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import os
import time

import numpy as np
import tensorflow as tf
from casapose.data_handler.vectorfield_dataset import VectorfieldDataset
from casapose.pose_estimation.pose_evaluation import (
    estimate_and_evaluate_poses,
    evaluate_pose_estimates,
)
from casapose.pose_estimation.voting_layers_2d import CoordLSVotingWeighted
from casapose.pose_models.tfkeras import Classifiers
from casapose.utils.config_parser import parse_config
from casapose.utils.image_utils import get_all_vectorfields
from casapose.utils.learning_rate_schedules import (
    ExponentialDecayLateStart,
    LossWeightHandler,
)
from casapose.utils.loss_functions import (
    keypoint_reprojection_loss,
    proxy_voting_dist,
    proxy_voting_loss_v2,
    smooth_l1_loss,
)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        tf.print(e)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def compute_loss(
    output_seg,
    target_seg,
    output_vert,
    target_vert,
    target_points,
    loss_factors,
    filtered_seg=None,
    pixel_gt_count=None,
    kp_loss=None,
):

    oc = np.int32(target_seg.shape[3] - 1)  # object count
    vc = target_points.shape[3] * 2  # vertex count
    mask_loss = tf.constant(0.0, dtype=tf.float32)
    vertex_loss = tf.constant(0.0, dtype=tf.float32)
    proxy_loss = tf.constant(0.0, dtype=tf.float32)
    separated_vectors = oc > 1 and output_vert.shape[-1] == oc * vc  # original pvnet with multiple objects

    if loss_factors.mask_loss_weight > 0.0:
        mask_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_seg, logits=output_seg))

    if filtered_seg is not None:
        target_seg = tf.one_hot(tf.squeeze(filtered_seg, -1), oc + 1, dtype=tf.float32)
    if loss_factors.filter_vertex_with_segmentation:
        target_seg = tf.where(
            (tf.expand_dims(tf.argmax(target_seg, -1), -1) == tf.expand_dims(tf.argmax(output_seg, -1), -1)),
            target_seg,
            tf.one_hot([0], oc + 1),
        )

    if loss_factors.filter_high_proxy_errors and pixel_gt_count is not None:
        _, object_loss_values = proxy_voting_dist(
            output_vert,
            target_points,
            vertex_one_hot_weights=target_seg[:, :, :, 1:],
            vertex_weights=target_seg[:, :, :, 0:1],
            invert_weights=True,
        )
        object_loss_values = tf.concat(
            [
                tf.ones([tf.shape(object_loss_values)[0], 1]),
                tf.cast((object_loss_values < 5), dtype=tf.float32),
            ],
            axis=-1,
        )
        object_loss_values = tf.expand_dims(tf.expand_dims(object_loss_values, axis=1), axis=1)
        object_loss_values = tf.stop_gradient(object_loss_values)

        cond = tf.expand_dims(
            tf.reduce_sum(object_loss_values * target_seg, axis=-1) > 0,
            axis=-1,
        )
        target_seg = tf.where(cond, target_seg, tf.one_hot([0], oc + 1))

    target_seg = tf.stop_gradient(target_seg)
    if loss_factors.vertex_loss_weight > 0.0:
        if separated_vectors:
            vertex_loss = sum(
                smooth_l1_loss(
                    output_vert[:, :, :, i * vc : (i + 1) * vc],
                    target_vert[:, :, :, i * vc : (i + 1) * vc],
                    target_seg[:, :, :, i + 1 : i + 2],
                )
                for i in range(oc)
            )

        else:
            vertex_loss = smooth_l1_loss(
                output_vert,
                target_vert,
                target_seg[:, :, :, 0:1],
                invert_weights=True,
            )

    if loss_factors.proxy_loss_weight > 0.0:
        if separated_vectors:
            proxy_loss = sum(
                proxy_voting_loss_v2(
                    output_vert[:, :, :, i * vc : (i + 1) * vc],
                    target_points[:, i : i + 1, :, :, :],
                    vertex_one_hot_weights=target_seg[:, :, :, i + 1 : i + 2],
                    vertex_weights=target_seg[:, :, :, i + 1 : i + 2],
                )
                for i in range(oc)
            )
        else:
            proxy_loss = proxy_voting_loss_v2(
                output_vert,
                target_points,
                vertex_one_hot_weights=target_seg[:, :, :, 1:],
                vertex_weights=target_seg[:, :, :, 0:1],
                invert_weights=True,
                loss_per_object=False,
            )

    if kp_loss is None:
        kp_loss = tf.constant(0.0, dtype=tf.float32)

    loss = (
        tf.multiply(mask_loss, loss_factors.mask_loss_weight)
        + tf.multiply(proxy_loss, loss_factors.proxy_loss_weight)
        + tf.multiply(vertex_loss, loss_factors.vertex_loss_weight)
        + tf.multiply(kp_loss, loss_factors.kp_loss_weight)
    )
    return [loss, mask_loss, vertex_loss, proxy_loss, kp_loss]


##################################################
# TRAINING CODE MAIN STARTING HERE
##################################################

tf.print("start:", datetime.datetime.now().time())

opt = parse_config()

checkpoint_path = opt.outf + "/" + opt.net

frozen_path = opt.outf + "/frozen_model"


def create_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


create_dir(opt.outf)
create_dir(checkpoint_path)
create_dir(frozen_path)


# save the hyper parameters passed
with open(opt.outf + "/header.txt", "w") as file:
    file.write(str(opt))
    # file.write("seed: "+ str(opt.manualseed)+'\n')

# set the manual seed.
np.random.seed(opt.manualseed)
tf.random.set_seed(opt.manualseed)

# load the dataset using the loader in utils_pose
trainingdata = None
train_dataset = None
test_dataset = None

device_ids = []
if len(opt.gpuids) == 1 and opt.gpuids[0] < 0:
    device_ids.append("/cpu:0")
else:
    for gpu in opt.gpuids:
        device_ids.append("/gpu:{}".format(gpu))
tf.print(device_ids)

mirrored_strategy = tf.distribute.MirroredStrategy(devices=device_ids)

worker = opt.workers
if worker == 0:
    worker = tf.data.experimental.AUTOTUNE

prefetch = opt.prefetch
if prefetch == 0:
    prefetch = tf.data.experimental.AUTOTUNE
objectsofinterest = [x.strip() for x in opt.object.split(",")]
no_objects = len(objectsofinterest)

trainingdata = None
testingdata = None

normal_imgs = [0.5, 0.5]

use_split = False
if opt.data == opt.datatest:
    tf.print("split datasets with ratio {}".format(opt.train_validation_split))
    use_split = True

if opt.data_wxyz_quaterion or opt.datatest_wxyz_quaterion:
    tf.print("\nWARNING: DATASET HAS WRONG QUATERNION FORMAT. USE fix_dataset.py FOR CORRECTION.\n")


separated_vectorfields = opt.modelname == "pvnet"

train_batches = 0
test_batches = 0
if not opt.data == "":
    train_dataset = VectorfieldDataset(
        root=opt.data,
        path_meshes=opt.datameshes,
        path_filter_root=opt.data_path_filter,
        color_input=opt.color_dataset,
        no_points=opt.no_points,
        objectsofinterest=objectsofinterest,
        noise=opt.noise,
        data_size=None,
        save=opt.save_debug_batch,
        normal=normal_imgs,
        contrast=opt.contrast,
        brightness=opt.brightness,
        hue=opt.hue,
        saturation=opt.saturation,
        random_translation=(opt.translation, opt.translation),
        random_rotation=opt.rotation,
        use_train_split=use_split,
        train_validation_split=opt.train_validation_split,
        output_folder=opt.outf,
        use_imgaug=opt.use_imgaug,
        random_crop=True,
        separated_vectorfields=separated_vectorfields,
        wxyz_quaterion_input=opt.data_wxyz_quaterion,
    )
    trainingdata, train_batches = train_dataset.generate_dataset(
        opt.batchsize,
        opt.epochs,
        prefetch,
        opt.imagesize,
        opt.crop_factor,
        worker,
        no_objects,
        mirrored_strategy=mirrored_strategy,
    )

    tf.print("training data: {} batches".format(train_batches))
tf.print(opt.datatest)
if not opt.datatest == "":
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
        use_validation_split=use_split,
        train_validation_split=opt.train_validation_split,
        output_folder=opt.outf,
        random_crop=False,
        separated_vectorfields=separated_vectorfields,
        wxyz_quaterion_input=opt.datatest_wxyz_quaterion,
    )
    tf.print(len(test_dataset))
    testingdata, test_batches = test_dataset.generate_dataset(
        opt.batchsize,
        opt.epochs,
        prefetch,
        opt.imagesize,
        opt.crop_factor,
        worker,
        no_objects,
        mirrored_strategy=mirrored_strategy,
    )
    tf.print("testing data: {} batches".format(test_batches))

if opt.backbonename != "resnet18":
    tf.print(opt.backbonename + " is not a supported backbone.")
    exit()


height = opt.imagesize[0]
width = opt.imagesize[1]

input_segmentation_shape = None
if opt.train_vectors_with_ground_truth is True:
    input_segmentation_shape = (height, width, 1 + no_objects)


with mirrored_strategy.scope():

    CASAPose = Classifiers.get(opt.modelname)
    ver_dim = opt.no_points * 2
    if separated_vectorfields:
        ver_dim = ver_dim * no_objects

    if opt.estimate_confidence:
        assert separated_vectorfields is not None, "confidence not compaitble with this model"
        ver_dim += opt.no_points

    backbone = None

    net = CASAPose(
        ver_dim=ver_dim,
        seg_dim=1 + no_objects,
        input_shape=(height, width, 3),
        input_segmentation_shape=input_segmentation_shape,
        weights="imagenet",
        base_model=opt.backbonename,
        backbone=backbone,
    )

    if opt.lr_epochs_steps is not None:
        boundaries = ((np.array(opt.lr_epochs_steps) * train_batches) - 1).tolist()
        values = (np.power(opt.lr_decay, np.arange(len(boundaries) + 1)) * opt.lr).tolist()
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    else:
        lr_schedule = ExponentialDecayLateStart(
            opt.lr,
            decay_steps=train_batches * opt.lr_epochs,
            decay_steps_start=train_batches * opt.lr_epochs_start,
            decay_rate=opt.lr_decay,
            staircase=True,
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")

    checkpoint = tf.train.Checkpoint(network=net)

    if opt.copy_weights_add_confidence_maps and opt.estimate_confidence:
        net_backup = CASAPose(
            ver_dim=ver_dim - opt.no_points,
            seg_dim=1 + no_objects,
            input_shape=(height, width, 3),
            input_segmentation_shape=input_segmentation_shape,
            weights="imagenet",
            base_model=opt.backbonename,
            backbone=backbone,
        )
    elif opt.copy_weights_from_backup_network:
        net_backup = CASAPose(
            ver_dim=ver_dim,
            seg_dim=1 + opt.objects_in_input_network,
            input_shape=(height, width, 3),
            input_segmentation_shape=None,
            weights="imagenet",
            base_model=opt.backbonename,
        )

    #################################################################
    if opt.copy_weights_from_backup_network or opt.copy_weights_add_confidence_maps:
        net_backup.load_weights(frozen_path + "/" + opt.load_h5_filename + ".h5", by_name=True, skip_mismatch=True)
        tf.print("loaded backup network")

    if opt.load_h5_weights is True:
        net.load_weights(frozen_path + "/" + opt.load_h5_filename + ".h5", by_name=True, skip_mismatch=True)
    elif opt.net != "":
        # freeze multiple checkpoints
        # model_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        # no = int(model_checkpoint_path[-2:])
        # model_checkpoint_base = model_checkpoint_path[:-2]
        # print(model_checkpoint_base)
        # for i in range(no):
        #     #net = CASAPose(ver_dim = ver_dim, seg_dim = 1 + no_objects, input_shape=(height, width,3), input_segmentation_shape=input_segmentation_shape, weights='imagenet', base_model=opt.backbonename, backbone = backbone)
        #     checkpoint = tf.train.Checkpoint(network=net)
        #     new_path = model_checkpoint_base + str(i+1)
        #     print(new_path)
        #     checkpoint.restore(new_path).expect_partial()
        #     net.save_weights(frozen_path + '/result_w' + str(i+1) +'.h5')
        # exit()
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
        # net.save_weights(frozen_path + '/result_w.h5')
        # exit()

    tf.print(net.summary())

    if opt.copy_weights_add_confidence_maps and opt.estimate_confidence:

        def copy_weights_vertex(name, net, net_backup, ver_dim):
            tf.print("Copy weights for {}".format(name))
            block = net.get_layer(name).get_weights()
            backup_block = net_backup.get_layer(name).get_weights()
            block[0][0, 0, :, :ver_dim] = backup_block[0][0, 0, :, :ver_dim]  # 1, 1, 32, n
            net.get_layer(name).set_weights(block)
            return net

        net = copy_weights_vertex("pv_final_conv_vertex", net, net_backup, ver_dim - opt.no_points)
    elif opt.copy_weights_from_backup_network:
        range_in = opt.objects_to_copy[:, 0].tolist()  # add list to config to allow more specific copying
        range_out = opt.objects_to_copy[:, 1].tolist()

        def copy_weights_segmentation(name, net, net_backup, range_out, range_in):
            tf.print("Copy weights for {}".format(name))
            block = net.get_layer(name).get_weights()
            backup_block = net_backup.get_layer(name).get_weights()
            block[0][0, 0, :, range_out] = backup_block[0][0, 0, :, range_in]  # 1, 1, 32, n
            net.get_layer(name).set_weights(block)
            return net

        def copy_weights_scale(name, net, net_backup, range_out, range_in):
            tf.print("Copy weights for {}".format(name))
            block = net.get_layer(name).get_weights()
            backup_block = net_backup.get_layer(name).get_weights()
            block[0][np.ix_([0], [0], range_out, range_out)] = backup_block[0][np.ix_([0], [0], range_in, range_in)]
            net.get_layer(name).set_weights(block)
            return net

        def copy_clade_weights(name, net, net_backup, range_out, range_in):
            tf.print("Copy weights for {}".format(name))
            block = net.get_layer(name).get_weights()
            backup_block = net_backup.get_layer(name).get_weights()
            block[0][range_out] = backup_block[0][range_in]
            block[1][range_out] = backup_block[1][range_in]
            net.get_layer(name).set_weights(block)
            return net

        net = copy_weights_segmentation("pv_final_conv_segmentation", net, net_backup, range_out, range_in)
        # net = copy_weights_scale('segmentation_half_size', net, net_backup, range_out, range_in)
        # net = copy_weights_scale('segmentation_quater_size', net, net_backup, range_out, range_in)
        # net = copy_weights_scale('segmentation_eighth_size', net, net_backup, range_out, range_in)
        net = copy_clade_weights("pv_block_6_clade", net, net_backup, range_out, range_in)
        net = copy_clade_weights("pv_block_7_clade", net, net_backup, range_out, range_in)
        net = copy_clade_weights("pv_block_8_clade", net, net_backup, range_out, range_in)
        net = copy_clade_weights("pv_block_9_clade", net, net_backup, range_out, range_in)
        net = copy_clade_weights("pv_block_10_clade", net, net_backup, range_out, range_in)

    # net.save_weights(frozen_path + '/result_w_13obj.h5')
    # exit()

    # for layer in net.layers:
    #    if layer.trainable is False:
    #        tf.print("Frozen Layer: {}".format(layer.name))

    tf.print(optimizer.get_config())

    loss_factors = LossWeightHandler(
        mask_loss_weight=opt.mask_loss_weight,
        vertex_loss_weight=opt.vertex_loss_weight,
        proxy_loss_weight=opt.proxy_loss_weight,
        kp_loss_weight=opt.keypoint_loss_weight,
        filter_vertex_with_segmentation=opt.filter_vertex_with_segmentation,
        filter_high_proxy_errors=opt.filter_high_proxy_errors,
    )

    tf.print("Number of layers in the base model: ", len(net.layers))


# prepare output files
with open(opt.outf + "/loss_train.csv", "w") as file:  # write title for documentation csv
    file.write(
        "epoch,batchid,loss,mask_loss,vertex_loss,proxy_loss,keypoint_loss,mask_loss_weight,vertex_loss_weight,proxy_loss_weight, kp_loss_weight\n"
    )

with open(opt.outf + "/loss_test.csv", "w") as file:
    file.write(
        "epoch,batchid,loss,mask_loss,vertex_loss,proxy_loss,keypoint_loss,mask_loss_weight,vertex_loss_weight,proxy_loss_weight, kp_loss_weight\n"
    )

with open(opt.outf + "/train_summary.csv", "w") as file:
    file.write("epoch,learning_rate,loss,mask_loss,vertex_loss,proxy_loss,keypoint_loss\n")

with open(opt.outf + "/test_summary.csv", "w") as file:
    s = "epoch,learning_rate,loss,mask_loss,vertex_loss,proxy_loss,keypoint_loss"
    for obj in objectsofinterest:
        s += ",2d_{}".format(obj)
    for obj in objectsofinterest:
        s += ",3d_{}".format(obj)
    s += "\n"
    file.write(s)


def runnetwork(
    loader_iterator,
    batches_per_epoch,
    epoch,
    checkpoint,
    loss_factors,
    train=True,
    pose_validation=False,
):

    with mirrored_strategy.scope():

        # @tf.function
        def train_step(img_batch, loss_factors, train, pose_estimation):
            loss = 0
            confidence = None
            kp_loss = None
            img = img_batch[0]
            target_seg = img_batch[1]
            target_vert = img_batch[3]
            keypoints = img_batch[4]
            cam_mat = img_batch[5]
            diameters = img_batch[6]
            offsets = img_batch[7]
            filtered_seg = img_batch[8]
            poses_gt = img_batch[10]
            pixel_gt_count = img_batch[11]

            separated_vectorfields = opt.modelname == "pvnet"
            no_objects = tf.shape(target_seg)[3]
            no_points = opt.no_points
            net_input = [img]
            if opt.train_vectors_with_ground_truth:
                net_input.append(target_seg)

            target_dirs = get_all_vectorfields(
                target_seg,
                target_vert,
                filtered_seg,
                separated_vectorfields,
            )

            if train:
                with tf.GradientTape() as gradient_tape:
                    output_net = net(net_input, training=train)  # all stages are present here
                    if opt.estimate_confidence:
                        output_seg, output_dirs, confidence = tf.split(output_net, [no_objects, no_points * 2, -1], 3)
                    else:
                        output_seg, output_dirs = tf.split(output_net, [no_objects, -1], 3)

                    if opt.estimate_coords:

                        if opt.train_vectors_with_ground_truth:
                            coordLSV_in = [
                                target_seg,
                                output_dirs,
                                confidence,
                            ]
                        else:
                            coordLSV_in = [
                                output_seg,
                                output_dirs,
                                confidence,
                            ]
                        coords = CoordLSVotingWeighted(
                            name="coords_ls_voting",
                            num_classes=no_objects,
                            num_points=no_points,
                            filter_estimates=False,
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
                            max_pixel_error=opt.max_keypoint_pixel_error,
                            min_num=50,
                            use_bpnp_reprojection_loss=opt.use_bpnp_reprojection_loss,
                            estimate_poses=opt.use_bpnp_reprojection_loss,
                            confidence_regularization=opt.confidence_regularization,
                        )

                    loss = compute_loss(
                        output_seg,
                        target_seg,
                        output_dirs,
                        target_dirs,
                        target_vert,
                        loss_factors,
                        filtered_seg,
                        pixel_gt_count,
                        kp_loss=kp_loss,
                    )

                trainable_variables = net.trainable_variables

                gradients = gradient_tape.gradient(loss[0], trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))
            else:
                output_net = net(net_input, training=train)  # all stages are present here
                if opt.estimate_confidence:
                    output_seg, output_dirs, confidence = tf.split(output_net, [no_objects, no_points * 2, -1], 3)
                else:
                    output_seg, output_dirs = tf.split(output_net, [no_objects, -1], 3)

                if opt.estimate_coords:
                    if opt.train_vectors_with_ground_truth:
                        coordLSV_in = [
                            target_seg,
                            output_dirs,
                            confidence,
                        ]
                    else:
                        coordLSV_in = [
                            output_seg,
                            output_dirs,
                            confidence,
                        ]
                    coords = CoordLSVotingWeighted(
                        name="coords_ls_voting",
                        num_classes=no_objects,
                        num_points=no_points,
                        filter_estimates=False,
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
                        max_pixel_error=opt.max_keypoint_pixel_error,
                        min_num=50,
                        use_bpnp_reprojection_loss=opt.use_bpnp_reprojection_loss,
                        estimate_poses=True,
                        confidence_regularization=False,
                    )

                loss = compute_loss(
                    output_seg,
                    target_seg,
                    output_dirs,
                    target_dirs,
                    target_vert,
                    loss_factors,
                    filtered_seg=None,
                    pixel_gt_count=None,
                    kp_loss=kp_loss,
                )

            if pose_estimation:
                if opt.estimate_coords:
                    pose_stats, _, _ = evaluate_pose_estimates(
                        points_est,
                        poses_est,
                        poses_gt,
                        target_seg,
                        keypoints,
                        cam_mat,
                        diameters,
                        offsets,
                        min_num=200,
                    )
                else:
                    pose_stats, _, _ = estimate_and_evaluate_poses(
                        output_seg,
                        target_seg,
                        output_dirs,
                        poses_gt,
                        keypoints,
                        cam_mat,
                        diameters,
                        offsets,
                        min_num=200,
                    )
                loss.append(pose_stats)
            return loss

        @tf.function
        def distributed_train_step(dataset_inputs, loss_factors):
            per_replica_losses = mirrored_strategy.run(
                train_step,
                args=(
                    dataset_inputs,
                    loss_factors,
                    True,
                    False,
                ),
            )
            l0 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[0], axis=None)
            l1 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[1], axis=None)
            l2 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[2], axis=None)
            l3 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[3], axis=None)
            l4 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[4], axis=None)
            return [l0, l1, l2, l3, l4]

        @tf.function
        def distributed_test_step(dataset_inputs, loss_factors):
            per_replica_losses = mirrored_strategy.run(
                train_step,
                args=(
                    dataset_inputs,
                    loss_factors,
                    False,
                    False,
                ),
            )
            l0 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[0], axis=None)
            l1 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[1], axis=None)
            l2 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[2], axis=None)
            l3 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[3], axis=None)
            l4 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[4], axis=None)

            return [l0, l1, l2, l3, l4]

        @tf.function
        def distributed_test_pose_step(dataset_inputs, loss_factors):
            per_replica_losses = mirrored_strategy.run(
                train_step,
                args=(
                    dataset_inputs,
                    loss_factors,
                    False,
                    True,
                ),
            )
            l0 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[0], axis=None)
            l1 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[1], axis=None)
            l2 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[2], axis=None)
            l3 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[3], axis=None)
            l4 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[4], axis=None)
            l_pose_2d = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[5][0], axis=None)
            l_pose_3d = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[5][1], axis=None)
            l_pose_gt = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[5][2], axis=None)
            l_pose_fp = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[5][3], axis=None)
            err_2d = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[5][4], axis=None)
            err_3d = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[5][5], axis=None)
            return [
                l0,
                l1,
                l2,
                l3,
                l4,
                l_pose_2d,
                l_pose_3d,
                l_pose_gt,
                l_pose_fp,
                err_2d,
                err_3d,
            ]

        lr = optimizer._decayed_lr(tf.float32)
        epoch_loss = tf.zeros([5], dtype=tf.float32)
        if pose_validation:
            epoch_pose_2d_count = tf.zeros([no_objects], dtype=tf.float32)
            epoch_pose_3d_count = tf.zeros([no_objects], dtype=tf.float32)
            epoch_pose_count_gt = tf.zeros([no_objects], dtype=tf.float32)
            epoch_pose_count_fp = tf.zeros([no_objects], dtype=tf.float32)
            epoch_pose_err_2d = tf.zeros([no_objects], dtype=tf.float32)
            epoch_pose_err_3d = tf.zeros([no_objects], dtype=tf.float32)

        start = time.time()

        for batch_idx in range(batches_per_epoch):
            img_batch = loader_iterator.get_next()
            if train:
                loss = distributed_train_step(img_batch, loss_factors)
            elif pose_validation:
                loss = distributed_test_pose_step(img_batch, loss_factors)
                epoch_pose_2d_count += loss[5]
                epoch_pose_3d_count += loss[6]
                epoch_pose_count_gt += loss[7]
                epoch_pose_count_fp += loss[8]
                epoch_pose_err_2d += loss[9]
                epoch_pose_err_3d += loss[10]
            else:
                loss = distributed_test_step(img_batch, loss_factors)

            epoch_loss += loss[0:5]

            if train:
                namefile = "/loss_train.csv"
            else:
                namefile = "/loss_test.csv"

            with open(opt.outf + namefile, "a") as file:
                s = "{}, {},{:.15f},{:.7f},{:.7f},{:.7f},{:.7f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    epoch,
                    batch_idx + 1,
                    loss[0],
                    loss[1],
                    loss[2],
                    loss[3],
                    loss[4],
                    loss_factors.mask_loss_weight,
                    loss_factors.vertex_loss_weight,
                    loss_factors.proxy_loss_weight,
                    loss_factors.kp_loss_weight,
                )
                file.write(s)

            if train:
                if (batch_idx + 1) % opt.loginterval == 0:
                    tf.print(
                        "{}  Train Epoch: {}, Batch idx: {}, Loss: {:.15f}, Epoch Loss: {:.15f}\n".format(
                            datetime.datetime.now().time(),
                            epoch,
                            batch_idx + 1,
                            loss[0],
                            epoch_loss[0] / (batch_idx + 1),
                        )
                    )
                    tf.print("Time {}".format(time.time() - start))

            else:
                if (batch_idx + 1) % opt.loginterval == 0:
                    tf.print(
                        "{}  Test Epoch: {}, Batch idx: {}, Loss: {:.15f}, Epoch Loss: {:.15f}\n".format(
                            datetime.datetime.now().time(),
                            epoch,
                            batch_idx + 1,
                            loss[0],
                            epoch_loss[0] / (batch_idx + 1),
                        )
                    )
                    tf.print("Time {}".format(time.time() - start))

            start = time.time()

        epoch_loss /= batches_per_epoch

        if pose_validation:
            err_2d = tf.math.divide_no_nan(epoch_pose_2d_count, epoch_pose_count_gt)
            err_3d = tf.math.divide_no_nan(epoch_pose_3d_count, epoch_pose_count_gt)

        tf.print("==========================")
        if train:
            tf.print(
                "== TRAINING == Finished epoch {} (lr={:.7f}) with total loss: {:.7f} --- mask: {:.7f}, vector: {:.7f}, proxy: {:.7f}, keypoint: {:.7f} ==".format(
                    epoch,
                    lr,
                    epoch_loss[0],
                    epoch_loss[1],
                    epoch_loss[2],
                    epoch_loss[3],
                    epoch_loss[4],
                )
            )
        else:
            tf.print(
                "== VALIDATION == Finished epoch {} with total loss: {:.7f} --- mask: {:.7f}, vector: {:.7f}, proxy: {:.7f}, keypoint: {:.7f} ==".format(
                    epoch,
                    epoch_loss[0],
                    epoch_loss[1],
                    epoch_loss[2],
                    epoch_loss[3],
                    epoch_loss[4],
                )
            )
            if pose_validation:
                tf.print("2D Valid: {}".format(err_2d), summarize=-1)
                tf.print("2D Valid (mean): {}".format(tf.reduce_mean(err_2d)), summarize=-1)

                tf.print("3D Valid: {}".format(err_3d), summarize=-1)
                tf.print("3D Valid (mean): {}".format(tf.reduce_mean(err_3d)), summarize=-1)

                tf.print(
                    "Err 2D: {}".format(tf.math.divide_no_nan(epoch_pose_err_2d, epoch_pose_count_gt)),
                    summarize=-1,
                )

        loss_factors.print()
        tf.print("==========================")

        # write summary
        if train:
            namefile = "/train_summary.csv"
        else:
            namefile = "/test_summary.csv"

        with open(opt.outf + namefile, "a") as file:

            s = "{},{},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f}".format(
                epoch,
                lr,
                epoch_loss[0],
                epoch_loss[1],
                epoch_loss[2],
                epoch_loss[3],
                epoch_loss[4],
            )
            if pose_validation:
                for i in range(no_objects):
                    s += ",{:.4f}".format(err_2d[i])
                for i in range(no_objects):
                    s += ",{:.4f}".format(err_3d[i])
            s += "\n"
            file.write(s)

        if epoch % opt.saveinterval == 0 and train:
            checkpoint.save(file_prefix=checkpoint_prefix)
            tf.print("\nSave results weights as h5...\n")
            net.save_weights(frozen_path + "/result_w.h5")

        return loader_iterator


tf.print("Batches per epoch: {} Epochs: {} : ".format(train_batches, opt.epochs))
tf.print("Test Batches per epoch: {} Epochs: {} : ".format(test_batches, opt.epochs))

if trainingdata is not None:
    trainingdata_iterator = iter(trainingdata)
if testingdata is not None:
    testingdata_iterator = iter(testingdata)

for epoch in range(1, opt.epochs + 1):
    if trainingdata is not None:
        trainingdata_iterator = runnetwork(
            trainingdata_iterator,
            int(train_batches),
            epoch,
            checkpoint,
            loss_factors,
            train=True,
            pose_validation=False,
        )
    if testingdata is not None:
        testingdata_iterator = runnetwork(
            testingdata_iterator,
            int(test_batches),
            epoch,
            checkpoint,
            loss_factors,
            train=False,
            pose_validation=(epoch % opt.validationinterval == 0),
        )

checkpoint.save(file_prefix=checkpoint_prefix)

print("end:", datetime.datetime.now().time())
