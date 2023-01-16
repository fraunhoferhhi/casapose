from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.extend([".", ".."])  # adds the folder from which you call the script
os.environ["CASAPOSE_INFERENCE"] = "True"

from casapose.data_handler.image_only_dataset import ImageOnlyDataset
from casapose.data_handler.vectorfield_dataset import VectorfieldDataset
from casapose.pose_estimation.pose_evaluation import poses_pnp
from casapose.pose_estimation.voting_layers_2d import CoordLSVotingWeighted
from casapose.pose_models.tfkeras import Classifiers
from casapose.utils.config_parser import parse_config

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

opt = parse_config()

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

test_dataset = ImageOnlyDataset(root=opt.datatest)
testing_images, _ = test_dataset.generate_dataset(1)

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
)
print(len(test_dataset))
testingdata, test_batches = test_dataset.generate_dataset(
    1, 1, 0, opt.imagesize_test, 1.0, 1, no_objects, shuffle=False
)

mesh_vertex_array, mesh_vertex_count = test_dataset.generate_object_vertex_array()

print("testing data: {} batches".format(test_batches))


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
    net.load_weights(frozen_path + "/" + opt.load_h5_filename + ".h5", by_name=True, skip_mismatch=True)

elif opt.net != "":
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

for layer in net.layers:
    layer.trainable = False

net.summary()

with open(opt.evalf + "/speed_eval.csv", "w") as file:
    file.write("batchid,speed \n")


def runnetwork(loader_iterator, batches, keypoints, camera_matrix, no_objects):
    @tf.function
    def test_step(img_batch):
        start_time = tf.timestamp()
        no_points = opt.no_points
        net_input = [tf.expand_dims(img_batch[0], 0)]
        output_net = net(net_input, training=False)  # all stages are present here
        output_seg, output_dirs, confidence = tf.split(output_net, [no_objects, no_points * 2, -1], 3)
        coordLSV_in = [output_seg, output_dirs, confidence]
        coords = CoordLSVotingWeighted(
            name="coords_ls_voting",
            num_classes=no_objects,
            num_points=no_points,
            filter_estimates=True,
        )(coordLSV_in)
        poses_est = poses_pnp(
            coords, output_seg, keypoints, camera_matrix, no_objects - 1, min_num=opt.min_object_size_test
        )
        tf.print(poses_est[0, 0, 0, 0, 0])  # access memory to get sure everything is done

        end_time = tf.timestamp()
        time_needed = end_time - start_time
        return time_needed

    speed = []
    for batch_idx in range(batches):
        img_batch = loader_iterator.get_next()
        time_needed = test_step(img_batch)
        speed.append(time_needed)
        with open(opt.evalf + "/speed_eval.csv", "a") as file:
            s = "{},{:.7f}\n".format(batch_idx + 1, time_needed)
            file.write(s)

    tf.print("average speed: {}".format(tf.reduce_mean(speed[10:])), summarize=-1)

    return loader_iterator


print("Test Batches: {} ".format(test_batches))

testingdata_iterator = iter(testingdata)
img_batch = testingdata_iterator.get_next()

keypoints = img_batch[4]
camera_matrix = img_batch[5]

testingdata_iterator = iter(testing_images)
runnetwork(testingdata_iterator, int(test_batches), keypoints, camera_matrix, no_objects + 1)
