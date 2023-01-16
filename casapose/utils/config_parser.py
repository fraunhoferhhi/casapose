import argparse
import configparser

import numpy as np


def parse_config():
    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    conf_parser.add_argument("-c", "--config", help="Specify config file", metavar="FILE")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="", help="path to training data")
    parser.add_argument("--data_path_filter", default=None, help="a list of allowed direct subsolders for the data folder")
    parser.add_argument("--datatest", default="", help="path to data testing/validation set")
    parser.add_argument("--datatest_path_filter", default=None, help="a list of allowed direct subsolders for the datatest folder")
    parser.add_argument("--color_dataset", type=str2bool, default=True, help="is true if dataset is rgb")
    parser.add_argument("--data_wxyz_quaterion", type=str2bool, default=False, help=" data has wxyz quaternion format")
    parser.add_argument("--datatest_wxyz_quaterion", type=str2bool, default=False, help=" datatest has wxyz quaternion format")

    parser.add_argument("--datameshes", default="", help="path to meshes from dataset")
    parser.add_argument("--modelname", default="casapose_cond_weighted", help="name of the model to use")
    parser.add_argument("--backbonename", default="resnet18", help="name of the backbone to use (currently only resnet18)")
    parser.add_argument("--train_validation_split", type=float, default=0.9, help="train validation split")
    parser.add_argument("--estimate_confidence", type=str2bool, default=False, help="netork estimates confidence map (adds no_points output maps)")
    parser.add_argument("--estimate_coords", type=str2bool, default=False, help="netork estimates coords via reprojection and bpnp")
    parser.add_argument("--confidence_regularization", type=str2bool, default=False, help="added loss regularization to ensure that the estimates do not get too small")
    parser.add_argument("--confidence_filter_estimates", type=str2bool, default=True, help="apply connected component ananlysis and choose largest")
    parser.add_argument("--confidence_choose_second", type=str2bool, default=False, help="choose second largest component during testing")

    parser.add_argument("--mask_loss_weight", type=float, default=1.0, help="mask loss weight")
    parser.add_argument("--vertex_loss_weight", type=float, default=0.5, help="vertex loss weight")
    parser.add_argument("--proxy_loss_weight", type=float, default=0.013, help="proxy loss weight")
    parser.add_argument("--keypoint_loss_weight", type=float, default=0.0, help="keypoint loss weight")
    parser.add_argument("--filter_vertex_with_segmentation", type=str2bool, default=False, help="only calculate proxy and vertex error in regions, wehere the segmentation was estimated correctly")
    parser.add_argument("--filter_high_proxy_errors", type=str2bool, default=False, help="ignore objects with high proxy error in training")
    parser.add_argument("--use_bpnp_reprojection_loss", type=str2bool, default=False, help="calculate error on reprojected points")
    parser.add_argument("--max_keypoint_pixel_error", type=float, default=25.0, help="if the reprojection error is larger than 25.0 pixel the influence is reduced")

    parser.add_argument("--object", default=None, help="which object in the dataset is of interest")
    parser.add_argument("--no_points", type=int, default=9, help="number of keypoints to find")

    parser.add_argument("--workers", type=int, default=1, help="number of data loading workers")
    parser.add_argument("--prefetch", type=int, default=0, help="size of prefetch buffer")
    parser.add_argument("--pretrained", type=str2bool, default=True, help="do you want to use resnet-18 imagenet pretrained weights")
    parser.add_argument("--batchsize", type=int, default=32, help="input batch size")
    parser.add_argument("--imagesize", nargs="+", type=int, default=[448], help="the height / width of the input image to network")
    parser.add_argument("--imagesize_test", nargs="+", type=int, default=[448], help="the height / width of the input image to network in evaluation")

    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate, default=0.001")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="learning rate decay")
    parser.add_argument("--lr_epochs", type=int, default=15, help="apply decay every n epochs")
    parser.add_argument("--lr_epochs_start", type=int, default=0, help="the initial learning rate is kept for n epochs, then decay starts")
    parser.add_argument("--lr_epochs_steps", default=None, help="a list of int, epochs where the lr is decayed")
    parser.add_argument("--noise", type=float, default=0.0, help="gaussian noise added to the image")
    parser.add_argument("--contrast", type=float, default=0.4, help="contrast manipulation during training")
    parser.add_argument("--brightness", type=float, default=0.2, help="brightness manipulation during training")
    parser.add_argument("--saturation", type=float, default=0.001, help="saturation manipulation during training")
    parser.add_argument("--hue", type=float, default=0.001, help="hue manipulation during training")
    parser.add_argument("--use_imgaug", type=str2bool, default=False, help="use advanced augmentation_model with imgaug")
    parser.add_argument("--rotation", type=float, default=15, help="rotation manipulation during training")
    parser.add_argument("--translation", type=float, default=25, help="translation manipulation during training")
    parser.add_argument("--crop_factor", type=float, default=1.0, help="factor of crop of input image along image height, image will be resized to imagesize later")
    parser.add_argument("--epochs", type=int, default=60, help="number of epochs to train")
    parser.add_argument("--loginterval", type=int, default=100, help="logging interval")
    parser.add_argument("--saveinterval", type=int, default=10, help="interval of epochs to save")
    parser.add_argument("--validationinterval", type=int, default=1, help="interval of epochs for pose evaluation during training")
    parser.add_argument("--save_debug_batch", type=str2bool, default=False, help="save debug batch and exit (training)")
    parser.add_argument("--save_eval_batches", type=str2bool, default=False, help="save eval batchs")
    parser.add_argument("--write_poses", type=str2bool, default=False, help="write poses for bop evaluation")
    parser.add_argument("--filter_test_with_gt", type=str2bool, default=False, help="do not consider objects which are not in gt")
    parser.add_argument("--min_object_size_test", type=int, default=1, help="min size of objects to be detected")

    parser.add_argument("--net", default="./output/training_checkpoints", help="path to net (to continue training)")

    parser.add_argument("--manualseed", type=int, help="manual seed")
    parser.add_argument("--outf", default="tmp", help="folder to output images and model checkpoints, it will \add a train_ in front of the name")
    parser.add_argument("--evalf", default="", help="folder to store eval logs")
    parser.add_argument("--gpuids", nargs="+", type=int, default=[0], help="GPUs to use")

    parser.add_argument("--train_vectors_with_ground_truth", type=str2bool, default=False, help="use ground truth segmentation for CLADE training")
    parser.add_argument("--load_h5_weights", type=str2bool, default=False, help="load h5 weights")
    parser.add_argument("--load_h5_filename", default="result_w", help="filename of h5 file (without extension)")

    parser.add_argument("--copy_weights_from_backup_network", type=str2bool, default=False, help="copy semantic segmentation and clade from an existing network, which should be expanded")
    parser.add_argument("--copy_weights_add_confidence_maps", type=str2bool, default=False, help="use old model without confidence maps and add them")
    parser.add_argument("--objects_to_copy", type=int, default=0, help="the first n objects are copied to the new network, which then can be extended")
    parser.add_argument("--objects_in_input_network", type=int, default=0, help="number of objects in input network to copy from")
    parser.add_argument("--objects_to_copy_list", default="", help="a csv file specifying which objects to copy to which index")
    # fmt: on

    # Read the config but do not overwrite the args written
    args, remaining_argv = conf_parser.parse_known_args()
    defaults = {}

    if args.config:
        config = configparser.SafeConfigParser(allow_no_value=True)
        config.read([args.config])
        defaults.update(dict(config.items("defaults")))
        if "gpuids" in defaults:
            defaults["gpuids"] = [int(t) for t in defaults["gpuids"].split(",")]
        if "imagesize" in defaults:
            defaults["imagesize"] = [int(t) for t in defaults["imagesize"].split(",")]
        if "imagesize_test" in defaults:
            defaults["imagesize_test"] = [int(t) for t in defaults["imagesize_test"].split(",")]

    parser.set_defaults(**defaults)
    opt = parser.parse_args(remaining_argv)

    if len(opt.imagesize) == 1:
        opt.imagesize = (opt.imagesize[0], opt.imagesize[0])
    else:
        opt.imagesize = (opt.imagesize[0], opt.imagesize[1])

    if len(opt.imagesize_test) == 1:
        opt.imagesize_test = (opt.imagesize_test[0], opt.imagesize_test[0])
    else:
        opt.imagesize_test = (opt.imagesize_test[0], opt.imagesize_test[1])

    def split_string(val):
        if val is not None:
            return [x.strip() for x in val.split(",")]
        return None

    opt.data_path_filter = split_string(opt.data_path_filter)
    opt.datatest_path_filter = split_string(opt.datatest_path_filter)

    if opt.lr_epochs_steps is not None:
        opt.lr_epochs_steps = [int(x) for x in split_string(opt.lr_epochs_steps)]

    if opt.objects_to_copy_list == "":
        opt.objects_to_copy = np.array(
            [range(opt.objects_to_copy + 1), range(opt.objects_to_copy + 1)], np.int32
        ).transpose()
    else:
        opt.objects_to_copy = np.array(np.genfromtxt(opt.objects_to_copy_list, delimiter=","), np.int32)
        opt.objects_to_copy = np.concatenate((np.array([[0, 0]], np.int32), opt.objects_to_copy))  # add background

    if opt.objects_in_input_network == 0:
        opt.objects_in_input_network = opt.objects_to_copy.shape[0] - 1

    if opt.pretrained in ["false", "False"]:
        opt.pretrained = False

    if opt.evalf == "":
        opt.evalf = opt.outf
    if "/" not in opt.outf:
        opt.outf = "output/{}".format(opt.outf)

    if "/" not in opt.evalf:
        opt.evalf = opt.outf + "/" + opt.evalf

    if opt.manualseed is None:
        opt.manualseed = np.random.randint(1, 10000)

    return opt
