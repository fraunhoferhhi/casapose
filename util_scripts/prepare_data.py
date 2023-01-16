import argparse
import errno
import glob
import os
import shutil
import sys

import wget

sys.path.extend([".", ".."])  # adds the folder from which you call the script

from zipfile import ZipFile

from dataset_converter import generate_data

BASE_URL = r"https://bop.felk.cvut.cz/media/data/bop_datasets/"


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else:
            raise


def copydir(source, dest):
    """Copy a directory structure overwriting existing files"""
    for root, dirs, files in os.walk(source):
        if not os.path.isdir(root):
            os.makedirs(root)
        for each_file in files:
            rel_path = root.replace(source, "").lstrip(os.sep)
            dest_path = os.path.join(dest, rel_path, each_file)
            shutil.copyfile(os.path.join(root, each_file), dest_path)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--download_path", required=True, help="path to where to download the dataset")
parser.add_argument("-lm", "--gen_lm", action="store_true", help="generate lm test data")
parser.add_argument("-lmo", "--gen_lmo", action="store_true", help="generate lmo test data")
parser.add_argument("-pbr", "--gen_train", action="store_true", help="generate lm/lmo pbr train data")
parser.add_argument("-bop", "--gen_bop", action="store_true", help="generate and download bop subset for lm or lmo")
parser.add_argument("-hb", "--gen_hb", action="store_true", help="generate hb testdata")
parser.add_argument("-c", "--cleanup", action="store_true", help="delete temporary zip files after extraction")

args = parser.parse_args()

tmp_path = os.path.join(args.download_path, "tmp")
out_path = args.download_path

lm_path = os.path.join(tmp_path, "lm")
lmo_path = os.path.join(tmp_path, "lmo")
hb_path = os.path.join(tmp_path, "hb")
lm_path_out = os.path.join(out_path, "lm")
lmo_path_out = os.path.join(out_path, "lmo")
hb_path_out = os.path.join(out_path, "hb")


if os.path.exists(lm_path) and (args.gen_lm and args.gen_train):
    shutil.rmtree(lm_path)
if os.path.exists(lmo_path) and args.gen_lmo:
    shutil.rmtree(lmo_path)
if os.path.exists(hb_path) and args.gen_hb:
    shutil.rmtree(hb_path)

if not os.name == "nt":
    os.environ["PYOPENGL_PLATFORM"] = "egl"

download_filenames = {"lm": ["lm_base.zip", "lm_models.zip"]}

if args.gen_hb:
    download_filenames["hb"] = [
        "hb_base.zip",
        "hb_models.zip",
        "hb_val_primesense.zip",  # 000002
        "hb_val_kinect.zip",  # 000002
    ]
    hb_models = {
        "obj_000002.ply": "obj_000002.ply",
        "obj_000007.ply": "obj_000008.ply",
        "obj_000021.ply": "obj_000015.ply",
    }

if args.gen_lmo:
    download_filenames["lmo"] = ["lmo_base.zip", "lmo_test_all.zip"]
    if args.gen_bop:
        download_filenames["lmo"].append("lmo_test_bop19.zip")

if args.gen_lm:
    if args.gen_bop:
        download_filenames["lm"].append("lm_test_bop19.zip")
    download_filenames["lm"].append("lm_test_all.zip")

if args.gen_train:
    download_filenames["lm"].append("lm_train_pbr.zip")


if not os.path.exists(os.path.join(tmp_path)):
    os.makedirs(tmp_path)

if not os.path.exists(os.path.join(out_path)):
    os.makedirs(out_path)

if args.gen_lmo and not os.path.exists(lmo_path_out):
    os.makedirs(lmo_path_out)
if (args.gen_lm or args.gen_train) and not os.path.exists(lm_path_out):
    os.makedirs(lm_path_out)
if args.gen_hb and not os.path.exists(hb_path_out):
    os.makedirs(hb_path_out)

# download
for dataset in download_filenames:
    for idx, filename in enumerate(download_filenames[dataset]):
        if not os.path.exists(os.path.join(tmp_path, filename)):
            print(filename)
            wget.download(BASE_URL + filename, out=tmp_path)
# unzip
for dataset in download_filenames:
    for idx, filename in enumerate(download_filenames[dataset]):
        tmp_dataset_path = tmp_path
        if idx != 0:
            tmp_dataset_path = os.path.join(tmp_path, dataset)

        with ZipFile(os.path.join(tmp_path, filename), "r") as zip_ref:
            if "models" in filename:
                for model_file in zip_ref.namelist():
                    if "eval" in model_file:
                        if dataset == "hb":
                            for extraction_file in hb_models:
                                if extraction_file in model_file:
                                    zip_ref.extract(model_file, tmp_dataset_path)
                                    path_out = os.path.join(tmp_dataset_path, model_file)
                                    os.rename(path_out, path_out.replace(extraction_file, hb_models[extraction_file]))
                        else:
                            zip_ref.extract(model_file, tmp_dataset_path)
            elif dataset == "hb" and "val" in filename:
                for model_file in zip_ref.namelist():
                    if r"/000002/" in model_file:
                        zip_ref.extract(model_file, tmp_dataset_path)
            # elif dataset == "lm" and "pbr" in filename:
            #     for model_file in zip_ref.namelist():
            #         if r"/000001/" in model_file or r"/000049/" in model_file:
            #             zip_ref.extract(model_file, tmp_dataset_path)
            else:
                zip_ref.extractall(tmp_dataset_path)

            if "bop19" in filename:
                os.rename(os.path.join(tmp_path, dataset + "/test"), os.path.join(tmp_path, dataset + "/test_bop"))

        if args.cleanup:
            os.remove(os.path.join(tmp_path, filename))

# modify dirs
if args.gen_hb:
    os.rename(os.path.join(tmp_path, "hb/val_primesense"), os.path.join(tmp_path, "hb/test_primesense"))
    os.rename(os.path.join(tmp_path, "hb/val_kinect"), os.path.join(tmp_path, "hb/test_kinect"))
    files = sorted(glob.glob(tmp_path + "/lm/models_eval/" + "*.ply"))
    for scr_file in files:
        dst = scr_file.replace("lm/models_eval", "hb/models_eval")
        if not os.path.exists(dst):
            shutil.copyfile(scr_file, dst)

if args.gen_train:
    os.makedirs(os.path.join(tmp_path, "lm/val_pbr"))
    copyanything(os.path.join(tmp_path, "lm/train_pbr/000049"), os.path.join(tmp_path, "lm/val_pbr/000049"))
    shutil.rmtree(os.path.join(tmp_path, "lm/train_pbr/000049"))

# overwrite with prepared data
if args.gen_hb:
    copydir("data/datasets/hb", hb_path)

copydir("data/datasets/lm", lm_path)


settings = {}
settings["near"] = 100
settings["far"] = 2000
settings["width"] = 640
settings["height"] = 480
settings["filetype_in"] = "png"

settings["mask"] = "reuse"  # can also be "render" or sth. elso to ignore segmentation
settings["draw_debug_image"] = False

if args.gen_lmo:
    settings["copy_meshes"] = True
    generate_data(lmo_path, lmo_path_out, settings, model_folder="../lm/models_eval", image_folder="test")
    if args.gen_bop:
        settings["copy_meshes"] = False
        generate_data(lmo_path, lmo_path_out, settings, model_folder="../lm/models_eval", image_folder="test_bop")

if args.gen_lm:
    settings["copy_meshes"] = True
    generate_data(lm_path, lm_path_out, settings, model_folder="models_eval", image_folder="test")
    if args.gen_bop:
        settings["copy_meshes"] = False
        generate_data(lm_path, lm_path_out, settings, model_folder="models_eval", image_folder="test_bop")

if args.gen_train:
    settings["copy_meshes"] = True
    settings["filetype_in"] = "jpg"
    generate_data(lm_path, lm_path_out, settings, model_folder="models_eval", image_folder="train_pbr")
    settings["copy_meshes"] = False
    generate_data(lm_path, lm_path_out, settings, model_folder="models_eval", image_folder="val_pbr")

if args.gen_hb:
    settings["filetype_in"] = "png"
    settings["copy_meshes"] = True
    settings["far"] = 2500
    generate_data(hb_path, hb_path_out, settings, model_folder="models_eval", image_folder="test_primesense")
    settings["width"] = 1920
    settings["height"] = 1080
    settings["copy_meshes"] = False
    generate_data(hb_path, hb_path_out, settings, model_folder="models_eval", image_folder="test_kinect")
