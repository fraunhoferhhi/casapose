import glob
import json
import os
from itertools import compress
from os.path import exists

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import trimesh
import yaml
from casapose.data_handler.augmentation_model import seq  # noqa: E402
from casapose.data_handler.augmentation_model import seq_grayscale  # noqa: E402
from casapose.utils.dataset_utils import load_split, save_batches  # noqa: E402
from casapose.utils.geometry_utils import get_rotation_matrix_2D  # noqa: E402
from casapose.utils.geometry_utils import (  # noqa: E402
    quaternion_matrix,
    reproject,
    transform_points,
)
from casapose.utils.image_utils import add_noise  # noqa: E402


class VectorfieldDataset:
    def __init__(
        self,
        root,
        path_meshes,
        no_points=9,
        color_input=False,
        normal=[0.5, 0.5],
        test=False,
        objectsofinterest=[],
        save=False,
        noise=2,
        data_size=None,
        random_translation=(25.0, 25.0),
        random_rotation=15.0,
        random_crop=True,
        contrast=0.2,
        brightness=0.2,
        hue=0.05,
        saturation=0.2,
        use_train_split=False,
        use_validation_split=False,
        train_validation_split=0.9,
        output_folder="",
        use_imgaug=False,
        visibility_filter=False,
        separated_vectorfields=False,
        wxyz_quaterion_input=False,  # dataset has wrong quaternion format (only for test on server use fix_dataset.py instead)
        path_filter_root=None,
    ):
        ###################
        self.path_meshes = path_meshes
        self.no_points = no_points
        self.color_input = color_input
        self.normal = normal
        self.test = test
        self.objectsofinterest = objectsofinterest
        self.save = save
        self.noise = noise
        self.data_size = data_size
        self.random_translation = random_translation
        self.random_rotation = random_rotation
        self.random_crop = random_crop
        self.contrast = contrast
        self.brightness = brightness
        self.hue = hue
        self.saturation = saturation
        self.use_train_split = use_train_split
        self.use_validation_split = use_validation_split
        self.train_validation_split = train_validation_split
        self.output_folder = output_folder
        self.use_imgaug = use_imgaug
        self.visibility_filter = visibility_filter
        self.separated_vectorfields = separated_vectorfields
        self.wxyz_quaterion_input = wxyz_quaterion_input
        self.imgs = []

        # second dataset can be used if a special distribution between a new and a known object should be used
        def load_data(
            path,
            path_meshes,
            path_filer=None,
        ):
            """Load the meshes and the recursively search for images in the subfolders."""
            imgs = []
            class_labels = {}
            fixed_transformations = {}

            camera_data = {}
            meshes = self.load_meshes(path_meshes)
            # Check all the folders in path
            for name in os.listdir(str(path)):
                if path_filer is None or name in path_filer:
                    print(path + "/" + name)
                    (
                        img_tmp,
                        class_labels_tmp,
                        fixed_transformations_tmp,
                        camera_data_tmp,
                    ) = self.load_image_data(path + "/" + name)
                    imgs += img_tmp
                    class_labels.update(class_labels_tmp)
                    camera_data.update(camera_data_tmp)
                    fixed_transformations.update(fixed_transformations_tmp)

            return (
                imgs,
                class_labels,
                camera_data,
                fixed_transformations,
                meshes,
            )

        (self.imgs, self.class_labels, self.camera_data, self.fixed_transformations, self.meshes,) = load_data(
            root, path_meshes, path_filter_root
        )  # creates a list of json files

    def __len__(self):
        # When limiting the number of data
        if self.data_size is not None:
            return int(self.data_size)

        return len(self.imgs)

    # always picks from first dataset
    def __getitem__(self, index):
        path, name, txt, seg, path_raw = self.imgs[index]
        return {
            "path": path,
            "name": name,
            "txt": txt,
            "seg": seg,
            "path_raw": path_raw,
        }

    def tf_random_augmentations(
        self,
        img,
        seg,
        points,
        points3d,
        camera_data,
        diameters,
        off,
        affine,
        cuboid3d,
        transform_mats,
        pixel_gt_count,
        image_id,
        new_labels,
    ):
        def augment_image(image):
            if self.color_input:
                return seq(images=image.numpy())
            else:
                return seq_grayscale(images=image.numpy())

        [
            img,
        ] = tf.py_function(augment_image, [img], [tf.uint8])

        return (
            img,
            seg,
            points,
            points3d,
            camera_data,
            diameters,
            off,
            affine,
            cuboid3d,
            transform_mats,
            pixel_gt_count,
            image_id,
            new_labels,
        )

    @tf.function
    def image_transformation(
        self,
        img,
        seg,
        points,
        points3d,
        camera_data,
        diameters,
        off,
        affine,
        cuboid3d,
        transform_mats,
        pixel_gt_count,
        image_id,
        new_labels,
        img_size_out,
    ):
        @tf.function
        def crop_image(img, crop):
            return tf.image.crop_to_bounding_box(img, crop[0], crop[1], crop[2], crop[3])

        img = tfa.image.transform(img, affine, "BILINEAR")
        seg = tfa.image.transform(seg, affine)

        fn = lambda x: crop_image(x[0], x[1])
        elems = (img, tf.cast(off, tf.int32))
        img = tf.map_fn(fn, elems=elems, dtype=tf.uint8)

        elems = (seg, tf.cast(off, tf.int32))
        seg = tf.map_fn(fn, elems=elems, dtype=tf.uint8)

        return (
            img,
            seg,
            points,
            points3d,
            camera_data,
            diameters,
            off,
            affine,
            cuboid3d,
            transform_mats,
            pixel_gt_count,
            image_id,
            new_labels,
        )

    @tf.function
    def image_augmentation(
        self,
        img,
        seg,
        points,
        points3d,
        camera_data,
        diameters,
        off,
        affine,
        cuboid3d,
        transform_mats,
        pixel_gt_count,
        image_id,
        new_labels,
        img_size_out,
        input_size,
    ):
        img.set_shape([None, input_size[0], input_size[1], 3 if self.color_input else 1])
        seg = tf.image.resize(seg, img_size_out, method="nearest")
        img = tf.image.resize(img, img_size_out)

        (mask, dir_maps, seg,) = self.generate_segmentation_and_direction_maps_batch_v2(
            seg,
            points,
            new_labels,
            separated_vectorfields=self.separated_vectorfields,
        )

        # _______ augmentation ____________#

        if self.use_imgaug is False:
            if self.color_input:
                img = tf.image.random_hue(img, self.hue)
                img = tf.image.random_saturation(img, 1 - self.saturation, 1 + self.saturation)
            img = tf.image.random_brightness(img, self.brightness)
            img = tf.image.random_contrast(img, 1 - self.contrast, 1 + self.contrast)

        # ________end augmentation________#

        img = ((img / 255) - self.normal[0]) / self.normal[1]
        img = add_noise(img, self.noise)
        if self.color_input is False:
            img = tf.repeat(img, 3, axis=-1)  # for grayscale images this can be the very last operation

        return (
            img,
            mask,
            dir_maps,
            points,
            points3d,
            camera_data,
            diameters,
            off,
            seg,
            cuboid3d,
            transform_mats,
            pixel_gt_count,
            image_id,
        )

    def apply_preprocessing(
        self,
        img,
        name,
        txt,
        seg_img,
        path_raw,
        imagesize,
        cropratio,
        max_instance_count,
        no_points,
    ):
        # start = time.time()

        data = self.load_json_minimal(txt)
        # generate identifier
        p = path_raw.decode("utf-8")
        p = os.path.normpath(p.replace("\\", "/")).split(os.sep)
        image_id = p[-2] + "_" + p[-1] + "_" + os.path.splitext(name)[0].decode("utf-8")

        class_labels = self.class_labels[path_raw.decode("utf-8")]
        camera_data = self.camera_data[path_raw.decode("utf-8")]
        fixed_transformations = self.fixed_transformations[path_raw.decode("utf-8")]

        keypoints2d_all = data["keypoints2d"]  # points in image

        poses_loc_all = data["poses_loc"]  # points in image
        poses_quaternion_all = data["poses_quaternions"]  # points in image
        object_classes = data["objectClasses"]  # objects in image

        # ____PREPARATION_______
        width = img.shape[1]
        height = img.shape[0]
        img_size_orig = (width, height)  # size (width first)
        crop_height = round(float(height) * cropratio)
        crop_width = crop_height * (float(imagesize[1]) / float(imagesize[0]))

        img_size_out = (int(crop_height), int(crop_width))  # (height,height)
        # size output and cropsize (height first)
        scale = imagesize[0] / img_size_out[0]
        px_count_all = [int((float(i) * scale) + 0.5) for i in data["px_count_all"]]

        if self.random_crop:
            w_crop = np.random.randint(0, img_size_orig[0] - img_size_out[1] + 1)
            h_crop = np.random.randint(0, img_size_orig[1] - img_size_out[0] + 1)
        else:
            w_crop = int((img_size_orig[0] - img_size_out[1]) / 2)
            h_crop = int((img_size_orig[1] - img_size_out[0]) / 2)

        keypoints2d = []
        keypoints3d = []
        cuboid3d = []
        transform_mats_batch = []

        pixel_gt_batch = []

        object_labels = []
        diameters = []
        # collect points
        for objectofinterest in self.objectsofinterest:  # check all objects of interest
            points2d = []
            points3d = []
            cuboid_points3d = []
            transform_mats = []
            pixel_gt = []
            diameter = []
            transformed_points = []
            label = None

            keypoints3d_mesh = self.meshes[objectofinterest]["keypoints"]

            cuboid_points3d_mesh = self.meshes[objectofinterest]["volume"]

            # do not transform points if no fixed transformation is provided
            if objectofinterest in fixed_transformations:
                transformed_points = np.array(
                    transform_points(keypoints3d_mesh, fixed_transformations[objectofinterest])
                )
                transformed_cuboid_points = np.array(
                    transform_points(cuboid_points3d_mesh, fixed_transformations[objectofinterest])
                )
            else:
                transformed_points = np.array(keypoints3d_mesh)
                transformed_cuboid_points = np.array(cuboid_points3d_mesh)

            transformed_points = transformed_points[0:no_points]

            for i in range(max_instance_count):
                points3d.append(transformed_points)
                cuboid_points3d.append(transformed_cuboid_points)

            for object_name in object_classes:  # check all present objects
                if objectofinterest in object_name:  # check is present object is object of interest

                    label = class_labels[objectofinterest]
                    iCount = 0
                    for object_id in object_classes[object_name]:  # check all object instances

                        # due to a bug in transformed data the input format is wxyz_input for bop ( has to be changed )
                        transform_mat = quaternion_matrix(
                            poses_quaternion_all[object_id],
                            poses_loc_all[object_id],
                            wxyz_input=self.wxyz_quaterion_input,
                        )

                        points2d.append(np.array(keypoints2d_all[object_id])[0:no_points])

                        transform_mats.append(transform_mat)
                        pixel_gt.append(px_count_all[object_id])
                        object_scale = np.linalg.norm(fixed_transformations[objectofinterest][:, 0])
                        object_diameter = self.meshes[objectofinterest]["diameter"] * object_scale
                        diameter.append(object_diameter)
                        iCount = iCount + 1

                    for i in range(iCount, max_instance_count):  # add none to fill up the array
                        points2d.append(None)
                        transform_mats.append(None)
                        pixel_gt.append(None)
                        diameter.append(None)
                    break

            keypoints2d.append(points2d)
            keypoints3d.append(points3d)
            cuboid3d.append(cuboid_points3d)
            transform_mats_batch.append(transform_mats)
            pixel_gt_batch.append(pixel_gt)
            object_labels.append(label)
            diameters.append(diameter)

        # ____RANDOM KEYPOINT MANIPULATION_______
        dx = round(np.random.normal(0, 2) * float(self.random_translation[0]))
        dy = round(np.random.normal(0, 2) * float(self.random_translation[1]))
        angle = round(np.random.normal(0, 1) * float(self.random_rotation))
        offsets = [
            h_crop,
            w_crop,
            img_size_out[0],
            img_size_out[1],
            dx,
            dy,
            angle,
            scale,
            img_size_orig[0],
            img_size_orig[1],
        ]
        offsets = np.asarray(offsets)
        offsets = offsets.astype("float32")
        tm = np.float32([[1, 0, dx], [0, 1, dy]])
        rm = get_rotation_matrix_2D((img_size_orig[0] / 2, img_size_orig[1] / 2), angle)

        tm2 = np.float32([[1, 0, -dx], [0, 1, -dy]])
        rm2 = get_rotation_matrix_2D((img_size_orig[0] / 2, img_size_orig[1] / 2), -angle)
        affine_r = np.identity(3)
        affine_r[0:2] = rm2
        affine_t = np.identity(3)
        affine_t[0:2] = tm2
        affine = np.matmul(affine_r, affine_t).flatten()[0:8]

        crop_offset = [w_crop, h_crop]

        for object_id in range(len(keypoints2d)):
            for instance_id in range(len(keypoints2d[object_id])):
                keypoints3d[object_id][instance_id] = np.array(keypoints3d[object_id][instance_id])
                cuboid3d[object_id][instance_id] = np.array(cuboid3d[object_id][instance_id])

            if len(keypoints2d[object_id]) == 0:
                keypoints2d[object_id] = np.full(
                    [max_instance_count, no_points, 2], -1000, dtype=np.float32
                )  # use maximum instance count

                transform_mats_batch[object_id] = np.full([max_instance_count, 3, 4], 0.0, dtype=np.float32)
                pixel_gt_batch[object_id] = np.full([max_instance_count, 1], 0.0, dtype=np.float32)
                diameters[object_id] = np.full([max_instance_count, 1], -1.0, dtype=np.float32)
            else:
                for instance_id in range(len(keypoints2d[object_id])):  # use maximum instance count
                    if keypoints2d[object_id][instance_id] is None:
                        keypoints2d[object_id][instance_id] = np.full([no_points, 2], -1000, dtype=np.float32)
                        transform_mats_batch[object_id][instance_id] = np.full([3, 4], 0.0, dtype=np.float32)
                        pixel_gt_batch[object_id][instance_id] = np.full([1], 0.0, dtype=np.float32)
                        diameters[object_id][instance_id] = np.array([-1.0], dtype=np.float32)
                    else:
                        keypoints2d[object_id][instance_id] = (
                            reproject(keypoints2d[object_id][instance_id], tm, rm, crop_offset)
                        ) * scale
                        transform_mats_batch[object_id][instance_id] = np.array(
                            transform_mats_batch[object_id][instance_id]
                        )
                        pixel_gt_batch[object_id][instance_id] = np.array([pixel_gt_batch[object_id][instance_id]])
                        diameters[object_id][instance_id] = np.array([diameters[object_id][instance_id]])
        keypoints2d = np.asarray(keypoints2d).astype("float32")
        keypoints2d = keypoints2d[..., ::-1]  # this is inverting the array TO-Do:Remove
        keypoints3d = np.asarray(keypoints3d).astype("float32")
        cuboid3d = np.asarray(cuboid3d).astype("float32")
        transform_mats_batch = np.asarray(transform_mats_batch).astype("float32")
        pixel_gt_batch = np.asarray(pixel_gt_batch).astype("float32")
        camera_data = np.asarray(camera_data).astype("float32")
        diameters = np.asarray(diameters).astype("float32")

        new_labels = self.set_new_labels(seg_img, object_labels)
        new_labels = np.asarray(new_labels).astype("uint8")

        affine = np.asarray(affine).astype("float32")
        image_id = np.asarray([image_id]).astype("unicode_")

        return (
            img,
            seg_img,
            keypoints2d,
            keypoints3d,
            camera_data,
            diameters,
            offsets,
            affine,
            cuboid3d,
            transform_mats_batch,
            pixel_gt_batch,
            image_id,
            new_labels,
        )

    def load_images(self, path, name, txt, seg, path_raw):

        img = tf.image.decode_image(tf.io.read_file(path))
        channels = tf.shape(img)[2]

        if self.color_input:
            if tf.math.greater(channels, 3):
                img = img[:, :, 0:3]
            if tf.math.equal(channels, 1):
                img = tf.repeat(img, 3, axis=2)
        else:
            if not tf.math.equal(channels, 1):
                img = tf.image.rgb_to_grayscale(img[:, :, 0:3])

        seg_img = tf.image.decode_image(tf.io.read_file(seg), channels=1)

        return img, name, txt, seg_img, path_raw

    def load_json_instance_count(self, path, objectsofinterest):
        """
        Loads the maximum number of instances in a datasample from a json file.
        """
        with open(path) as data_file:
            data = json.load(data_file)
        class_count = np.zeros([len(objectsofinterest)], dtype=np.int32)
        for i_line in range(len(data["objects"])):
            info = data["objects"][i_line]
            object_class = info["class"]
            for idx, obj in enumerate(objectsofinterest):
                # if obj in object_class.lower():
                if obj in object_class:
                    class_count[idx] = class_count[idx] + 1
        return np.max(class_count)

    def load_json_minimal(self, path):
        """
        Loads gt data from a json file.
        """
        with open(path) as data_file:
            data = json.load(data_file)
        # print (path)
        keypoints2d = []
        keypoints3d = []
        poses_quaternions = []
        poses_loc = []
        px_count_all = []
        objectClasses = {}
        obj_idx = 0
        for i_line in range(len(data["objects"])):
            info = data["objects"][i_line]

            if not self.visibility_filter or info["visibility"] > 0.10:
                try:
                    objectClasses[info["class"]].append(obj_idx)
                except KeyError:
                    objectClasses[info["class"]] = [obj_idx]

                if "px_count_all" in info:
                    px_count_all.append(int(info["px_count_all"]))
                else:
                    px_count_all.append(0)

                points2d = []
                pointdata = info["keypoints_2d"]

                for p in pointdata:
                    points2d.append((p[0], p[1]))

                points3d = []
                pointdata = info["keypoints_3d"]
                for p in pointdata:
                    points3d.append((p[0], p[1], p[2]))

                poses_quaternions.append(np.array(info["quaternion_xyzw"], dtype=np.float32, copy=True))
                poses_loc.append(np.array(info["location"], dtype=np.float32, copy=True))

                # Get the centroids
                keypoints2d.append(points2d)
                keypoints3d.append(points3d)
                obj_idx += 1
        return {
            "keypoints2d": keypoints2d,
            "keypoints3d": keypoints3d,
            "objectClasses": objectClasses,
            "poses_quaternions": poses_quaternions,
            "poses_loc": poses_loc,
            "px_count_all": px_count_all,
        }

    def load_json_classes(self, path):
        """
        Loads the data from a object_settings json file.
        """
        with open(path) as data_file:
            data = json.load(data_file)

        objectSegmentationClasses = {}
        fixedTransformations = {}

        for info in data["exported_objects"]:
            objectSegmentationClasses[info["class"]] = info["segmentation_class_id"]
            fixedTransformations[info["class"]] = np.transpose(
                np.array(info["fixed_model_transform"], dtype=np.float32, copy=True)
            )

        return objectSegmentationClasses, fixedTransformations

    def load_json_camera(self, path):
        """
        Loads the data from a camera_settings json file.
        """
        with open(path) as data_file:
            data = json.load(data_file)
        cam = data["camera_settings"][0]["intrinsic_settings"]

        matrix_camera = np.zeros((3, 3))
        matrix_camera[0, 0] = cam["fx"]
        matrix_camera[1, 1] = cam["fy"]
        matrix_camera[0, 2] = cam["cx"]
        matrix_camera[1, 2] = cam["cy"]
        matrix_camera[2, 2] = 1

        return matrix_camera

    def load_mesh(self, path_keypoints, path_mesh, name, meshes, diameter=None):
        loaded_keypoints = trimesh.load(path_keypoints)
        mesh = trimesh.load(path_mesh)
        meshes[name] = {}
        meshes[name]["keypoints"] = loaded_keypoints.vertices
        meshes[name]["vertices"] = mesh.vertices

        meshes[name]["volume"] = mesh.bounding_box.vertices

        if diameter is not None:
            meshes[name]["diameter"] = diameter
        else:
            # http://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
            v = np.asarray(mesh.vertices)  # n*d
            G = np.matmul(v, np.transpose(v))  # n*n
            dist_mat = np.diag(G) + np.transpose(np.diag(G)) - 2 * G
            diameter = np.sqrt(np.max(dist_mat))
            meshes[name][
                "diameter"
            ] = diameter  # np.linalg.norm(sphere_center - mesh.bounding_sphere.vertices[1]) * 2 # wrong calculation

        return meshes

    def load_meshes(self, path):
        folders_names = [o for o in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, o))]
        meshes = {}
        for i in range(len(folders_names)):
            name = folders_names[i]

            filetype = ".obj"

            model_file = path + "/" + name + "/" + name + filetype
            if not exists(model_file):
                model_file = model_file.replace(filetype, ".ply")

            print(model_file)
            model_keypoint_file = path + "/" + name + "/" + name + "_keypoints.ply"
            if os.path.isfile(model_file) and os.path.isfile(model_keypoint_file):
                diameter = None
                info_file = path + "/models_info.json"
                if os.path.isfile(info_file):
                    with open(info_file) as f:
                        model_info = yaml.load(f, Loader=yaml.FullLoader)
                    diameter = model_info[name]["diameter"]
                meshes = self.load_mesh(model_keypoint_file, model_file, name, meshes, diameter=diameter)
        return meshes

    def load_image_data(self, root):
        """
        Search for images in all subfolders and add them to imgs.
        """
        imgs = []
        class_labels = {}
        fixed_transformations = {}

        camera_data = {}

        def collect_image_files(
            path,
        ):
            files = sorted(glob.glob(path + "/*seg.png"))
            if len(files) != 0:
                if self.use_train_split or self.use_validation_split:
                    split = np.array(load_split(path, self.train_validation_split), dtype=bool)
                    if self.use_train_split:
                        files = list(compress(files, split.tolist()))
                    else:
                        files = list(compress(files, np.invert(split).tolist()))
                    print(len(files))
                if path not in class_labels:
                    (
                        class_labels[path],
                        fixed_transformations[path],
                    ) = self.load_json_classes(path + "/_object_settings.json")

                if path not in camera_data:
                    camera_data[path] = self.load_json_camera(path + "/_camera_settings.json")
            for seg_path in files:
                filetype = "png"
                # test different filetypes
                imgpath = seg_path.replace("seg.png", filetype)
                if not exists(imgpath):
                    imgpath = imgpath.replace(filetype, "bmp")
                    filetype = "bmp"
                    if not exists(imgpath):
                        imgpath = imgpath.replace(filetype, "jpg")
                        filetype = "jpg"

                if exists(imgpath) and exists(seg_path) and exists(imgpath.replace(filetype, "json")):
                    imgs.append(
                        (
                            imgpath,
                            imgpath.replace(path, "").replace("/", "").replace("\\", ""),
                            imgpath.replace(filetype, "json"),
                            seg_path,
                            path,
                        )
                    )

        def explore(path):
            if not os.path.isdir(path):
                return
            folders = [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
            if len(folders) > 0:
                for folder in folders:
                    explore(folder)
            else:
                collect_image_files(path)

        explore(root)

        return imgs, class_labels, fixed_transformations, camera_data

    def set_new_labels(self, seg_img, object_labels):
        new_labels = []
        # seg_img_new = np.zeros_like(seg_img, dtype=np.uint8)
        for idx, obj in enumerate(object_labels):
            if obj is not None:
                new_labels.append(tf.constant([object_labels[idx], idx + 1]))
                # seg_img_new = tf.where(seg_img_new==object_labels[idx], np.uint8(idx+1), seg_img_new)
            else:
                new_labels.append(tf.constant([0, 0]))
                # HACK beacause of wrong IDs in dataset. has to be removed!!!!!
                # new_labels = np.where(seg_img>1, np.uint8(idx+1), new_labels)

        new_labels = tf.stack(new_labels, axis=0)
        return new_labels  # , seg_img_new

    def generate_dataset(
        self,
        batchsize,
        epochs,
        prefetch,
        imagesize,
        cropratio,
        worker,
        no_objects,
        shuffle=True,
        mirrored_strategy=None,
    ):
        def create_base_dataset(imgs):
            path_list = []
            name_list = []
            txt_list = []
            seg_list = []
            path_raw_list = []

            for path, name, txt, seg, path_raw in imgs:
                path_list.append(path)
                name_list.append(name)
                txt_list.append(txt)
                seg_list.append(seg)
                path_raw_list.append(path_raw)

            # print("find max instance count of {} paths".format(len(txt_list)))

            # max_count = 0
            # for path in txt_list:
            #    max_count = max(max_count, self.load_json_instance_count(path, self.objectsofinterest))
            max_count = 1
            # print("max INSTANCE count: {}".format(max_count))

            dataset_out = tf.data.Dataset.from_tensor_slices((path_list, name_list, txt_list, seg_list, path_raw_list))
            first_img = tf.image.decode_image(tf.io.read_file(path_list[0]))
            input_size = [first_img.shape[0], first_img.shape[1]]
            return dataset_out, max_count, input_size

        dataset_out, max_count, input_size = create_base_dataset(self.imgs)

        data_size = len(self.imgs) - (len(self.imgs) % batchsize)

        epoch_batches = data_size / batchsize

        def set_shapes(
            img,
            seg,
            points,
            points3d,
            camera_data,
            diameters,
            off,
            affine,
            cuboid3d,
            transform_mats,
            pixel_gt_count,
            image_id,
            new_labels,
            no_objects,
            max_count,
            input_size,
            no_points,
        ):
            channels = 3 if self.color_input else 1
            img.set_shape([input_size[0], input_size[1], channels])
            seg.set_shape([input_size[0], input_size[1], 1])  # 540 960
            points.set_shape([no_objects, max_count, no_points, 2])  # ver_dim
            points3d.set_shape([no_objects, max_count, no_points, 3])  # ver_dim
            cuboid3d.set_shape([no_objects, max_count, 8, 3])  # ver_dim
            transform_mats.set_shape([no_objects, max_count, 3, 4])  # ver_dim
            pixel_gt_count.set_shape([no_objects, max_count, 1])  # ver_dim
            diameters.set_shape([no_objects, max_count, 1])  # ver_dim
            camera_data.set_shape([3, 3])
            off.set_shape([10])
            affine.set_shape([8])
            image_id.set_shape([1])
            new_labels.set_shape([no_objects, 2])  # ver_dim

            return (
                img,
                seg,
                points,
                points3d,
                camera_data,
                diameters,
                off,
                affine,
                cuboid3d,
                transform_mats,
                pixel_gt_count,
                image_id,
                new_labels,
            )

        dataset_out = dataset_out.take(data_size)
        if shuffle:
            dataset_out = dataset_out.shuffle(data_size)
        dataset_out = dataset_out.repeat(epochs)

        dataset_out = dataset_out.map(
            lambda path, name, text, seg, path_raw: self.load_images(path, name, text, seg, path_raw),
            num_parallel_calls=1,
        )
        dataset_out = dataset_out.map(
            lambda path, name, text, seg, path_raw: tuple(
                tf.numpy_function(
                    self.apply_preprocessing,
                    inp=[
                        path,
                        name,
                        text,
                        seg,
                        path_raw,
                        imagesize,
                        cropratio,
                        max_count,
                        self.no_points,
                    ],
                    Tout=[
                        tf.uint8,
                        tf.uint8,
                        tf.float32,
                        tf.float32,
                        tf.float32,
                        tf.float32,
                        tf.float32,
                        tf.float32,
                        tf.float32,
                        tf.float32,
                        tf.float32,
                        tf.string,
                        tf.uint8,
                    ],
                )
            ),
            num_parallel_calls=1,
        )
        dataset_out = dataset_out.map(
            lambda img, seg, points, points3d, camera_data, diameters, off, affine, cuboid3d, transform_mats, pixel_gt_count, image_id, new_labels: set_shapes(
                img,
                seg,
                points,
                points3d,
                camera_data,
                diameters,
                off,
                affine,
                cuboid3d,
                transform_mats,
                pixel_gt_count,
                image_id,
                new_labels,
                no_objects,
                max_count,
                input_size,
                self.no_points,
            ),
            num_parallel_calls=1,
        )
        dataset_out = dataset_out.batch(batchsize)
        dataset_out = dataset_out.map(
            lambda img, seg, points, points3d, camera_data, diameters, off, affine, cuboid3d, transform_mats, pixel_gt_count, image_id, new_labels: self.image_transformation(
                img,
                seg,
                points,
                points3d,
                camera_data,
                diameters,
                off,
                affine,
                cuboid3d,
                transform_mats,
                pixel_gt_count,
                image_id,
                new_labels,
                imagesize,
            ),
            num_parallel_calls=worker,
        )

        if self.use_imgaug:
            dataset_out = dataset_out.map(
                lambda img, seg, points, points3d, camera_data, diameters, off, affine, cuboid3d, transform_mats, pixel_gt_count, image_id, new_labels: self.tf_random_augmentations(
                    img,
                    seg,
                    points,
                    points3d,
                    camera_data,
                    diameters,
                    off,
                    affine,
                    cuboid3d,
                    transform_mats,
                    pixel_gt_count,
                    image_id,
                    new_labels,
                ),
                num_parallel_calls=worker,
            )
            # dataset_out = dataset_out.prefetch(prefetch)
        dataset_out = dataset_out.map(
            lambda img, seg, points, points3d, camera_data, diameters, off, affine, cuboid3d, transform_mats, pixel_gt_count, image_id, new_labels: self.image_augmentation(
                img,
                seg,
                points,
                points3d,
                camera_data,
                diameters,
                off,
                affine,
                cuboid3d,
                transform_mats,
                pixel_gt_count,
                image_id,
                new_labels,
                imagesize,
                input_size,
            ),
            num_parallel_calls=worker,
        )
        dataset_out = dataset_out.prefetch(prefetch)

        if self.save:
            path_out = self.output_folder + "/visual_batch"
            print(path_out)
            save_batches(
                dataset_out,
                path_out,
                self.separated_vectorfields,
                no_objects,
                self.no_points,
                self.normal,
                1,
            )
            exit()

        if mirrored_strategy is not None:
            with mirrored_strategy.scope():
                dataset_out = mirrored_strategy.experimental_distribute_dataset(dataset_out)
        return dataset_out, epoch_batches

    def generate_segmentation_and_direction_maps(self, seg_img, object_labels, coords, normalize=True):
        masks = []
        direction_maps = []
        masks.append(tf.cast(tf.equal(seg_img, 0), dtype=tf.float32))
        for idx, obj in enumerate(object_labels):
            masks.append(tf.cast(tf.equal(seg_img, object_labels[obj]), dtype=tf.float32))
            direction_maps.append(self.compute_vertex_hcoords(masks[-1], coords[idx]))
        masks = tf.concat(masks, 2)
        direction_maps = tf.concat(direction_maps, 2)
        return masks, direction_maps

    @tf.function
    def generate_segmentation_and_direction_maps_batch_v2(
        self,
        seg_batch,
        coords,
        new_labels,
        normalize=True,
        separated_vectorfields=False,
    ):
        no_objects = tf.shape(coords)[1]

        seg_batch_transformed = tf.zeros_like(seg_batch)
        for j in range(no_objects):
            n = tf.expand_dims(tf.expand_dims(tf.expand_dims(new_labels[:, j, 0], -1), -1), -1)
            m = tf.expand_dims(tf.expand_dims(tf.expand_dims(new_labels[:, j, 1], -1), -1), -1)
            seg_batch_transformed = tf.where(seg_batch == n, m, seg_batch_transformed)
        seg_batch = seg_batch_transformed

        masks = tf.one_hot(tf.squeeze(seg_batch, -1), coords.shape[1] + 1, dtype=tf.float32)

        # if separated_vectorfields:
        #     direction_maps = self.compute_vertex_hcoords_batch_v3(masks[:,:,:,1:2], coords[:,0:1,:,:,:])
        #     for idx in range(1,no_objects): # iterate over objects
        #         direction_map = self.compute_vertex_hcoords_batch_v3(masks[:,:,:,idx+1:idx+2], coords[:,idx:idx+1,:,:,:]) # instance axis is squeezed!
        #         direction_maps = tf.concat([direction_maps, direction_map], 3)
        # else:
        #     direction_maps = self.compute_vertex_hcoords_batch_v3(seg_batch, coords)
        direction_maps = tf.zeros_like(seg_batch)
        return masks, direction_maps, seg_batch

    def generate_object_vertex_array(self):

        vertex_count = np.zeros([len(self.objectsofinterest), 1], dtype=np.int32)
        idx = 0
        for name, mesh in self.meshes.items():
            if name in self.objectsofinterest:
                vertex_count[idx, 0] = len(mesh["vertices"])
                idx = idx + 1
        vertex_array = np.zeros([len(self.objectsofinterest), np.max(vertex_count), 3], dtype=np.float32)
        for idx, objectofinterest in enumerate(self.objectsofinterest):
            if objectofinterest in self.meshes:
                fixed_transformation_found = False
                for _, fixed_transformations in self.fixed_transformations.items():
                    if objectofinterest in fixed_transformations:
                        vertices = np.array(
                            transform_points(
                                self.meshes[objectofinterest]["vertices"],
                                fixed_transformations[objectofinterest],
                            )
                        )
                        fixed_transformation_found = True
                        break
                if fixed_transformation_found:
                    vertex_array[idx, : vertex_count[idx][0]] = vertices

        vertex_count = tf.convert_to_tensor(vertex_count, dtype=tf.int32)
        vertex_array = tf.convert_to_tensor(vertex_array, dtype=tf.float32)

        return vertex_array, vertex_count
