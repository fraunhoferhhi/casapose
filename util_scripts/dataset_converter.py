import glob
import json
import os
import re
from shutil import copyfile

import numpy as np
import pyrender
import trimesh
from casapose.utils.draw_utils import draw_bb, draw_points
from casapose.utils.geometry_utils import (
    create_transformation_matrix,
    get_horizontal_width_angle,
    matrix_to_quaternion,
    project,
)
from casapose.utils.io_utils import to_json
from PIL import Image

CV_TO_OPENGL = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def get_cam(camera_matrix, zNear, zFar):
    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]
    cx = camera_matrix[0][2]
    cy = camera_matrix[1][2]  # + 0.5
    return pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=zNear, zfar=zFar)  # still causing a small shift


def render_scene_4x4(scene, node, renderer, fixed_model_transform, cv_to_opengl, transform_mat):
    transform_mat = np.matmul(transform_mat, fixed_model_transform)
    transform_mat = np.matmul(cv_to_opengl, transform_mat)
    scene.set_pose(node, pose=transform_mat)
    return renderer.render(scene)


def init_scene(camera):
    scene = pyrender.Scene()
    scene.add(camera)
    return scene


def create_bop_mask(path, path_out, gt, digits, width, height, filetypet):
    mask = np.zeros([height, width], dtype=np.uint8)
    path = path.replace("rgb", "mask_visib")
    for idx, mesh_gt in enumerate(gt):
        # for idx, mesh in enumerate(meshes):
        path_new = path.replace(digits + "." + filetypet, digits + "_" + str(idx).zfill(6) + ".png")
        input_image = Image.open(path_new)
        input_image_np = np.array(input_image)
        mask[input_image_np == 255] = mesh_gt["id"]
    im = Image.fromarray(mask)
    im.save(path_out)


def create_ndds_mask(path, renderer, camera_matrix, gt, meshes, settings):
    width = settings["width"]
    height = settings["height"]
    camera = get_cam(camera_matrix, settings["near"], settings["far"])
    scene = init_scene(camera)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3)
    scene.add(light)
    stacked_depth = np.zeros([1, height, width], dtype="float32")
    for mesh_gt in gt:

        nm = pyrender.Node(mesh=meshes[mesh_gt["id"]]["mesh"])

        scene.add_node(nm)
        transform_mat = create_transformation_matrix(mesh_gt["R"], mesh_gt["t"])
        _, depth = render_scene_4x4(scene, nm, renderer, np.eye(4), CV_TO_OPENGL, transform_mat)
        stacked_depth[0] = stacked_depth[0] + depth * 2
        depth[depth == 0] = 10000.0

        stacked_depth = np.concatenate((stacked_depth, np.expand_dims(depth, axis=0)), axis=0)
        scene.remove_node(nm)

    depth_index = np.argmin(stacked_depth, axis=0)
    # path_mask = path.replace('.png', '.seg.png')
    mask = np.zeros([height, width])
    index = 1
    for mesh_gt in gt:
        mask[depth_index == index] = mesh_gt["id"]
        index = index + 1
    mask = mask.astype(np.uint8)
    im = Image.fromarray(mask)

    im.save(path)


def write_camera_setting(path, name, camera_matrix, width, height):

    json_data = {}
    json_data["camera_settings"] = []
    camera = {}
    camera["name"] = name
    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]
    camera["horizontal_fov"] = get_horizontal_width_angle(width, height, fx, fy)
    intrinsics = {}
    intrinsics["resX"] = width
    intrinsics["resY"] = height
    intrinsics["fx"] = fx
    intrinsics["fy"] = fy
    intrinsics["cx"] = camera_matrix[0][2]
    intrinsics["cy"] = camera_matrix[1][2]
    intrinsics["s"] = 0  # info['cam_K'][1]
    camera["intrinsic_settings"] = intrinsics

    img_size = {}
    img_size["width"] = width
    img_size["height"] = height
    camera["captured_image_size"] = img_size
    json_data["camera_settings"].append(camera)

    with open(path, "w") as outfile:
        outfile.write(to_json(json_data))


def write_object_settings(path, meshes):

    json_data = {}
    json_data["exported_object_classes"] = []
    json_data["exported_objects"] = []
    for mesh in meshes:
        if meshes[mesh]["counter"] > 0:
            json_data["exported_object_classes"].append(meshes[mesh]["name"])
            mesh_info = {}
            mesh_info["class"] = meshes[mesh]["name"]
            mesh_info["segmentation_class_id"] = meshes[mesh]["id"]
            mesh_info["segmentation_instance_id"] = 0
            mesh_info["fixed_model_transform"] = meshes[mesh]["fixed_model_transform"].tolist()
            mesh_info["cuboid_dimensions"] = meshes[mesh]["volume_size"]
            json_data["exported_objects"].append(mesh_info)

    with open(path, "w") as outfile:
        outfile.write(to_json(json_data))


def create_ndds_json(path, camera_matrix, gt, meshes, debug_image=None):
    # create and save json

    json_data = {}
    json_data["camera_data"] = {}
    json_data["camera_data"]["location_worldframe"] = [0.0, 0.0, 0.0]
    json_data["camera_data"]["quaternion_xyzw_worldframe"] = [0.0, 0.0, 0.0, 1.0]
    json_data["objects"] = []
    for mesh_gt in gt:
        object_id = mesh_gt["id"]
        t = mesh_gt["t"]
        R = mesh_gt["R"]
        bb = mesh_gt["bb"]
        pose = create_transformation_matrix(R, t)
        meshes[object_id]["counter"] = meshes[object_id]["counter"] + 1
        mesh_info = {}
        mesh_info["class"] = meshes[object_id]["name"]
        mesh_info["instance_id"] = 0  # add instance counter

        if "visib_fract" in mesh_gt:
            mesh_info["visibility"] = mesh_gt["visib_fract"]
        else:
            mesh_info["visibility"] = 1  # calculate visibility
        if "px_count_all" in mesh_gt:
            mesh_info["px_count_all"] = mesh_gt["px_count_all"]
        if "px_count_valid" in mesh_gt:
            mesh_info["px_count_valid"] = mesh_gt["px_count_valid"]
        if "px_count_visib" in mesh_gt:
            mesh_info["px_count_visib"] = mesh_gt["px_count_visib"]

        mesh_info["location"] = t
        mesh_info["quaternion_xyzw"] = matrix_to_quaternion(R)
        mesh_info["pose_transform"] = np.transpose(pose).tolist()
        center = np.array(np.expand_dims(meshes[object_id]["center"], 0))
        center_2d, center_3d = project(center, camera_matrix, pose[0:3])
        mesh_info["cuboid_centroid"] = center_3d[0]
        mesh_info["projected_cuboid_centroid"] = center_2d[0]
        mesh_info["bounding_box"] = {}

        mesh_info["bounding_box"]["top_left"] = [bb[0], bb[1]]
        mesh_info["bounding_box"]["bottom_right"] = [bb[0] + bb[2], bb[1] + bb[3]]

        if "bb_visib" in mesh_gt:
            bb_visib = mesh_gt["bb_visib"]
            mesh_info["bounding_box_visible"] = {}
            mesh_info["bounding_box_visible"]["top_left"] = [bb_visib[0], bb_visib[1]]
            mesh_info["bounding_box_visible"]["bottom_right"] = [
                bb_visib[0] + bb_visib[2],
                bb_visib[1] + bb_visib[3],
            ]

        cuboid_2d, cuboid_3d = project(meshes[object_id]["volume"], camera_matrix, pose[0:3])
        mesh_info["cuboid"] = cuboid_3d.tolist()
        mesh_info["projected_cuboid"] = cuboid_2d.tolist()
        keypoints_2d, keypoints_3d = project(meshes[object_id]["keypoints"], camera_matrix, pose[0:3])
        mesh_info["keypoints_2d"] = keypoints_2d.tolist()
        mesh_info["keypoints_3d"] = keypoints_3d.tolist()
        json_data["objects"].append(mesh_info)
        if debug_image is not None:
            draw_points(keypoints_2d, debug_image)
            draw_bb(cuboid_2d, debug_image)

    with open(path, "w") as outfile:
        outfile.write(to_json(json_data))
    # create and save mask
    return meshes  # updated counter


def load_object(
    path_keypoints,
    path_mesh,
    index,
    name,
    meshes,
    fixed_transform=None,
):
    loaded_keypoints = trimesh.load(path_keypoints)
    mesh = trimesh.load(path_mesh)
    meshes[index] = {}
    meshes[index]["name"] = name
    meshes[index]["keypoints"] = loaded_keypoints.vertices
    meshes[index]["volume"] = mesh.bounding_box_oriented.vertices
    meshes[index]["volume_size"] = np.max(loaded_keypoints.vertices, 0) - np.min(loaded_keypoints.vertices, 0)
    meshes[index]["center"] = (np.max(loaded_keypoints.vertices, 0) + np.min(loaded_keypoints.vertices, 0)) / 2.0
    meshes[index]["counter"] = 0
    meshes[index]["fixed_model_transform"] = np.eye(4)
    meshes[index]["id"] = index
    # meshes[index]['center'] =
    # if fixed_transform is not None:
    #    meshes[index]['keypoints'] = transformPoints(kmeshes[index]['keypoints'], fixed_transform)
    #    meshes[index]['volume'] = transformPoints(meshes[index]['volume'], fixed_transform)
    #    mesh.vertices = transformPoints(mesh.vertices, fixed_transform)

    meshes[index]["mesh"] = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    return meshes


def get_cam_matrix_bop(info):
    cam = np.eye(3)
    cam[0][0] = info["cam_K"][0]
    cam[1][1] = info["cam_K"][4]
    cam[0][2] = info["cam_K"][2]
    cam[1][2] = info["cam_K"][5]
    return cam


def update_data(path, path_out, meshes, settings):
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    # Check all the folders in path
    for name in os.listdir(str(path)):
        # print(path +"/"+name)
        parse_bop(path + "/" + name, path_out + "/" + name, meshes, settings)


def load_model_infos(files):
    model_info = {}
    model_info_file = ""
    # print('load {}'.format(len(files)))
    for file in files:
        name = os.path.basename(file)
        print(name)
        if name == "models_info.json":
            model_info_file = file
            with open(file) as f:
                model_info = json.load(f)
        else:
            print("unknown file: {}".format(name))
    return model_info, model_info_file


def load_json_info(files):
    cameras = {}
    gts = {}
    cameras_out = {}
    gts_out = {}
    gt_infos = {}
    # print('load {}'.format(len(files)))
    for file in files:
        name = os.path.basename(file)
        if name == "scene_gt.json":
            with open(file) as f:
                gts = json.load(f)
        elif name == "scene_camera.json":
            with open(file) as f:
                cameras = json.load(f)
        elif name == "scene_gt_info.json":
            with open(file) as f:
                gt_infos = json.load(f)
        else:
            print("unknown json file: {}".format(name))

    for camera in cameras:
        new_camera = {}
        new_camera["cam_mat"] = get_cam_matrix_bop(cameras[camera])
        cameras_out[int(camera)] = new_camera
    for gt in gts:
        new_gts = []
        for obj_gt in gts[gt]:
            new_gt = {}
            new_gt["id"] = obj_gt["obj_id"]
            new_gt["t"] = obj_gt["cam_t_m2c"]
            r = obj_gt["cam_R_m2c"]
            R = np.array([[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]])
            new_gt["R"] = R
            # new_gt['bb'] = obj_gt['obj_bb']
            new_gts.append(new_gt)
        gts_out[int(gt)] = new_gts

    for gt_info in gt_infos:
        for i, obj_gt in enumerate(gt_infos[gt_info]):
            gts_out[int(gt_info)][i]["bb"] = obj_gt["bbox_obj"]
            gts_out[int(gt_info)][i]["bb_visib"] = obj_gt["bbox_visib"]
            gts_out[int(gt_info)][i]["px_count_all"] = obj_gt["px_count_all"]
            gts_out[int(gt_info)][i]["px_count_valid"] = obj_gt["px_count_valid"]
            gts_out[int(gt_info)][i]["px_count_visib"] = obj_gt["px_count_visib"]
            gts_out[int(gt_info)][i]["visib_fract"] = obj_gt["visib_fract"]
            i += 1

    return cameras_out, gts_out


def parse_bop(root, root_out, meshes, settings):

    r = pyrender.OffscreenRenderer(settings["width"], settings["height"])

    def update_bop_files(path, info, gt, meshes):
        filetype = "." + settings["filetype_in"]
        files = sorted(glob.glob(path + "/[0-9][0-9][0-9][0-9][0-9][0-9]" + filetype))  # six digits
        # files += glob.glob(path+"/*" + "right.json")
        if len(files) > 0:
            path_out = path.replace(root, root_out)
            if not os.path.exists(path_out):
                os.mkdir(path_out)
        for filepath in files:
            digits = re.findall(r"\d+", os.path.basename(filepath))
            if len(digits) > 0:
                filepath_out = filepath.replace(root, root_out)
                if filepath_out != filepath:
                    copyfile(filepath, filepath_out)
                idx = int(digits[0])
                camera_matrix = info[idx]["cam_mat"]

                input_image_np = None
                if settings["draw_debug_image"]:
                    input_image = Image.open(filepath).convert("RGB")
                    input_image_np = np.array(input_image)
                meshes = create_ndds_json(
                    filepath_out.replace(filetype, ".json"),
                    camera_matrix,
                    gt[idx],
                    meshes,
                    input_image_np,
                )
                if settings["draw_debug_image"]:
                    input_image = Image.fromarray(input_image_np)
                    input_image.save(filepath_out.replace(filetype, "_debug.jpg"))
                    exit()
                if settings["mask"] == "reuse":
                    create_bop_mask(
                        filepath,
                        filepath_out.replace(filetype, ".seg.png"),
                        gt[idx],
                        digits[0],
                        settings["width"],
                        settings["height"],
                        settings["filetype_in"],
                    )
                elif settings["mask"] == "render":
                    create_ndds_mask(
                        filepath_out.replace(filetype, ".seg.png"), r, camera_matrix, gt[idx], meshes, settings
                    )
                    ####

        return meshes

    def explore(path, meshes):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]

        folders_names = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]

        if "rgb" in folders_names:
            print(path)
            path_out = path.replace(root, root_out)
            if not os.path.exists(path_out):
                os.mkdir(path_out)
            if not os.path.exists(path_out + "/rgb"):
                os.mkdir(path_out + "/rgb")

            for mesh in meshes:
                meshes[mesh]["counter"] = 0
            files = sorted(glob.glob(path + "/*" + ".json"))
            info, gt = load_json_info(files)
            json_found = True
            {"A": 1, "B": 2}
            camera_matrix = next(iter(info.values()))["cam_mat"]
            print(camera_matrix)
            write_camera_setting(
                path_out + "/rgb/_camera_settings.json",
                "Viewpoint",
                camera_matrix,
                settings["width"],
                settings["height"],
            )

        if len(folders) > 0 and not json_found:
            for folder in folders:
                explore(folder, meshes)
        elif json_found:
            meshes = update_bop_files(folders[folders_names.index("rgb")], info, gt, meshes)
            write_object_settings(path_out + "/rgb/_object_settings.json", meshes)

    explore(root, meshes)


def load_models_bop(path, path_root_out, copy_meshes=False):
    if not os.path.exists(path_root_out):
        os.mkdir(path_root_out)

    json_files = sorted(glob.glob(path + "/*" + ".json"))
    model_infos, model_info_file = load_model_infos(json_files)

    if len(model_infos) == 0:
        return
    model_files = sorted(glob.glob(path + "/*" + ".ply"))
    model_keypoint_files = sorted(glob.glob(path + "/*" + "keypoints.ply"))
    model_files = list(filter(lambda a: a not in model_keypoint_files, model_files))  # filter dublicates
    meshes = {}
    if len(model_files) == 0:
        model_files = sorted(glob.glob(path + "/*" + ".obj"))

    for i in range(len(model_files)):
        name = os.path.splitext(os.path.basename(model_files[i]))[0]
        print(name)
        digits_model = re.findall(r"\d+", name)
        digits_model_keypoints = re.findall(r"\d+", os.path.basename(model_keypoint_files[i]))
        if (
            len(digits_model) > 0
            and len(digits_model_keypoints) > 0
            and int(digits_model_keypoints[0]) == int(digits_model[0])
        ):
            meshes = load_object(model_keypoint_files[i], model_files[i], int(digits_model[0]), name, meshes)

        if copy_meshes:
            path_out = path_root_out + "/" + name
            if not os.path.exists(path_out):
                os.mkdir(path_out)
            copyfile(model_files[i], path_out + "/" + name + ".ply")
            copyfile(model_keypoint_files[i], path_out + "/" + name + "_keypoints.ply")

    if copy_meshes:
        print("Copy: " + model_info_file)
        copyfile(model_info_file, path_root_out + "/models_info.json")

    return meshes
    # for info in model_infos:


def generate_data(
    dataset_path,
    dataset_path_out,
    settings,
    model_folder="models",
    model_folder_out="models",
    image_folder="train_pbr",
):
    path_models = os.path.join(dataset_path, model_folder)
    path_models_out = os.path.join(dataset_path_out, model_folder_out)
    path_images = os.path.join(dataset_path, image_folder)
    path_images_out = os.path.join(dataset_path_out, image_folder)

    meshes = load_models_bop(path_models, path_models_out, settings["copy_meshes"])

    update_data(path_images, path_images_out, meshes, settings)


# if not os.name == "nt":
#     os.environ["PYOPENGL_PLATFORM"] = "egl"
# dataset_path = "E:/DeepLearningData/DTWIN_Demo3_9904_BlenderProc_2/"  # ADD to command line

# settings_bop = {}
# settings_bop["type"] = "bop"
# settings_bop["near"] = 100
# settings_bop["far"] = 2000

# settings_bop["width"] = 1024  # ADD to command line
# settings_bop["height"] = 1024  # ADD to command line
# settings_bop["filetype_in"] = "png"  # ADD to command line

# settings_bop["render_mask"] = True
# settings_bop["rename_data"] = True
# settings_bop["draw_debug_image"] = False

# generate_data(dataset_path, settings_bop, model_folder="models", image_folder="train_pbr")

# path_models = dataset_path + "models"
# path_models_out = dataset_path + "models_converted"
# meshes = load_models_sixd_bop(path_models, path_models_out, settings_bop["rename_data"], json=True)
# path_images = dataset_path + "train_pbr"
# path_images_out = dataset_path + "train_pbr_converted"
# update_data(path_images, path_images_out, meshes, settings_bop)
