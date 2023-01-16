import glob
import os
from os.path import exists

import tensorflow as tf


class ImageOnlyDataset:
    def __init__(
        self,
        root,
        normal=[0.5, 0.5],
    ):
        ###################
        self.normal = normal

        def load_data(path):
            imgs = []

            img_tmp = self.load_image_data(path)
            imgs += img_tmp

            return imgs

        self.imgs = load_data(root)

    def __len__(self):
        return len(self.imgs)

    # always picks from first dataset
    def __getitem__(self, index):
        path = self.imgs[index]
        name = os.path.splitext(os.path.basename(path))[0]
        return {"path": path, "name": name}

    def load_images(self, path, input_size):
        # t = tf.timestamp()
        img = tf.image.decode_image(tf.io.read_file(path))

        img.set_shape([input_size[0], input_size[1], input_size[2]])
        channels = tf.shape(img)[2]
        if tf.math.greater(channels, 3):
            img = img[:, :, 0:3]
        if tf.math.equal(channels, 1):
            img = tf.repeat(img, 3, axis=2)

        img = ((img / 255) - self.normal[0]) / self.normal[1]
        # img = tf.image.resize(img, output_size)
        return img

    def load_image_data(self, root):
        """
        Search for images in all subfolders and add them to imgs.
        """
        imgs = []

        def collect_image_files(path):
            files = sorted(glob.glob(path + "/*[0-9].png"))
            if len(files) == 0:
                files = sorted(glob.glob(path + "/*[0-9].jpg"))

            for img_path in files:
                if exists(img_path):
                    p = img_path  # .decode("utf-8")
                    p = os.path.normpath(p.replace("\\", "/")).split(os.sep)
                    # image_id = p[-2] + '_' + p[-1]#.decode("utf-8")# + '_' + os.path.splitext(name)[0].decode("utf-8")
                    # print(image_id)
                    imgs.append(img_path)

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

        return imgs

    def generate_dataset(self, batchsize):
        def create_base_dataset(imgs):
            path_list = []

            for path in imgs:
                path_list.append(path)

            dataset_out = tf.data.Dataset.from_tensor_slices((path_list))
            first_img = tf.image.decode_image(tf.io.read_file(path_list[0]))
            input_size = [first_img.shape[0], first_img.shape[1], first_img.shape[2]]

            return dataset_out, input_size

        dataset_out, input_size = create_base_dataset(self.imgs)

        data_size = len(self.imgs) - (len(self.imgs) % batchsize)
        epoch_batches = data_size / batchsize
        dataset_out = dataset_out.take(data_size)

        dataset_out = dataset_out.map(lambda path: self.load_images(path, input_size), num_parallel_calls=1)
        dataset_out = dataset_out.batch(batchsize)

        return dataset_out, epoch_batches
