import glob
import os
import uuid

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset

from data_aug import (Sequence, RandomScale, RandomRotate,
                      RandomTranslate)


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect', anti_aliasing=True)
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, train=True):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [
            path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for
            path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50
        self.train = train

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        _, ext = os.path.splitext(img_path)
        u = str(uuid.uuid4())
        aux_img_path = os.path.join('/tmp', f'{u}_img{ext}')
        aux_lab_path = os.path.join('/tmp', f'{u}_lab{ext}')

        # Data augmentation
        if self.train:
            img = cv2.imread(img_path)
            bboxes = parse_yolo_coordinates(open(label_path, "r").read().split("\n"), img)
            transforms = Sequence(
                [RandomScale(0.2, diff=True), RandomRotate(20), RandomTranslate(0.02)])

            og_img = img
            og_boxes = bboxes

            # Keep trying to transform if there are errors
            # There is some randomness in the transformation, so we cannot
            # know when or what there will be errors with the bboxes
            while True:
                try:
                    img, bboxes = transforms(og_img, og_boxes)

                    break
                except IndexError:
                    continue

            status = write_yolo_coordinates(
                bboxes, img,
                aux_lab_path)

            if status:
                cv2.imwrite(aux_img_path, img)

                img = np.array(Image.open(aux_img_path))

                labels = None
                if os.path.exists(aux_lab_path):
                    labels = np.loadtxt(aux_lab_path).reshape(-1, 5)

                os.remove(aux_img_path)
                os.remove(aux_lab_path)

            else:
                img = np.array(Image.open(img_path))

                labels = None
                if os.path.exists(label_path):
                    labels = np.loadtxt(label_path).reshape(-1, 5)

        else:
            img = np.array(Image.open(img_path))

            labels = None
            if os.path.exists(label_path):
                labels = np.loadtxt(label_path).reshape(-1, 5)

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect', anti_aliasing=True)
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        # ---------
        #  Label
        # ---------

        if labels is not None:
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3] / 2)
            y1 = h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = h * (labels[:, 2] + labels[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)


def parse_yolo_coordinates(rows, img):
    width_img = img.shape[1]
    height_img = img.shape[0]
    ret = []
    for r in rows:

        a = dict()
        r = r.split(" ")
        if len(r) == 1:
            continue
        x_center = float(r[1])
        y_center = float(r[2])
        width = float(r[3])
        height = float(r[4])
        label = float(r[0])
        ret.append([width_img * (x_center - width / 2.0), height_img * (y_center - height / 2.0),
                    width_img * (x_center + width / 2.0), height_img * (y_center + height / 2.0),
                    label])
    y = np.array([np.array(xi) for xi in ret])
    return y


def write_yolo_coordinates(bboxes, img, out_file):
    width_img = img.shape[1]
    height_img = img.shape[0]

    # bb: x1 y1 x2 y2 c
    if len(bboxes) == 0:
        return False

    with open(out_file, "w") as a:
        for bb in bboxes:
            width = (bb[2] - bb[0]) / float(width_img)
            height = (bb[3] - bb[1]) / float(height_img)

            center_x = (bb[0] + bb[2]) / (2.0 * width_img)
            center_y = (bb[1] + bb[3]) / (2.0 * height_img)

            a.write(f'{int(bb[4])} {center_x} {center_y} {width} {height}\n')

    return True
