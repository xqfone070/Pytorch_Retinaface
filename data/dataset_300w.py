import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np


def load_pts_file(filename):
    with open(filename) as f:
        lines = [line.strip() for line in f]

    """Use the curly braces to find the start and end of the point data"""
    head = lines.index('{') + 1
    tail = lines.index('}')

    """Select the point data split into coordinates"""
    point_lines = lines[head:tail]
    points = [point.split() for point in point_lines]
    points = np.array(points).astype(np.float)
    return points


def generate_min_bbox(points):
    box = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])
    return box


def get_landmarks_from_indices(points, indices):
    result = []
    for index in indices:
        val = points[index]
        result.append(val)

    return np.array(result)


class Dataset300W(data.Dataset):
    def __init__(self, dataset_dir, preproc=None, landmark_indices=None):
        self.preproc = preproc
        self.imgs_path = []
        self.annotations_path = []
        self.landmark_indices = landmark_indices

        filelist = os.listdir(dataset_dir)
        for file in filelist:
            file_base, file_ext = os.path.splitext(file)
            if file_ext == ".jpg" or file_ext == ".png":
                image_file = os.path.join(dataset_dir, file)
                pts_file = os.path.join(dataset_dir, file_base + ".pts")
                if not os.path.exists(pts_file):
                    print("pts_file not exist: %s" % pts_file)
                self.imgs_path.append(image_file)
                self.annotations_path.append(pts_file)

        if landmark_indices is None:
            points = load_pts_file(self.annotations_path[0])
            self.landmark_num = points.shape[0]
        else:
            self.landmark_num = len(landmark_indices)
        # annotation_len    bbox: 4, landmark_data: landmark_num * 2, landmark_valid: 1
        self.annotation_len = 4 + self.landmark_num * 2 + 1

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        annotations = np.zeros((0, self.annotation_len))
        points = load_pts_file(self.annotations_path[index])
        bbox = generate_min_bbox(points)

        if self.landmark_indices is not None:
            points = get_landmarks_from_indices(points, self.landmark_indices)

        annotation = np.zeros((1, self.annotation_len))
        annotation[0, 0:4] = bbox
        annotation[0, 4:4 + self.landmark_num*2] = points.flatten()
        annotation[0, -1] = 1

        annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target





