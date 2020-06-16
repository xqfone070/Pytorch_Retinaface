import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class Dataset300W(data.Dataset):
    def __init__(self, dataset_dir, preproc=None, landmark_num=68):
        self.preproc = preproc
        self.imgs_path = []
        self.annotations_path = []
        self.landmark_num = landmark_num
        # annotation_len    bbox: 4, landmark_data: landmark_num * 2, landmark_valid: 1
        self.annotation_len = 4 + landmark_num * 2 + 1

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

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        annotations = np.zeros((0, self.annotation_len))
        points = self.load_annotation(self.annotations_path[index])
        assert points.shape[0] == self.landmark_num
        bbox = self.generate_bbox(points)

        annotation = np.zeros((1, self.annotation_len))
        annotation[0, 0:4] = bbox
        annotation[0, 4:4 + self.landmark_num*2] = points.flatten()
        annotation[0, -1] = 1

        annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

    @classmethod
    def load_annotation(cls, filename):
        with open(filename) as f:
            lines = [line.strip() for line in f]

        """Use the curly braces to find the start and end of the point data"""
        head = lines.index('{') + 1
        tail = lines.index('}')

        """Select the point data split into coordinates"""
        point_lines = lines[head:tail]
        points = [point.split() for point in point_lines]

        points = []
        for point in point_lines:
            pt = [float(x) for x in point.split()]
            points.append(pt)

        return np.array(points)

    @classmethod
    def generate_bbox(cls, points):
        box = np.array([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])
        return box

