# -*- coding: utf-8 -*-
import glob
import logging
import math
import os
import random
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import json
import sys
import matplotlib.pyplot as plt

sys.path.append('/home/user/Desktop/CFLD_pl/CFLD')
img_dir_replace_from = "D:\\Animal_pose\\AP-36k-patr1\\"
img_dir_replace_to = "/home/user/Desktop/CFLD_pl/CFLD/AP-36k-patr1/"

from pose_utils import (cords_to_map, draw_pose_from_cords, load_pose_cords_from_strings)

logger = logging.getLogger()

class BaseDeepFashion(Dataset):
    def __init__(self, json_file, img_dir, gt_img_size, pose_img_size, cond_img_size, min_scale,
                 log_aspect_ratio, pred_ratio, pred_ratio_var, psz, train=True):
        super().__init__()
        self.pose_img_size = pose_img_size
        self.cond_img_size = cond_img_size
        self.log_aspect_ratio = log_aspect_ratio
        self.pred_ratio = pred_ratio
        self.pred_ratio_var = pred_ratio_var
        self.psz = psz
        self.train = train

        with open(json_file, 'r') as f:
            data = json.load(f)

        self.img_dir = img_dir
        self.images = {img['id']: img for img in data['images']}
        self.annotations = {anno['image_id']: anno for anno in data['annotations'] if anno['category_id'] == 2}

        # 전체 이미지 ID 목록을 90%와 10%로 분할
        total_images = list(self.annotations.keys())
        split_idx = int(len(total_images) * 0.9)
        random.shuffle(total_images)
        self.train_images = total_images[:split_idx]
        self.test_images = total_images[split_idx:]

        self.folder_map = {}
        for img_id, img in self.images.items():
            # Replace backslashes with forward slashes and fix the path
            folder_path = os.path.dirname(img['file_name'].replace("\\", "/").replace(img_dir_replace_from.replace("\\", "/"), img_dir_replace_to))
            if folder_path not in self.folder_map:
                self.folder_map[folder_path] = []
            self.folder_map[folder_path].append(img_id)

        self.transform_gt = transforms.Compose([
            transforms.Resize(gt_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_cond = transforms.Compose([
            transforms.Resize(cond_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        aspect_ratio = cond_img_size[1] / cond_img_size[0]
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(cond_img_size, scale=(min_scale, 1.), ratio=(aspect_ratio * 3. / 4., aspect_ratio * 4. / 3.),
                                         interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) if min_scale < 1.0 else transforms.Compose([
            transforms.Resize(cond_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.skeleton = [[1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [6, 7], [7, 8], [4, 9], [9, 10], [10, 11], [5, 12], [12, 13], [13, 14], [5, 15], [15, 16], [16, 17]]
        self.colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 128, 0), (0, 255, 255), (0, 0, 255), (128, 0, 128), (255, 192, 203), (192, 192, 192), (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 0, 0), (0, 0, 128), (128, 128, 128), (192, 128, 128), (128, 192, 192)]

    def __len__(self):
        return len(self.train_images) if self.train else len(self.test_images)

    def __getitem__(self, idx):
        image_ids = self.train_images if self.train else self.test_images
        image_id = image_ids[idx]
        
        if image_id not in self.annotations:
            #print(f"Annotation for image_id {image_id} not found.")
            return self.__getitem__((idx + 1) % len(image_ids))  # Try next index
        
        image_info = self.images[image_id]
        annotation = self.annotations[image_id]

        # Replace backslashes with forward slashes and fix the path
        img_file_name = image_info['file_name'].replace("\\", "/").replace(img_dir_replace_from.replace("\\", "/"), img_dir_replace_to)
        img_path = os.path.join(self.img_dir, img_file_name)

        if not os.path.exists(img_path):
            #print(f"Image path {img_path} does not exist.")
            return self.__getitem__((idx + 1) % len(image_ids))  # Try next index

        try:
            img_from = Image.open(img_path).convert('RGB')
        except Exception as e:
            #print(f"Error opening image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(image_ids))  # Try next index

        bbox = annotation['bbox']
        bbox_x, bbox_y, bbox_w, bbox_h = map(int, bbox)
        img_from = img_from.crop((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))

        folder_path = os.path.dirname(img_file_name)
        other_image_ids = [id for id in self.folder_map[folder_path] if id != image_id]

        valid_target_found = False
        attempts = 0
        target_image_id = None
        while other_image_ids and not valid_target_found and attempts < len(other_image_ids):
            target_image_id = random.choice(other_image_ids)
            if target_image_id in self.annotations:
                valid_target_found = True
            else:
                other_image_ids.remove(target_image_id)
            attempts += 1

        if not valid_target_found:
            #print(f"No valid target image ID found for image_id {image_id} after {attempts} attempts.")
            return self.__getitem__((idx + 1) % len(image_ids))  # Try next index

        target_image_info = self.images[target_image_id]
        target_annotation = self.annotations[target_image_id]

        # Replace backslashes with forward slashes and fix the path
        target_file_name = target_image_info['file_name'].replace("\\", "/").replace(img_dir_replace_from.replace("\\", "/"), img_dir_replace_to)
        target_img_path = os.path.join(self.img_dir, target_file_name)

        if not os.path.exists(target_img_path):
            #print(f"Target image path {target_img_path} does not exist.")
            return self.__getitem__((idx + 1) % len(image_ids))  # Try next index

        try:
            img_to = Image.open(target_img_path).convert('RGB')
        except Exception as e:
            #print(f"Error opening target image {target_img_path}: {e}")
            return self.__getitem__((idx + 1) % len(image_ids))  # Try next index

        target_bbox = target_annotation['bbox']
        target_bbox_x, target_bbox_y, target_bbox_w, target_bbox_h = map(int, target_bbox)
        img_to = img_to.crop((target_bbox_x, target_bbox_y, target_bbox_x + target_bbox_w, target_bbox_y + target_bbox_h))

        img_src = self.transform_gt(img_from)
        img_tgt = self.transform_gt(img_to)
        img_cond = self.transform(img_from)

        pose_img_src = self.build_pose_img(annotation, bbox_w, bbox_h)
        pose_img_tgt = self.build_pose_img(target_annotation, target_bbox_w, target_bbox_h)

        return_dict = {
            "img_src": img_src,
            "img_tgt": img_tgt,
            "img_cond": img_cond,
            "pose_img_src": pose_img_src,
            "pose_img_tgt": pose_img_tgt
        }
        return return_dict

    def build_pose_img(self, annotation, img_width, img_height):
        keypoints = np.array(annotation['keypoints']).reshape(-1, 3)
        keypoints = keypoints[:, :2]
        #print(f"keypoints :{keypoints} ")

        keypoints[:, 0] = np.clip(keypoints[:, 0] - annotation['bbox'][0], 0, img_width) * self.pose_img_size[1] / img_width
        keypoints[:, 1] = np.clip(keypoints[:, 1] - annotation['bbox'][1], 0, img_height) * self.pose_img_size[0] / img_height

        pose_map = cords_to_map(keypoints, tuple(self.pose_img_size), (img_width, img_height)).transpose(2, 0, 1)
        pose_img = draw_pose_from_cords(keypoints, tuple(self.pose_img_size), (img_width, img_height), self.skeleton, self.colors)
        #print(f"pose_map.shape: {pose_map.shape}, pose_img.shape: {pose_img.shape}")

        pose_map_tensor = torch.tensor(pose_map, dtype=torch.float32)
        pose_img_tensor = torch.tensor(pose_img.transpose(2, 0, 1), dtype=torch.float32)
        pose_img = torch.cat([pose_img_tensor, pose_map_tensor], dim=0)

        return pose_img

    def get_pred_ratio(self):
        pred_ratio = []
        for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
            assert prm >= prv
            pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
            pred_ratio.append(pr)
        pred_ratio = random.choice(pred_ratio)
        return pred_ratio

class PisTrainDeepFashion(BaseDeepFashion):
    def __init__(self, json_file, img_dir, gt_img_size, pose_img_size, cond_img_size, min_scale,
                 log_aspect_ratio, pred_ratio, pred_ratio_var, psz):
        super().__init__(json_file, img_dir, gt_img_size, pose_img_size, cond_img_size, min_scale,
                         log_aspect_ratio, pred_ratio, pred_ratio_var, psz, train=True)

class PisTestDeepFashion(BaseDeepFashion):
    def __init__(self, json_file, img_dir, gt_img_size, pose_img_size, cond_img_size, min_scale,
                 log_aspect_ratio, pred_ratio, pred_ratio_var, psz):
        super().__init__(json_file, img_dir, gt_img_size, pose_img_size, cond_img_size, min_scale,
                         log_aspect_ratio, pred_ratio, pred_ratio_var, psz, train=False)



class FidRealDeepFashion(Dataset):
    def __init__(self, root_dir, test_img_size):
        super().__init__()
        self.img_items = self.process_dir(root_dir)

        self.transform_test = transforms.Compose([
            transforms.Resize(test_img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor()
        ])

    def process_dir(self, root_dir):
        data = []
        img_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
        for img_path in img_paths:
            data.append(img_path)
        return data

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path = self.img_items[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return self.transform_test(img)

# 시각화 함수
def visualize_sample(sample):
    if sample is None:
        print("Sample is None")
        return

    img_src = sample['img_src']
    img_tgt = sample['img_tgt']
    img_cond = sample['img_cond']
    pose_img_src = sample['pose_img_src']
    pose_img_tgt = sample['pose_img_tgt']

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    if img_src is not None:
        axes[0].imshow(img_src.permute(1, 2, 0).numpy())
        axes[0].set_title('Source Image')
    else:
        axes[0].set_title('Source Image (None)')
    axes[0].axis('off')

    if img_tgt is not None:
        axes[1].imshow(img_tgt.permute(1, 2, 0).numpy())
        axes[1].set_title('Target Image')
    else:
        axes[1].set_title('Target Image (None)')
    axes[1].axis('off')

    if img_cond is not None:
        axes[2].imshow(img_cond.permute(1, 2, 0).numpy())
        axes[2].set_title('Condition Image')
    else:
        axes[2].set_title('Condition Image (None)')
    axes[2].axis('off')

    if pose_img_src is not None:
        axes[3].imshow(pose_img_src[:3].permute(1, 2, 0).numpy())
        axes[3].set_title('Source Pose')
    else:
        axes[3].set_title('Source Pose (None)')
    axes[3].axis('off')

    if pose_img_tgt is not None:
        axes[4].imshow(pose_img_tgt[:3].permute(1, 2, 0).numpy())
        axes[4].set_title('Target Pose')
    else:
        axes[4].set_title('Target Pose (None)')
    axes[4].axis('off')

    plt.show()

json_file = "/home/user/Desktop/CFLD_pl/CFLD/AP-36k-patr1/apt36k_annotations.json"
img_dir = "/home/user/Desktop/CFLD_pl/CFLD/AP-36k-patr1/2dog"

gt_img_size = (256, 256)
pose_img_size = (256, 256)
cond_img_size = (256, 256)
min_scale = 0.5
log_aspect_ratio = (0.75, 1.3333)
pred_ratio = [0.15, 0.3, 0.45]
pred_ratio_var = [0.1, 0.1, 0.1]
psz = 16

dataset = PisTestDeepFashion(json_file, img_dir, gt_img_size, pose_img_size, cond_img_size, min_scale, log_aspect_ratio, pred_ratio, pred_ratio_var, psz)
sample = dataset[38]
#visualize_sample(sample)
