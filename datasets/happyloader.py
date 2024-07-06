import glob
import logging
import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import json
import matplotlib.pyplot as plt

# 로거 설정
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

class HappyDataset(Dataset):
    def __init__(self, img_dir, gt_img_size, pose_img_size, cond_img_size, min_scale, log_aspect_ratio, train=True):
        super().__init__()
        self.img_dir = img_dir
        self.pose_img_size = pose_img_size
        self.cond_img_size = cond_img_size
        self.gt_img_size = gt_img_size
        self.train = train
        
        self.images = glob.glob(os.path.join(img_dir, '*.jpg'))
        self.annotations = {os.path.splitext(os.path.basename(img))[0]: img.replace('.jpg', '.json') for img in self.images}
        
        total_images = list(self.annotations.keys())
        split_idx = int(len(total_images) * 1.0)
        random.shuffle(total_images)
        self.train_images = total_images[:split_idx]
        self.test_images = total_images[split_idx:]
        
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

        self.skeleton = [
            ["nose", "right_eye"], ["nose", "left_eye"], ["right_eye", "right_earbase"], ["left_eye", "left_earbase"],
            ["right_earbase", "right_earend"], ["left_earbase", "left_earend"], ["nose", "tail_base"],
            ["tail_base", "tail_end"], ["right_earbase", "right_antler_base"], ["right_antler_base", "right_antler_end"],
            ["left_earbase", "left_antler_base"], ["left_antler_base", "left_antler_end"], ["front_left_thai", "front_left_knee"],
            ["front_left_knee", "front_left_paw"], ["front_right_thai", "front_right_knee"], ["front_right_knee", "front_right_paw"],
            ["back_left_thai", "back_left_knee"], ["back_left_knee", "back_left_paw"], ["back_right_thai", "back_right_knee"],
            ["back_right_knee", "back_right_paw"], ["belly_bottom", "body_middle_left"], ["body_middle_left", "body_middle_right"]
        ]
        
        self.colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 128, 0), (0, 255, 255), (0, 0, 255), 
                       (128, 0, 128), (255, 192, 203), (192, 192, 192), (128, 128, 0), (128, 0, 128), 
                       (0, 128, 128), (128, 0, 0), (0, 0, 128), (128, 128, 128), (192, 128, 128), 
                       (128, 192, 192)]

    def __len__(self):
        return len(self.train_images) if self.train else len(self.test_images)

    def __getitem__(self, idx):
        image_ids = self.train_images if self.train else self.test_images
        image_id = image_ids[idx]
        img_path = os.path.join(self.img_dir, image_id + '.jpg')
        anno_path = os.path.join(self.img_dir, image_id + '.json')

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error opening image {img_path}. Error: {e}")
            raise

        with open(anno_path, 'r') as f:
            annotation = json.load(f)

        # 원본 이미지 크기 저장
        original_width, original_height = img.size
        
        # 이미지와 주석을 640x640 크기로 변환
        img_resized = img.resize((640, 640), Image.BICUBIC)

        bbox = annotation['bb_0']['corner_1'] + annotation['bb_0']['corner_2']
        bbox_x, bbox_y, bbox_x2, bbox_y2 = map(int, bbox)
        
        # 640x640 크기의 이미지에서 주석에 맞게 잘라내기
        img_cropped = img_resized.crop((bbox_x, bbox_y, bbox_x2, bbox_y2))
        
        # 원하는 크기로 변환
        img_cropped = img_cropped.resize(self.gt_img_size, Image.BICUBIC)

        img_src = self.transform_gt(img_cropped)
        img_cond = self.transform(img_cropped)
        pose_img_src = self.build_pose_img(annotation['bb_0']['dlc_pred'], bbox_x2 - bbox_x, bbox_y2 - bbox_y)

        # Target image 선택
        target_idx = random.choice([i for i in range(len(image_ids)) if i != idx])
        target_image_id = image_ids[target_idx]
        target_img_path = os.path.join(self.img_dir, target_image_id + '.jpg')
        target_anno_path = os.path.join(self.img_dir, target_image_id + '.json')

        try:
            target_img = Image.open(target_img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error opening target image {target_img_path}. Error: {e}")
            raise

        with open(target_anno_path, 'r') as f:
            target_annotation = json.load(f)
        
        # 원본 타겟 이미지 크기 저장
        target_original_width, target_original_height = target_img.size
        
        # 타겟 이미지를 640x640 크기로 변환
        target_img_resized = target_img.resize((640, 640), Image.BICUBIC)

        target_bbox = target_annotation['bb_0']['corner_1'] + target_annotation['bb_0']['corner_2']
        target_bbox_x, target_bbox_y, target_bbox_x2, target_bbox_y2 = map(int, target_bbox)
        
        # 640x640 크기의 타겟 이미지에서 주석에 맞게 잘라내기
        target_img_cropped = target_img_resized.crop((target_bbox_x, target_bbox_y, target_bbox_x2, target_bbox_y2))
        
        # 원하는 크기로 변환
        target_img_cropped = target_img_cropped.resize(self.gt_img_size, Image.BICUBIC)

        img_tgt = self.transform_gt(target_img_cropped)
        pose_img_tgt = self.build_pose_img(target_annotation['bb_0']['dlc_pred'], target_bbox_x2 - target_bbox_x, target_bbox_y2 - target_bbox_y)

        return {
            "img_src": img_src,
            "img_tgt": img_tgt,
            "img_cond": img_cond,
            "pose_img_src": pose_img_src,
            "pose_img_tgt": pose_img_tgt,
            "img_gt": img_tgt 
        }

    def build_pose_img(self, keypoints, img_width, img_height):
        keypoints = {k: v for k, v in keypoints.items() if not any(np.isnan(v))}
        keypoints_arr = np.array([keypoints[k][:2] for k in keypoints])
        keypoints_arr[:, 0] = np.clip(keypoints_arr[:, 0], 0, img_width) * self.pose_img_size[1] / img_width
        keypoints_arr[:, 1] = np.clip(keypoints_arr[:, 1], 0, img_height) * self.pose_img_size[0] / img_height

        pose_map = np.zeros((self.pose_img_size[0], self.pose_img_size[1], 20), dtype=np.float32)  # 20채널로 수정
        for i, (x, y) in enumerate(keypoints_arr):
            if not np.isnan(x) and not np.isnan(y):
                pose_map[int(y), int(x), :3] = 1.0  # RGB 채널
                if i < 17:  # 17개의 키포인트를 3번 채널부터 19번 채널에 할당
                    pose_map[int(y), int(x), 3 + i] = 1.0
        
        for joint in self.skeleton:
            joint_idx = [list(keypoints.keys()).index(j) for j in joint if j in keypoints]
            if len(joint_idx) == 2:
                x1, y1 = keypoints_arr[joint_idx[0]]
                x2, y2 = keypoints_arr[joint_idx[1]]
                pose_map = self.draw_line(pose_map, x1, y1, x2, y2, joint_idx[0])

        pose_map_tensor = torch.tensor(pose_map.transpose(2, 0, 1), dtype=torch.float32)

        return pose_map_tensor

    def draw_line(self, image, x1, y1, x2, y2, joint_idx, thickness=2):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if joint_idx < 17:  # 17개의 키포인트를 3번 채널부터 19번 채널에 할당
            if x1 == x2:
                for y in range(min(y1, y2), max(y1, y2)):
                    image[y-thickness:y+thickness, x1-thickness:x1+thickness, 3 + joint_idx] = 1.0  # 추가 채널에 선을 그림
            elif y1 == y2:
                for x in range(min(x1, x2), max(x1, x2)):
                    image[y1-thickness:y1+thickness, x-thickness:x+thickness, 3 + joint_idx] = 1.0  # 추가 채널에 선을 그림
            else:
                for t in np.linspace(0, 1, max(abs(x2 - x1), abs(y2 - y1))):
                    x = int(x1 * (1 - t) + x2 * t)
                    y = int(y1 * (1 - t) + y2 * t)
                    image[y-thickness:y+thickness, x-thickness:x+thickness, 3 + joint_idx] = 1.0  # 추가 채널에 선을 그림
        return image

# 설정
img_dir = "/home/user/Desktop/CFLD_pl/CFLD/happy"
gt_img_size = (256, 256)
pose_img_size = (256, 256)
cond_img_size = (256, 256)
min_scale = 0.5
log_aspect_ratio = (0.75, 1.3333)

# 데이터셋 생성
dataset = HappyDataset(img_dir, gt_img_size, pose_img_size, cond_img_size, min_scale, log_aspect_ratio, train=True)
sample = dataset[0]

def visualize_sample(sample, save_path=None):
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
        axes[0].imshow(img_src.permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[0].set_title('Source Image')
    else:
        axes[0].set_title('Source Image (None)')
    axes[0].axis('off')

    if img_tgt is not None:
        axes[1].imshow(img_tgt.permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[1].set_title('Target Image')
    else:
        axes[1].set_title('Target Image (None)')
    axes[1].axis('off')

    if img_cond is not None:
        axes[2].imshow(img_cond.permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[2].set_title('Condition Image')
    else:
        axes[2].set_title('Condition Image (None)')
    axes[2].axis('off')

    if pose_img_src is not None:
        axes[3].imshow(pose_img_src[:3].permute(1, 2, 0).numpy() * 0.5 + 0.5)  # RGB 채널만 시각화
        axes[3].set_title('Source Pose')
    else:
        axes[3].set_title('Source Pose (None)')
    axes[3].axis('off')

    if pose_img_tgt is not None:
        axes[4].imshow(pose_img_tgt[:3].permute(1, 2, 0).numpy() * 0.5 + 0.5)  # RGB 채널만 시각화
        axes[4].set_title('Target Pose')
    else:
        axes[4].set_title('Target Pose (None)')
    axes[4].axis('off')

    if save_path:
        plt.savefig(save_path)
        print(f"Sample saved to {save_path}")
    plt.show()

visualize_sample(sample, save_path="sample_visualization.png")
