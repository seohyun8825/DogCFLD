import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json

class DogKeypointDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.img_dir = img_dir
        self.transform = transform
        self.images = {img['id']: img for img in data['images']}
        self.annotations = {anno['image_id']: anno for anno in data['annotations'] if anno['category_id'] == 2}

        # 폴더 기준으로 이미지들 그룹화
        self.folder_map = {}
        for img_id, img in self.images.items():
            folder_path = os.path.dirname(img['file_name'])
            if folder_path not in self.folder_map:
                self.folder_map[folder_path] = []
            self.folder_map[folder_path].append(img_id)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_ids = list(self.annotations.keys())
        image_id = image_ids[idx]
        image_info = self.images[image_id]
        annotation = self.annotations[image_id]
        img_path = os.path.join(self.img_dir, image_info['file_name'].replace("D:\\Animal_pose\\AP-36k-patr1\\", ""))
        
        image = Image.open(img_path).convert('RGB') if os.path.exists(img_path) else None
        bbox = annotation['bbox']

        # 바운딩 박스 내의 이미지만 크롭
        if image:
            bbox_x, bbox_y, bbox_w, bbox_h = map(int, bbox)
            image = image.crop((bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h))

        folder_path = os.path.dirname(image_info['file_name'])
        other_image_ids = [id for id in self.folder_map[folder_path] if id != image_id]
        target_image_id = random.choice(other_image_ids)
        target_image_info = self.images[target_image_id]
        target_annotation = self.annotations[target_image_id]
        target_img_path = os.path.join(self.img_dir, target_image_info['file_name'].replace("D:\\Animal_pose\\AP-36k-patr1\\", ""))

        target_image = Image.open(target_img_path).convert('RGB') if os.path.exists(target_img_path) else None
        target_bbox = target_annotation['bbox']

        # 타겟 이미지도 같은 방식으로 크롭
        if target_image:
            target_bbox_x, target_bbox_y, target_bbox_w, target_bbox_h = map(int, target_bbox)
            target_image = target_image.crop((target_bbox_x, target_bbox_y, target_bbox_x + target_bbox_w, target_bbox_y + target_bbox_h))

        if self.transform:
            image = self.transform(image) if image is not None else None
            target_image = self.transform(target_image) if target_image is not None else None

        # 키포인트를 조정 (바운딩 박스 상대 좌표로 변경)
        target_keypoints = torch.tensor(target_annotation['keypoints'], dtype=torch.float32).view(-1, 3)
        target_keypoints[:, 0] -= target_bbox_x
        target_keypoints[:, 1] -= target_bbox_y

        return image, target_keypoints[:, :2].flatten(), target_image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
