import numpy as np
import torch
from PIL import Image, ImageDraw

def load_pose_cords_from_strings(y_str, x_str):
    y_coords = list(map(float, y_str.split(',')))
    x_coords = list(map(float, x_str.split(',')))
    return np.array([x_coords, y_coords]).T
def cords_to_map(coords, output_size, input_size):
    map = np.zeros((output_size[0], output_size[1], len(coords)))
    for i, (x, y) in enumerate(coords):
        if np.isnan(x) or np.isnan(y):
            continue
        x = int(x * output_size[1] / input_size[1])
        y = int(y * output_size[0] / input_size[0])
        if 0 <= x < output_size[1] and 0 <= y < output_size[0]:
            map[y, x, i] = 1
    return map
def draw_pose_from_cords(coords, output_size, input_size, skeleton, colors):
    pose_img = Image.new('RGB', output_size, (0, 0, 0))  # 배경을 검정색으로 설정
    draw = ImageDraw.Draw(pose_img)
    coords = coords.astype(int)
    
    # Draw the skeleton
    for (start_idx, end_idx), color in zip(skeleton, colors):
        start_idx -= 1
        end_idx -= 1
        if all(0 <= coords[idx, 0] < output_size[1] and 0 <= coords[idx, 1] < output_size[0] for idx in [start_idx, end_idx]) and all(coords[idx, 0] > 0 and coords[idx, 1] > 0 for idx in [start_idx, end_idx]):
            draw.line([tuple(coords[start_idx]), tuple(coords[end_idx])], fill=color, width=3)

    # Draw the keypoints
    for x, y in coords:
        if 0 <= x < output_size[1] and 0 <= y < output_size[0] and (x, y) != (0, 0):
            draw.ellipse((x-4, y-4, x+4, y+4), fill=(255, 0, 0))  # 키포인트를 빨간색으로 그리기
    
    return np.array(pose_img)
