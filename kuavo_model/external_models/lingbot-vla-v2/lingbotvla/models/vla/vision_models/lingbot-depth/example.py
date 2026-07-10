import cv2
import torch
import numpy as np
import trimesh
import os
from pathlib import Path
from mdm.model.v2 import MDMModel as v2

def preprocess_input_image(image_path, device):
    """
    Preprocess input image

    Args:
        image_path (str): Image path
        device (torch.device): Device

    Returns:
        tuple: (numpy_image, tensor_image) Image in numpy and tensor format
    """
    # Read image and convert to RGB format
    image_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # Convert to tensor and normalize to [0, 1] range
    image_tensor = torch.tensor(image_np / 255, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    return image_np, image_tensor

def load_depth_map(depth_path, scale=1000.0):
    """
    Load depth map and convert to meters

    Args:
        depth_path (str): Depth map path

    Returns:
        np.ndarray: Depth map (in meters)
    """
    # Read depth map and convert to meters (original unit is millimeters)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / scale
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    return depth_map

def depth_to_color_opencv(depth_map, vmin=None, vmax=None, colormap=cv2.COLORMAP_TURBO):
    """
    Convert depth map using OpenCV colormap (faster)

    Args:
        depth_map: (H, W) numpy array
        colormap: cv2.COLORMAP_TURBO, cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS, etc.

    Returns:
        (H, W, 3) numpy array, BGR, 0-255
    """
    # Handle invalid values
    valid_mask = np.isfinite(depth_map)
    depth_clean = depth_map.copy()
    depth_clean[~valid_mask] = 0

    if vmin is None:
        vmin = depth_clean[valid_mask].min() if valid_mask.any() else 0
    if vmax is None:
        vmax = depth_clean[valid_mask].max() if valid_mask.any() else 1

    # Normalize to [0, 255]
    depth_normalized = np.clip((depth_clean - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)

    # Handle invalid values
    depth_colored[~valid_mask] = [0, 0, 0]

    return depth_colored

DEVICE = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = 'ckpt/model.pt' 
rgb_path = 'examples/0/rgb.png'
depth_path = 'examples/0/raw_depth.png'
intrinsics_path = 'examples/0/intrinsics.txt'


intrinsics = np.loadtxt(intrinsics_path)
intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=DEVICE)
image_np, image_tensor = preprocess_input_image(rgb_path, DEVICE)
depth_np = load_depth_map(depth_path)
depth_tensor = torch.tensor(depth_np, dtype=torch.float32, device=DEVICE)

h, w = image_np.shape[:2]
intrinsics[0] /= w
intrinsics[1] /= h

model = v2.from_pretrained(ckpt_path).to(DEVICE)

output = model.infer(
    image_tensor, 
    depth_in=depth_tensor, 
    apply_mask=True, 
    intrinsics=intrinsics[None]
    )

depth_pred = output['depth'].squeeze().cpu().numpy()

res_dir = Path('result')
res_dir.mkdir(exist_ok=True)
# save depth map
depth_raw_color = depth_to_color_opencv(depth_np)
depth_pred_color = depth_to_color_opencv(depth_pred)
depth_concat = np.concatenate([depth_raw_color, depth_pred_color], axis=1)
cv2.imwrite(res_dir/'res.png', depth_concat)

# save pcd
points_pred = output['points'].squeeze().cpu().numpy()
verts = points_pred.reshape(-1, 3)[::2]
verts_color = image_np.reshape(-1, 3)[::2]
point_cloud = trimesh.PointCloud(verts, verts_color)
point_cloud.export(res_dir/'pcd.ply')