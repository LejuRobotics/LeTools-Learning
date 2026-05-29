#!/usr/bin/env python3
"""
LingBot-Depth Example Script

Demonstrates depth refinement using the LingBot-Depth model.
Supports both local checkpoints and Hugging Face models.

Usage:
    python example.py
    python example.py --model robbyant/lingbot-depth-postrain-dc-vitl14
    python example.py --example 0 --output custom_output
"""

import cv2
import torch
import numpy as np
import trimesh
import argparse
import time
from pathlib import Path
from mdm.model.v2 import MDMModel


def preprocess_input_image(image_path, device):
    """
    Load and preprocess RGB image.

    Args:
        image_path (str): Path to RGB image
        device (torch.device): Device to load tensor on

    Returns:
        tuple: (numpy_image, tensor_image)
            - numpy_image: RGB numpy array (H, W, 3), uint8
            - tensor_image: RGB tensor (1, 3, H, W), float32, [0,1]
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image and convert BGR to RGB
    image_np = cv2.imread(image_path)
    if image_np is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Convert to tensor and normalize to [0, 1]
    image_tensor = torch.tensor(
        image_np / 255.0,
        dtype=torch.float32,
        device=device
    ).permute(2, 0, 1).unsqueeze(0)

    return image_np, image_tensor


def load_depth_map(depth_path, scale=1000.0):
    """
    Load depth map from PNG file (16-bit) and convert to meters.

    Args:
        depth_path (str): Path to depth image
        scale (float): Scale factor to convert to meters
            - 1000.0 for millimeters
            - 1.0 for meters

    Returns:
        np.ndarray: Depth map in meters (H, W), float32
    """
    if not Path(depth_path).exists():
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    # Read depth map as 16-bit
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise ValueError(f"Failed to read depth map: {depth_path}")

    # Convert to meters
    depth_map = depth_map.astype(np.float32) / scale

    # Replace invalid values with 0
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

    return depth_map

def load_intrinsics(intrinsics_path, width, height):
    """
    Load camera intrinsics and normalize by image dimensions.

    Args:
        intrinsics_path (str): Path to intrinsics file (.txt or .json)
        width (int): Image width
        height (int): Image height

    Returns:
        np.ndarray: Normalized intrinsics matrix (3, 3)
    """
    if not Path(intrinsics_path).exists():
        raise FileNotFoundError(f"Intrinsics not found: {intrinsics_path}")

    # Load intrinsics
    if intrinsics_path.endswith('.json'):
        import json
        with open(intrinsics_path, 'r') as f:
            intrinsics = np.array(json.load(f), dtype=np.float32)
    else:
        intrinsics = np.loadtxt(intrinsics_path, dtype=np.float32)

    # Normalize by image dimensions
    intrinsics_normalized = intrinsics.copy()
    intrinsics_normalized[0, 0] /= width   # fx
    intrinsics_normalized[0, 2] /= width   # cx
    intrinsics_normalized[1, 1] /= height  # fy
    intrinsics_normalized[1, 2] /= height  # cy

    return intrinsics_normalized


def depth_to_color_opencv(depth_map, vmin=None, vmax=None, colormap=cv2.COLORMAP_TURBO):
    """
    Convert depth map to color visualization using OpenCV colormap.

    Args:
        depth_map (np.ndarray): Depth map (H, W)
        vmin (float): Minimum depth for colormap (auto if None)
        vmax (float): Maximum depth for colormap (auto if None)
        colormap: OpenCV colormap (TURBO, JET, VIRIDIS, etc.)

    Returns:
        np.ndarray: Colored depth map (H, W, 3) in BGR format
    """
    # Handle invalid values
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    depth_clean = depth_map.copy()
    depth_clean[~valid_mask] = 0

    # Auto-range if not specified
    if vmin is None:
        vmin = depth_clean[valid_mask].min() if valid_mask.any() else 0
    if vmax is None:
        vmax = depth_clean[valid_mask].max() if valid_mask.any() else 1

    # Normalize to [0, 255]
    depth_normalized = np.clip(
        (depth_clean - vmin) / (vmax - vmin + 1e-8) * 255,
        0, 255
    ).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)

    # Set invalid pixels to black
    depth_colored[~valid_mask] = [0, 0, 0]

    return depth_colored

def main():
    """Main function with argument parsing and execution."""
    parser = argparse.ArgumentParser(
        description='LingBot-Depth Example: Refine depth maps using the model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (example 0)
  python example.py

  # Use a different example directory
  python example.py --example 1

  # Use Hugging Face model
  python example.py --model robbyant/lingbot-depth-postrain-dc-vitl14

  # Use local checkpoint
  python example.py --model ckpt/model.pt

  # Custom output directory
  python example.py --output my_results
        """
    )

    parser.add_argument(
        '--example', type=str, default='0',
        help='Example directory number (default: 0)'
    )
    parser.add_argument(
        '--model', type=str,
        default='robbyant/lingbot-depth-pretrain-vitl-14-v0.5',
        help='Model path or Hugging Face ID (default: robbyant/lingbot-depth-pretrain-vitl-14)'
    )
    parser.add_argument(
        '--output', type=str, default='result',
        help='Output directory (default: result)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto - uses CUDA if available)'
    )
    parser.add_argument(
        '--no-mask', action='store_true',
        help='Disable masking of invalid regions'
    )

    args = parser.parse_args()

    # Print header
    print("=" * 70)
    print("LingBot-Depth Example: Depth Refinement".center(70))
    print("=" * 70)

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n📱 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Setup paths
    example_dir = Path('examples') / args.example

    # Check if example directory exists first
    if not example_dir.exists():
        print(f"\n❌ Error: Example directory not found: {example_dir}")
        print(f"\n   Available examples:")
        examples_root = Path('examples')
        if examples_root.exists():
            for ex in sorted(examples_root.iterdir()):
                if ex.is_dir():
                    # Check what files exist in this example
                    rgb_file = None
                    for ext in ['.png', '.jpg', '.jpeg']:
                        if (ex / f'rgb{ext}').exists():
                            rgb_file = f'rgb{ext}'
                            break
                    depth_ok = (ex / 'raw_depth.png').exists()
                    intrinsics_ok = (ex / 'intrinsics.txt').exists()
                    status = '✓' if all([rgb_file, depth_ok, intrinsics_ok]) else '⚠'
                    print(f"     {status} {ex.name}")
        return 1

    # Auto-detect RGB file format (.png or .jpg)
    rgb_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = example_dir / f'rgb{ext}'
        if candidate.exists():
            rgb_path = candidate
            break

    depth_path = example_dir / 'raw_depth.png'
    intrinsics_path = example_dir / 'intrinsics.txt'

    # Check for required files
    missing_files = []
    if rgb_path is None:
        missing_files.append('rgb.{png,jpg,jpeg}')
    if not depth_path.exists():
        missing_files.append('raw_depth.png')
    if not intrinsics_path.exists():
        missing_files.append('intrinsics.txt')

    if missing_files:
        print(f"\n❌ Error: Missing required files in {example_dir}:")
        for f in missing_files:
            print(f"   - {f}")
        return 1

    print(f"\n📂 Input:")
    print(f"   RGB:        {rgb_path}")
    print(f"   Depth:      {depth_path}")
    print(f"   Intrinsics: {intrinsics_path}")

    try:
        # Load data
        print(f"\n📥 Loading data...")
        image_np, image_tensor = preprocess_input_image(str(rgb_path), device)
        depth_np = load_depth_map(str(depth_path))
        depth_tensor = torch.tensor(depth_np, dtype=torch.float32, device=device)

        h, w = image_np.shape[:2]
        print(f"   Image size: {w}×{h}")
        print(f"   Input depth range: {depth_np[depth_np > 0].min():.2f} - {depth_np.max():.2f} meters")

        # Load intrinsics
        intrinsics = load_intrinsics(str(intrinsics_path), w, h)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32, device=device).unsqueeze(0)

        # Load model
        print(f"\n🤖 Loading model: {args.model}")
        start_time = time.time()
        model = MDMModel.from_pretrained(args.model).to(device)
        load_time = time.time() - start_time
        print(f"   ✓ Model loaded in {load_time:.2f}s")

        # Run inference
        print(f"\n🔄 Running inference...")
        start_time = time.time()
        with torch.no_grad():
            output = model.infer(
                image_tensor,
                depth_in=depth_tensor,
                apply_mask=not args.no_mask,
                intrinsics=intrinsics_tensor
            )
        inference_time = time.time() - start_time

        depth_pred = output['depth'].squeeze().cpu().numpy()
        points_pred = output['points'].squeeze().cpu().numpy()

        print(f"   ✓ Inference completed in {inference_time:.3f}s")
        print(f"   Refined depth range: {depth_pred[depth_pred > 0].min():.2f} - {depth_pred.max():.2f} meters")

        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)

        print(f"\n💾 Saving results to: {output_dir}/")

        # 1. Save depth maps as numpy arrays
        np.save(output_dir / 'depth_input.npy', depth_np)
        np.save(output_dir / 'depth_refined.npy', depth_pred)
        print(f"   ✓ Depth arrays saved (.npy)")

        # 2. Save depth visualizations
        depth_raw_color = depth_to_color_opencv(depth_np)
        depth_pred_color = depth_to_color_opencv(depth_pred)
        depth_concat = np.concatenate([depth_raw_color, depth_pred_color], axis=1)

        cv2.imwrite(str(output_dir / 'depth_input.png'), depth_raw_color)
        cv2.imwrite(str(output_dir / 'depth_refined.png'), depth_pred_color)
        cv2.imwrite(str(output_dir / 'depth_comparison.png'), depth_concat)
        print(f"   ✓ Depth visualizations saved (.png)")

        # 3. Save RGB image for reference
        cv2.imwrite(str(output_dir / 'rgb.png'), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        # 4. Save point cloud
        valid_mask = np.isfinite(points_pred).all(axis=-1) & (points_pred[..., 2] > 0)
        verts = points_pred[valid_mask]
        verts_color = image_np[valid_mask]

        # Downsample for reasonable file size
        downsample = 2
        verts = verts[::downsample]
        verts_color = verts_color[::downsample]

        point_cloud = trimesh.PointCloud(verts, verts_color)
        point_cloud.export(output_dir / 'point_cloud.ply')
        print(f"   ✓ Point cloud saved ({len(verts):,} points)")

        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Model load time:   {load_time:.2f}s")
        print(f"   Inference time:    {inference_time:.3f}s")
        print(f"   Valid points:      {valid_mask.sum():,} / {valid_mask.size:,}")

        print(f"\n✅ Done! Results saved to: {output_dir}/")
        print("=" * 70)

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

