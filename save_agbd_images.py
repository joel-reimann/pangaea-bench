"""
save_agbd_images.py

Flexible script to save and visualize AGBD regression/segmentation results from tensors or numpy arrays.
- Saves images to /cluster/scratch/reimannj/agbd_images/ by default.
- Supports batch processing, custom colormaps, overlays, and high-res output.
- Can be called post-hoc on saved tensors/npys, or from evaluator after evaluation.

Usage (example):
    python save_agbd_images.py --input_dir /cluster/scratch/reimannj/agbd_tensors/ --output_dir /cluster/scratch/reimannj/agbd_images/ --split val --max_images 50

"""
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import cm
from PIL import Image
import random
import sys

# Sentinel-2 band names for AGBD/Prithvi convention
BANDS_PRITHVI = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']

def save_image(arr, out_path, cmap=None, vmin=None, vmax=None, overlay=None, alpha=0.4, resize=None):
    """Save a numpy array as an image, with optional colormap and overlay."""
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    arr = arr.astype(np.float32)  # Ensure float for normalization
    # Handle (C, H, W) to (H, W, C)
    if arr.ndim == 3 and arr.shape[0] in [1, 3]:
        arr = arr.transpose(1, 2, 0)
    # Normalize and apply colormap if needed
    if arr.ndim == 2:
        if cmap:
            normed = (arr - (vmin if vmin is not None else arr.min())) / ((vmax if vmax is not None else arr.max()) - (vmin if vmin is not None else arr.min()) + 1e-6)
            normed = np.clip(normed, 0, 1)
            arr = plt.get_cmap(cmap)(normed)[:, :, :3]  # drop alpha
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
    elif arr.ndim == 3:
        arr = np.clip(arr, 0, 255)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
    else:
        raise ValueError(f"Cannot handle array with shape {arr.shape} for image saving.")
    if resize:
        arr = np.array(Image.fromarray(arr).resize(resize, resample=Image.BILINEAR))
    img = Image.fromarray(arr)
    if overlay is not None:
        overlay = np.asarray(overlay)
        overlay = np.squeeze(overlay)
        if overlay.shape != arr.shape:
            overlay = np.array(Image.fromarray(overlay).resize(img.size, resample=Image.BILINEAR))
        overlay_img = Image.fromarray(overlay).convert('RGBA')
        img = Image.blend(img.convert('RGBA'), overlay_img, alpha=alpha)
    img.save(out_path)

def print_tensor_stats(name, arr):
    arr_np = arr.cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    arr_np = np.squeeze(arr_np)
    print(f"--- {name} ---", file=sys.stderr)
    print(f"  shape: {arr_np.shape}", file=sys.stderr)
    print(f"  dtype: {arr_np.dtype}", file=sys.stderr)
    print(f"  min: {np.nanmin(arr_np):.4f}, max: {np.nanmax(arr_np):.4f}, mean: {np.nanmean(arr_np):.4f}", file=sys.stderr)
    print(f"  NaNs: {np.isnan(arr_np).sum()}, Infs: {np.isinf(arr_np).sum()}", file=sys.stderr)
    if arr_np.size < 20:
        print(f"  values: {arr_np}", file=sys.stderr)
    else:
        uniq = np.unique(arr_np)
        if uniq.size < 10:
            print(f"  unique values: {uniq}", file=sys.stderr)
    if arr_np.ndim == 3:
        print(f"  channel dim: {arr_np.shape[0]} (first 3 min/max: {[arr_np[c].min() for c in range(min(3, arr_np.shape[0]))]}, {[arr_np[c].max() for c in range(min(3, arr_np.shape[0]))]})", file=sys.stderr)
    print(file=sys.stderr)

def save_side_by_side(input_img, gt_img, pred_img, out_path, titles=['Input', 'GT', 'Pred'], cmap=None):
    """Save input, gt, pred side by side with labels."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axs, [input_img, gt_img, pred_img], titles):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap if cmap else 'gray')
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def plot_modalities_grid(image_dict, bands, out_path, modalities=None):
    """Visualize all present modalities in a grid with stats overlays and NaN mask."""
    import matplotlib.pyplot as plt
    import numpy as np
    modality_order = ['optical', 'alos', 'dem', 's1', 'lc', 'ch']
    modality_titles = {
        'optical': 'Optical (S2)',
        'alos': 'ALOS',
        'dem': 'DEM',
        's1': 'Sentinel-1',
        'lc': 'Land Cover',
        'ch': 'Canopy Height',
    }
    modality_cmaps = {
        'optical': 'gray',
        'alos': 'seismic',
        'dem': 'terrain',
        's1': 'seismic',
        'lc': 'tab20',
        'ch': 'plasma',
    }
    present_modalities = [m for m in modality_order if m in image_dict]
    if modalities:
        present_modalities = [m for m in present_modalities if m in modalities]
    n_mods = len(present_modalities)
    if n_mods == 0:
        return
    max_bands = max([image_dict[m].squeeze().shape[0] if image_dict[m].squeeze().ndim==3 else 1 for m in present_modalities])
    fig, axs = plt.subplots(n_mods, max_bands, figsize=(3*max_bands, 3*n_mods))
    if n_mods == 1:
        axs = np.expand_dims(axs, 0)
    if max_bands == 1:
        axs = np.expand_dims(axs, 1)
    for row, mod in enumerate(present_modalities):
        arr = image_dict[mod].squeeze()
        arr = arr.cpu().numpy() if hasattr(arr, 'cpu') else arr
        if arr.ndim == 2:
            arr = arr[None, ...]
        n_bands = arr.shape[0]
        for col in range(max_bands):
            ax = axs[row, col]
            if col < n_bands:
                # Show band name for optical
                if mod == 'optical' and col < len(bands):
                    band_label = bands[col]
                else:
                    band_label = f'band {col}'
                im = ax.imshow(arr[col], cmap=modality_cmaps.get(mod, 'gray'))
                # Overlay stats
                stats = f"min={np.nanmin(arr[col]):.2f}\nmax={np.nanmax(arr[col]):.2f}\nmean={np.nanmean(arr[col]):.2f}\nstd={np.nanstd(arr[col]):.2f}"
                ax.set_title(f"{modality_titles.get(mod, mod)} {band_label}")
                ax.text(0.99, 0.01, stats, va='bottom', ha='right', fontsize=8, color='black',
                        transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                # NaN/invalid mask visualization
                nan_mask = np.isnan(arr[col])
                if np.any(nan_mask):
                    ax.contour(nan_mask, colors='red', linewidths=0.5)
            else:
                ax.axis('off')
    plt.suptitle(f'All Modalities (normalized)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_advanced_composite(optical, gt, pred, bands, meta, out_path):
    """Plot RGB, GT, error map, and histogram in a composite figure."""
    import matplotlib.pyplot as plt
    import numpy as np
    if optical is not None:
        img = optical.squeeze()
        img = img.cpu().numpy() if hasattr(img, 'cpu') else img
        if img.ndim == 2:
            img = img[None, ...]
        band_names = bands if len(img) == len(bands) else [f'B{i}' for i in range(len(img))]
        if set(['B02', 'B03', 'B04']).issubset(band_names):
            idx_b02 = band_names.index('B02')
            idx_b03 = band_names.index('B03')
            idx_b04 = band_names.index('B04')
            rgb = np.stack([img[idx_b04], img[idx_b03], img[idx_b02]], axis=-1)
            rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6), 0, 1)
        else:
            rgb = np.repeat(img[0][..., None], 3, axis=-1)
    else:
        rgb = None
    gt_np = gt.cpu().numpy() if hasattr(gt, 'cpu') else gt
    pred_np = pred.cpu().numpy() if hasattr(pred, 'cpu') else pred
    error_map = gt_np - pred_np
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    if rgb is not None:
        axs[0].imshow(rgb)
        axs[0].set_title('Input RGB (B04,B03,B02)')
        axs[0].axis('off')
    else:
        axs[0].axis('off')
    im1 = axs[1].imshow(gt_np, cmap='viridis')
    axs[1].set_title('GT')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    im2 = axs[2].imshow(error_map, cmap='bwr')
    axs[2].set_title('Error (GT-Pred)')
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    axs[3].hist(gt_np.flatten(), bins=20, alpha=0.7, label='GT')
    axs[3].hist(pred_np.flatten(), bins=20, alpha=0.7, label='Pred')
    axs[3].set_title('Histogram')
    axs[3].legend()
    plt.suptitle(f"Meta: {meta}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def write_batch_summary(summary_list, output_dir):
    import os
    import numpy as np
    summary_path = os.path.join(output_dir, 'batch_summary.txt')
    # Aggregate stats
    agg = {}
    for mod in ['optical', 'alos', 'dem', 's1', 'lc', 'ch', 'gt', 'pred']:
        vals = [s['modalities'][mod] for s in summary_list if s['modalities'][mod]['present']]
        if vals:
            agg[mod] = {
                'min': float(np.nanmin([v['min'] for v in vals])),
                'max': float(np.nanmax([v['max'] for v in vals])),
                'mean': float(np.nanmean([v['mean'] for v in vals])),
                'std': float(np.nanmean([v['std'] for v in vals])),
                'n_nan': int(np.sum([v['n_nan'] for v in vals])),
                'n_inf': int(np.sum([v['n_inf'] for v in vals])),
                'count': len(vals),
            }
    # Write summary
    with open(summary_path, 'w') as f:
        f.write(f"Batch Summary Report\n====================\n\n")
        for entry in summary_list:
            outlier = False
            for mod, stats in entry['modalities'].items():
                if stats['present'] and (stats['n_nan'] > 0 or stats['n_inf'] > 0 or abs(stats['min']) > 1e4 or abs(stats['max']) > 1e4):
                    outlier = True
            f.write(f"Sample {entry['index']}{' [OUTLIER]' if outlier else ''}\n")
            for mod, stats in entry['modalities'].items():
                f.write(f"  {mod}: ")
                if stats['present']:
                    f.write(f"shape={stats['shape']}, min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, std={stats['std']:.3f}, NaNs={stats['n_nan']}, Infs={stats['n_inf']}\n")
                else:
                    f.write("MISSING\n")
            if entry['warnings']:
                f.write("  WARNINGS:\n")
                for w in entry['warnings']:
                    f.write(f"    - {w}\n")
            f.write("\n")
        f.write(f"Total samples: {len(summary_list)}\n\n")
        f.write("Aggregate stats (across all present samples):\n")
        for mod, stats in agg.items():
            f.write(f"  {mod}: min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}, std={stats['std']:.3f}, total NaNs={stats['n_nan']}, total Infs={stats['n_inf']}, count={stats['count']}\n")
        f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Save AGBD images from tensors/npys.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with saved tensors/npys (input, pred, gt).')
    parser.add_argument('--output_dir', type=str, default='/cluster/scratch/reimannj/agbd_images/', help='Where to save output images.')
    parser.add_argument('--split', type=str, default='val', help='Dataset split (val/test/train).')
    parser.add_argument('--max_images', type=int, default=50, help='Max number of images to save.')
    parser.add_argument('--random', action='store_true', help='Randomly sample images instead of first N.')
    parser.add_argument('--resize', type=int, nargs=2, default=None, help='Resize output images to (W H).')
    parser.add_argument('--cmap', type=str, default='viridis', help='Colormap for gt/pred.')
    parser.add_argument('--overlay', action='store_true', help='Overlay pred on input.')
    parser.add_argument('--rgb_channels', type=int, nargs=3, default=[3,2,1], help='Channels to use for RGB visualization (default: 3 2 1 for Sentinel-2).')
    parser.add_argument('--save_all_channels', action='store_true', help='Save each input channel as a grayscale image for debugging.')
    parser.add_argument('--modalities', type=str, nargs='+', default=None, help='Modalities to plot (default: all present)')
    parser.add_argument('--advanced_composite', action='store_true', help='Save advanced composite plot (RGB, GT, error, hist)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support batch .pt files (dicts with 'input', 'pred', 'gt')
    batch_files = sorted(list(input_dir.glob(f'{args.split}_batch_*.pt')))
    print(f"Found {len(batch_files)} batch files matching '{args.split}_batch_*.pt' in {input_dir}")
    image_count = 0
    indices = list(range(len(batch_files)))
    if args.random:
        indices = random.sample(indices, len(indices))
    if not batch_files:
        print(f"No batch files found. Will try triplet files.")
    processed_any = False
    batch_summary = []
    for batch_idx in indices:
        if image_count >= args.max_images:
            break
        batch = torch.load(batch_files[batch_idx], map_location='cpu')
        input_arr = batch['input']
        pred_arr = batch['pred']
        gt_arr = batch['gt']
        # If input_arr is a dict, treat as multi-modal
        if isinstance(input_arr, dict):
            image_dict = {k: v for k, v in input_arr.items() if isinstance(v, (np.ndarray, torch.Tensor))}
        else:
            image_dict = {'optical': input_arr}
        B = list(image_dict.values())[0].shape[0]
        for i in range(B):
            if image_count >= args.max_images:
                break
            # Per-sample dict for modalities
            sample_modalities = {k: v[i] for k, v in image_dict.items() if v.shape[0] > i}
            # --- Batch summary collection ---
            summary_entry = {'index': image_count, 'modalities': {}, 'warnings': []}
            for mod in ['optical', 'alos', 'dem', 's1', 'lc', 'ch']:
                arr = sample_modalities.get(mod, None)
                if arr is not None:
                    arr_np = arr.cpu().numpy() if hasattr(arr, 'cpu') else arr
                    arr_np = np.squeeze(arr_np)
                    summary_entry['modalities'][mod] = {
                        'present': True,
                        'shape': arr_np.shape,
                        'min': float(np.nanmin(arr_np)),
                        'max': float(np.nanmax(arr_np)),
                        'mean': float(np.nanmean(arr_np)),
                        'std': float(np.nanstd(arr_np)),
                        'n_nan': int(np.isnan(arr_np).sum()),
                        'n_inf': int(np.isinf(arr_np).sum()),
                    }
                    if np.isnan(arr_np).any():
                        summary_entry['warnings'].append(f"{mod}: contains NaNs")
                    if np.isinf(arr_np).any():
                        summary_entry['warnings'].append(f"{mod}: contains Infs")
                else:
                    summary_entry['modalities'][mod] = {'present': False}
            # GT and pred stats
            for name, arr in [('gt', gt_arr[i]), ('pred', pred_arr[i])]:
                arr_np = arr.cpu().numpy() if hasattr(arr, 'cpu') else arr
                arr_np = np.squeeze(arr_np)
                summary_entry['modalities'][name] = {
                    'present': True,
                    'shape': arr_np.shape,
                    'min': float(np.nanmin(arr_np)),
                    'max': float(np.nanmax(arr_np)),
                    'mean': float(np.nanmean(arr_np)),
                    'std': float(np.nanstd(arr_np)),
                    'n_nan': int(np.isnan(arr_np).sum()),
                    'n_inf': int(np.isinf(arr_np).sum()),
                }
                if np.isnan(arr_np).any():
                    summary_entry['warnings'].append(f"{name}: contains NaNs")
                if np.isinf(arr_np).any():
                    summary_entry['warnings'].append(f"{name}: contains Infs")
            batch_summary.append(summary_entry)
            # Save modalities grid
            plot_modalities_grid(sample_modalities, bands=BANDS_PRITHVI, out_path=output_dir / f'{args.split}_modalities_grid_{image_count}.png', modalities=args.modalities)
            # Save advanced composite if requested
            if args.advanced_composite:
                plot_advanced_composite(
                    sample_modalities.get('optical', None),
                    gt_arr[i],
                    pred_arr[i],
                    bands=BANDS_PRITHVI,
                    meta=f'batch {batch_idx} sample {i}',
                    out_path=output_dir / f'{args.split}_composite_{image_count}.png'
                )
            # Use 'optical' modality for rgb if present, else fallback to first available modality
            if 'optical' in sample_modalities:
                rgb = sample_modalities['optical']
            else:
                rgb = next(iter(sample_modalities.values()))
            gt_img = gt_arr[i]
            pred_img = pred_arr[i]
            # --- Handle singleton temporal dimension for AGBD ---
            # If input is [C, 1, H, W], squeeze temporal dim
            if rgb.ndim == 4 and rgb.shape[1] == 1:
                print(f"[INFO] Squeezing singleton temporal dimension for input sample {i} (shape before: {rgb.shape})", file=sys.stderr)
                rgb = rgb.squeeze(1)
            print_tensor_stats(f'input[{i}]', rgb)
            print_tensor_stats(f'gt[{i}]', gt_img)
            print_tensor_stats(f'pred[{i}]', pred_img)
            rgb_np = rgb.cpu().numpy() if isinstance(rgb, torch.Tensor) else rgb
            rgb_np = np.squeeze(rgb_np)
            # Warn if input channels are all 0 or 1
            if np.all(rgb_np == 0) or np.all(rgb_np == 1):
                print(f"[WARN] Input sample {i} is all {int(np.all(rgb_np == 1))}", file=sys.stderr)
            # Save all channels as grayscale if requested
            if args.save_all_channels and rgb_np.ndim == 3:
                for ch in range(rgb_np.shape[0]):
                    ch_img = rgb_np[ch]
                    ch_img_norm = (ch_img - ch_img.min()) / (ch_img.max() - ch_img.min() + 1e-6)
                    ch_img_uint8 = (ch_img_norm * 255).astype(np.uint8)
                    save_image(ch_img_uint8, output_dir / f'{args.split}_input_ch{ch}_{image_count}.png', resize=args.resize)
            # RGB visualization with explicit channel selection
            if rgb_np.ndim == 3 and max(args.rgb_channels) < rgb_np.shape[0]:
                rgb_img = rgb_np[args.rgb_channels, ...]
                if np.all(rgb_img == 0):
                    print(f"[WARN] Selected RGB channels {args.rgb_channels} are all zero for image {image_count}", file=sys.stderr)
                rgb_img = rgb_img.astype(np.float32)
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6)
                rgb_img = (rgb_img * 255)
            elif rgb_np.ndim == 2:
                rgb_img = np.repeat(rgb_np[None, ...], 3, axis=0)
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6)
                rgb_img = (rgb_img * 255)
            else:
                rgb_img = rgb_np
            save_image(rgb_img, output_dir / f'{args.split}_input_rgb_{image_count}.png', resize=args.resize)
            gt_img = gt_img.cpu().numpy() if isinstance(gt_img, torch.Tensor) else gt_img
            pred_img = pred_img.cpu().numpy() if isinstance(pred_img, torch.Tensor) else pred_img
            gt_img = np.squeeze(gt_img).astype(np.float32)
            pred_img = np.squeeze(pred_img).astype(np.float32)
            # Optionally upsample GT/pred to input size for visualization
            if gt_img.shape != rgb_np.shape[-2:]:
                print(f"[INFO] Upsampling GT from {gt_img.shape} to {rgb_np.shape[-2:]} for visualization", file=sys.stderr)
                gt_img = np.array(Image.fromarray(gt_img).resize(rgb_np.shape[-2:][::-1], resample=Image.BILINEAR))
            if pred_img.shape != rgb_np.shape[-2:]:
                print(f"[INFO] Upsampling pred from {pred_img.shape} to {rgb_np.shape[-2:]} for visualization", file=sys.stderr)
                pred_img = np.array(Image.fromarray(pred_img).resize(rgb_np.shape[-2:][::-1], resample=Image.BILINEAR))
            gt_img_norm = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min() + 1e-6)
            pred_img_norm = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-6)
            save_image(gt_img_norm, output_dir / f'{args.split}_gt_{image_count}.png', cmap=args.cmap, resize=args.resize)
            save_image(pred_img_norm, output_dir / f'{args.split}_pred_{image_count}.png', cmap=args.cmap, resize=args.resize)
            if args.overlay:
                overlay = plt.get_cmap(args.cmap)(pred_img_norm)[:, :, :3]
                overlay = (overlay * 255).astype(np.uint8)
                save_image(rgb_img, output_dir / f'{args.split}_input_pred_overlay_{image_count}.png', overlay=overlay, alpha=0.5, resize=args.resize)
            # After saving individual images, also save side-by-side composite
            save_side_by_side(
                rgb_img.transpose(1,2,0) if rgb_img.ndim==3 and rgb_img.shape[0] in [1,3] else rgb_img,
                gt_img_norm,
                pred_img_norm,
                output_dir / f'{args.split}_sidebyside_{image_count}.png',
                titles=['Input', 'GT', 'Pred'],
                cmap=args.cmap
            )
            image_count += 1

    # Find all triplets: input, pred, gt (assume .npy or .pt files)
    input_files = sorted(list(input_dir.glob(f'{args.split}_input_*.*')))
    pred_files = sorted(list(input_dir.glob(f'{args.split}_pred_*.*')))
    gt_files = sorted(list(input_dir.glob(f'{args.split}_gt_*.*')))
    print(f"Found {len(input_files)} input, {len(pred_files)} pred, {len(gt_files)} gt triplet files in {input_dir}")
    n = min(len(input_files), len(pred_files), len(gt_files), args.max_images)
    indices = list(range(n))
    if args.random:
        indices = random.sample(indices, n)
    if n == 0 and not processed_any:
        print(f"No batch or triplet files found for split '{args.split}' in {input_dir}. Nothing to process.")
    for i in indices:
        # Load arrays
        def load_arr(f):
            if f.suffix == '.npy':
                return np.load(f)
            elif f.suffix == '.pt':
                return torch.load(f, map_location='cpu').numpy()
            else:
                raise ValueError(f'Unknown file type: {f}')
        input_arr = load_arr(input_files[i])
        pred_arr = load_arr(pred_files[i])
        gt_arr = load_arr(gt_files[i])
        print_tensor_stats('input', input_arr)
        print_tensor_stats('gt', gt_arr)
        print_tensor_stats('pred', pred_arr)
        rgb_np = input_arr.cpu().numpy() if isinstance(input_arr, torch.Tensor) else input_arr
        rgb_np = np.squeeze(rgb_np)
        if args.save_all_channels and rgb_np.ndim == 3:
            for ch in range(rgb_np.shape[0]):
                ch_img = rgb_np[ch]
                ch_img_norm = (ch_img - ch_img.min()) / (ch_img.max() - ch_img.min() + 1e-6)
                ch_img_uint8 = (ch_img_norm * 255).astype(np.uint8)
                save_image(ch_img_uint8, output_dir / f'{args.split}_input_ch{ch}_{i}.png', resize=args.resize)
        if rgb_np.ndim == 3 and max(args.rgb_channels) < rgb_np.shape[0]:
            rgb_img = rgb_np[args.rgb_channels, ...]
            if np.all(rgb_img == 0):
                print(f"[WARN] Selected RGB channels {args.rgb_channels} are all zero for image {i}", file=sys.stderr)
            rgb_img = rgb_img.astype(np.float32)
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6)
            rgb_img = (rgb_img * 255)
        elif rgb_np.ndim == 2:
            rgb_img = np.repeat(rgb_np[None, ...], 3, axis=0)
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6)
            rgb_img = (rgb_img * 255)
        else:
            rgb_img = rgb_np
        save_image(rgb_img, output_dir / f'{args.split}_input_rgb_{i}.png', resize=args.resize)
        gt_img = gt_arr
        pred_img = pred_arr
        gt_img = gt_img.cpu().numpy() if isinstance(gt_img, torch.Tensor) else gt_img
        pred_img = pred_img.cpu().numpy() if isinstance(pred_img, torch.Tensor) else pred_img
        gt_img = np.squeeze(gt_img).astype(np.float32)
        pred_img = np.squeeze(pred_img).astype(np.float32)
        gt_img = (gt_img - gt_img.min()) / (gt_img.max() - gt_img.min() + 1e-6)
        pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-6)
        save_image(gt_img, output_dir / f'{args.split}_gt_{i}.png', cmap=args.cmap, resize=args.resize)
        save_image(pred_img, output_dir / f'{args.split}_pred_{i}.png', cmap=args.cmap, resize=args.resize)
        if args.overlay:
            overlay = plt.get_cmap(args.cmap)(pred_img)[:, :, :3]
            overlay = (overlay * 255).astype(np.uint8)
            save_image(rgb_img, output_dir / f'{args.split}_input_pred_overlay_{i}.png', overlay=overlay, alpha=0.5, resize=args.resize)
        # After saving individual images, also save side-by-side composite
        save_side_by_side(
            rgb_img.transpose(1,2,0) if rgb_img.ndim==3 and rgb_img.shape[0] in [1,3] else rgb_img,
            gt_img,
            pred_img,
            output_dir / f'{args.split}_sidebyside_{i}.png',
            titles=['Input', 'GT', 'Pred'],
            cmap=args.cmap
        )
    if processed_any:
        print("Done processing images.")
    write_batch_summary(batch_summary, output_dir)

if __name__ == "__main__":
    main()
