import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pangaea.datasets.agbd import AGBDDataset, PRITHVI_BANDS
import argparse

# --- CONFIG ---
# Set this to your local copy of the AGBD HDF5 files
# ROOT_PATH = '/scratch/reimannj/pangaea_agbd_integration_final/data/agbd/'  # <-- your local path
ROOT_PATH = '/cluster/work/igp_psr/gsialelli/Data/patches/'
SPLIT = 'val'  # or 'train' or 'test'
IMG_SIZE = 25  # AGBD native patch size

# --- BAND SETS FROM AGBD OFFICIAL CODE ---
# Prithvi bands (used in your current pipeline)
BANDS_PRITHVI = PRITHVI_BANDS  # ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
# RGBN (used in some AGBD baselines)
BANDS_RGBN = ['B02', 'B03', 'B04', 'B08']
# All S2 bands (official AGBD order, check your HDF5 for exact order)
BANDS_ALL = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

# --- SELECT WHICH BANDS TO TEST ---
BANDS = BANDS_PRITHVI  # Change to BANDS_RGBN or BANDS_ALL as needed

# --- OUTPUT DIR SETUP ---
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
base_out_dir = f'/scratch/reimannj3/agbd_debug/{run_id}_{SPLIT}_' + '_'.join(BANDS)
os.makedirs(base_out_dir, exist_ok=True)
img_dir = os.path.join(base_out_dir, 'images')
os.makedirs(img_dir, exist_ok=True)
log_path = os.path.join(base_out_dir, 'debug_log.txt')

# --- LOGGING ---
import sys
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
logfile = open(log_path, 'w')
sys.stdout = Tee(sys.stdout, logfile)
sys.stderr = Tee(sys.stderr, logfile)

print(f'AGBD Debug Run {run_id}')
print(f'ROOT_PATH: {ROOT_PATH}')
print(f'SPLIT: {SPLIT}')
print(f'BANDS: {BANDS}')
print(f'Output dir: {base_out_dir}')

# Print normalization stats for each band
import pickle
# norm_stats_path = os.path.join(ROOT_PATH, 'statistics_subset_2019-2020-v4_new.pkl')
norm_stats_path = os.path.join('/cluster/home/reimannj/', 'statistics_subset_2019-2020-v4_new.pkl')
if os.path.exists(norm_stats_path):
    with open(norm_stats_path, 'rb') as f:
        norm_stats = pickle.load(f)
    print('\n--- Normalization stats for each band ---')
    for band in BANDS:
        if band in norm_stats['S2_bands']:
            print(f"Band {band}: {norm_stats['S2_bands'][band]}")
        else:
            print(f"Band {band}: NOT FOUND in stats file!")
else:
    print(f"Normalization stats file not found: {norm_stats_path}")

# --- DATASET LOAD ---
print('Instantiating AGBDDataset...')
dataset = AGBDDataset(
    root_path=ROOT_PATH,
    split=SPLIT,
    bands=BANDS,
    img_size=IMG_SIZE,
    dataset_name='agbd',
    multi_modal=False,
    multi_temporal=False,
    classes=['AGBD'],
    num_classes=1,
    ignore_index=-1,
    distribution='regression',
    data_mean=0,
    data_std=1,
    data_min=0,
    data_max=1,
    download_url=None,
    auto_download=False,
    debug=True,
)
print(f"Dataset length: {len(dataset)}")

# --- CLI ARGUMENTS ---
parser = argparse.ArgumentParser(description="AGBD Debug Sample Visualizer")
parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
parser.add_argument('--output_dir', type=str, default=None, help='Override output directory')
parser.add_argument('--modalities', type=str, nargs='+', default=None, help='Modalities to plot (default: all present)')
args, _ = parser.parse_known_args()

num_samples = args.num_samples
if args.output_dir:
    base_out_dir = args.output_dir
    img_dir = os.path.join(base_out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    log_path = os.path.join(base_out_dir, 'debug_log.txt')
    logfile = open(log_path, 'w')
    sys.stdout = Tee(sys.stdout, logfile)
    sys.stderr = Tee(sys.stderr, logfile)

# --- SAMPLE LOOP ---
for idx in range(min(num_samples, len(dataset))):
    print(f'\n===== Sample {idx} =====')
    sample = dataset[idx]
    meta = sample['meta']
    print(f"Meta: {meta}")
    image = sample['image']
    target = sample['target']
    print('--- MODALITIES PRESENT ---')
    for key in image:
        print(f"  {key}: shape {tuple(image[key].shape)}, dtype {image[key].dtype}")
    print('--- RAW GT STATS ---')
    print(f"Shape: {target.shape}, dtype: {target.dtype}")
    print(f"Min: {target.min().item()}, Max: {target.max().item()}, Mean: {target.mean().item()}")
    print(f"Unique values: {torch.unique(target)}")

    # --- Visualize all modalities in a grid ---
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
    present_modalities = [m for m in modality_order if m in image]
    if args.modalities:
        present_modalities = [m for m in present_modalities if m in args.modalities]
    n_mods = len(present_modalities)
    max_bands = max([image[m].squeeze().shape[0] if image[m].squeeze().ndim==3 else 1 for m in present_modalities])
    fig, axs = plt.subplots(n_mods, max_bands, figsize=(3*max_bands, 3*n_mods))
    if n_mods == 1:
        axs = np.expand_dims(axs, 0)
    if max_bands == 1:
        axs = np.expand_dims(axs, 1)
    for row, mod in enumerate(present_modalities):
        arr = image[mod].squeeze().cpu().numpy()
        if arr.ndim == 2:
            arr = arr[None, ...]
        n_bands = arr.shape[0]
        for col in range(max_bands):
            ax = axs[row, col]
            if col < n_bands:
                # Show band name for optical
                if mod == 'optical' and col < len(BANDS):
                    band_label = BANDS[col]
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
    plt.suptitle(f'Sample {idx} - All Modalities (normalized)')
    plt.tight_layout()
    out_path = os.path.join(img_dir, f'sample{idx}_modalities_grid.png')
    plt.savefig(out_path)
    plt.close()

    # --- RGB composite for S2/optical if possible, with GT, error map, histogram ---
    if 'optical' in image:
        img = image['optical'].squeeze().cpu().numpy()
        if img.ndim == 2:
            img = img[None, ...]
        band_names = BANDS if len(img) == len(BANDS) else [f'B{i}' for i in range(len(img))]
        if set(['B02', 'B03', 'B04']).issubset(band_names):
            idx_b02 = band_names.index('B02')
            idx_b03 = band_names.index('B03')
            idx_b04 = band_names.index('B04')
            rgb = np.stack([img[idx_b04], img[idx_b03], img[idx_b02]], axis=-1)
            rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6), 0, 1)
        else:
            rgb = np.repeat(img[0][..., None], 3, axis=-1)
        gt_np = target.cpu().numpy()
        # Placeholder for prediction (N/A)
        pred_np = np.zeros_like(gt_np)
        # Error map (GT - Pred)
        error_map = gt_np - pred_np
        # --- Plot composite ---
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].imshow(rgb)
        axs[0].set_title('Input RGB (B04,B03,B02)')
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
        plt.suptitle(f"Sample {idx} meta: {meta}")
        plt.tight_layout()
        out_path = os.path.join(img_dir, f'sample{idx}_rgb_gt_error_hist.png')
        plt.savefig(out_path)
        plt.close()

    # --- Print metadata fields ---
    print('--- METADATA FIELDS ---')
    for k, v in meta.items():
        print(f"  {k}: {v}")

print(f"\nAll images and logs saved to {base_out_dir}")
logfile.close()
