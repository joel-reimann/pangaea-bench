import torch
import sys
import os

# Set this to your tensor directory
tensor_dir = "/scratch/reimannj3/agbd_tensors/debug_32733932"

# List all .pt files
files = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]
files.sort()

def print_stats(arr, name):
    arr = arr.cpu().numpy() if hasattr(arr, 'cpu') else arr
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}, mean={arr.mean()}, std={arr.std()}")

for fname in files[:5]:  # Inspect first 5 files
    print(f"\n--- {fname} ---")
    d = torch.load(os.path.join(tensor_dir, fname))
    for k in ['input', 'pred', 'gt', 'logits']:
        if k in d:
            if isinstance(d[k], dict):  # input is a dict of modalities
                for mod, arr in d[k].items():
                    print_stats(arr, f"input[{mod}]")
            elif d[k] is not None:
                print_stats(d[k], k)
    if 'meta' in d:
        print("meta:", d['meta'])