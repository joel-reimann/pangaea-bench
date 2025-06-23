"""
Test script to compare validation speed with and without RAM preloading for a subset of the AGBD validation set.
"""
import time
import torch
from torch.utils.data import DataLoader, Subset
import os
from omegaconf import OmegaConf

# Import your dataset class here:
from pangaea.datasets.agbd import AGBD  # Adjust if your dataset class is named differently

# ---- CONFIG ----
VAL_SUBSET_SIZE = 2000  # Number of samples to test (adjust as needed)
BATCH_SIZE = 64
NUM_WORKERS = 0  # Use 0 for RAM preloading to avoid memory duplication

# ---- RAM Preloading Wrapper ----
class PreloadedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, max_samples=None):
        if max_samples is None:
            max_samples = len(base_dataset)
        self.samples = [base_dataset[i] for i in range(min(max_samples, len(base_dataset)))]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    # Load config and instantiate your validation dataset
    config_path = os.path.join(os.path.dirname(__file__), "configs/dataset/agbd.yaml")
    dataset_config = OmegaConf.load(config_path)
    val_dataset = AGBD(split="val", **dataset_config)
    subset_indices = list(range(min(VAL_SUBSET_SIZE, len(val_dataset))))
    val_subset = Subset(val_dataset, subset_indices)

    # --- Test 1: Standard (no preloading) ---
    print("\n[Standard DataLoader: no RAM preloading]")
    loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    t0 = time.time()
    for batch in loader:
        pass  # Optionally, run your model here
    print(f"Validation (no preloading) took {time.time() - t0:.2f} seconds for {len(val_subset)} samples.")

    # --- Test 2: RAM Preloading ---
    print("\n[RAM Preloaded DataLoader]")
    preloaded = PreloadedDataset(val_subset)
    loader_pre = DataLoader(preloaded, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    t0 = time.time()
    for batch in loader_pre:
        pass
    print(f"Validation (RAM preloading) took {time.time() - t0:.2f} seconds for {len(preloaded)} samples.")

    print("\nCompare the two timings above to see if RAM preloading helps on your system.")
