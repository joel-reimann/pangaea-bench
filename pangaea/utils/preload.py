import torch

class PreloadedDataset(torch.utils.data.Dataset):
    """
    Wraps a dataset and preloads all samples into RAM for fast access.
    """
    def __init__(self, base_dataset, max_samples=None):
        if max_samples is None:
            max_samples = len(base_dataset)
        self.samples = [base_dataset[i] for i in range(min(max_samples, len(base_dataset)))]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]
