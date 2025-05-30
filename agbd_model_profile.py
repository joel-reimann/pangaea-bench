import time
import torch
from torch.utils.data import DataLoader
from pangaea.datasets.agbd import AGBD

# Config from agbd.yaml (as in your test script)
config = {
    'split': 'train',
    'dataset_name': 'AGBD',
    'root_path': '/scratch/reimannj/pangaea_agbd_integration_final/data/agbd',
    'hdf5_dir': '/scratch/reimannj/pangaea_agbd_integration_final/data/agbd',
    'hdf5_pattern': 'data_subset-*-v4_*-20.h5',
    'split_files': '/scratch/reimannj/pangaea_agbd_integration_final/data/agbd',
    'img_size': 25,
    'multi_modal': True,
    'multi_temporal': False,
    'bands': {
        'optical': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
        'sar': ['VV', 'VH'],
        'auxiliary': ['DEM', 'CH_ch', 'CH_std', 'LC_1', 'LC_2', 'S2_SCL']
    },
    'data_mean': {
        'optical': [0.12478869, 0.13480005, 0.16031432, 0.1532097, 0.20312776, 0.32636437, 0.36605212, 0.3811653, 0.3910436, 0.3910644, 0.2917373, 0.21169408],
        'sar': [-10.381429, -16.722847],
        'auxiliary': [604.6373, 9.736144, 7.9882116, 85.912, 85.912, 4.0]
    },
    'data_std': {
        'optical': [0.024433358, 0.02822557, 0.032037303, 0.038628064, 0.04205057, 0.07139242, 0.08555025, 0.092815965, 0.0896364, 0.0836445, 0.07472579, 0.05880649],
        'sar': [8.561741, 8.718428],
        'auxiliary': [588.0209, 9.493601, 4.5494938, 39.0094, 39.0094, 1.0]
    },
    'data_min': {
        'optical': [0.0001, 0.0001, 0.0001, 0.0001, 0.0422, 0.0502, 0.0616, 0.0001, 0.055, 0.0012, 0.0953, 0.0975],
        'sar': [-83.0, -83.0],
        'auxiliary': [-82.0, 0.0, 0.0, 40.0, 40.0, 4.0]
    },
    'data_max': {
        'optical': [1.8808, 2.1776, 2.12, 2.0032, 1.7502, 1.7245, 1.7149, 1.7488, 1.688, 1.7915, 1.648, 1.6775],
        'sar': [13.329468, 11.688309],
        'auxiliary': [5205.0, 61.0, 254.0, 126.0, 126.0, 4.0]
    },
    'label_group': 'GEDI',
    'label_name': 'agbd',
    'ignore_index': -1
}

print('Instantiating AGBD dataset...')
dataset = AGBD(**config)
print(f'Dataset length: {len(dataset)}')

# Profile model forward/backward pass
try:
    import torch.nn as nn
    import torch.optim as optim
    # Dummy model: single conv layer to simulate encoder/decoder
    class DummyModel(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        def forward(self, x):
            return self.conv(x)

    # Use only optical bands for simplicity
    in_ch = len(config['bands']['optical'])
    out_ch = 1
    model = DummyModel(in_ch, out_ch)
    if torch.cuda.is_available():
        model = model.cuda()
    loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True, persistent_workers=True, shuffle=True)
    optimizer = optim.Adam(model.parameters())
    print('\nProfiling model forward/backward (10 batches)...')
    fw_times, bw_times = [], []
    for i, batch in enumerate(loader):
        x = batch['image']['optical'].squeeze(2)  # [B, C, H, W]
        y = batch['target']
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
        t0 = time.time()
        out = model(x)
        loss = (out.squeeze(1) - y).pow(2).mean()
        t1 = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.time()
        fw_times.append(t1-t0)
        bw_times.append(t2-t1)
        if i >= 9:
            break
    print(f'  Avg forward time: {sum(fw_times)/len(fw_times):.4f} s')
    print(f'  Avg backward time: {sum(bw_times)/len(bw_times):.4f} s')
except Exception as e:
    print(f"[WARN] Model profiling failed: {e}")

print('Done.')
