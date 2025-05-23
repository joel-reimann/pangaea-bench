import hashlib
import os as os
import pathlib
import pprint
import time

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pangaea.datasets.base import GeoFMDataset, GeoFMSubset, RawGeoFMDataset
from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder
from pangaea.engine.evaluator import Evaluator
from pangaea.engine.trainer import Trainer
from pangaea.utils.collate_fn import get_collate_fn
from pangaea.utils.logger import init_logger
from pangaea.utils.subset_sampler import get_subset_indices
from pangaea.utils.utils import (
    fix_seed,
    get_best_model_ckpt_path,
    get_final_model_ckpt_path,
    get_generator,
    seed_worker,
)


def get_exp_info(hydra_config: HydraConf) -> dict[str, str]:
    """Create a unique experiment name based on the choices made in the config.

    Args:
        hydra_config (HydraConf): hydra config.

    Returns:
        str: experiment information.
    """
    choices = OmegaConf.to_container(hydra_config.runtime.choices)
    cfg_hash = hashlib.sha1(
        OmegaConf.to_yaml(hydra_config).encode(), usedforsecurity=False
    ).hexdigest()[:6]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    fm = choices["encoder"]
    decoder = choices["decoder"]
    ds = choices["dataset"]
    task = choices["task"]
    exp_info = {
        "timestamp": timestamp,
        "fm": fm,
        "decoder": decoder,
        "ds": ds,
        "task": task,
        "exp_name": f"{timestamp}_{cfg_hash}_{fm}_{decoder}_{ds}",
    }
    return exp_info


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Geofm-bench main function.

    Args:
        cfg (DictConfig): main_config
    """
    # fix all random seeds
    fix_seed(cfg.seed)
    # distributed training variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl")

    # true if training else false
    train_run = cfg.train
    if train_run:
        exp_info = get_exp_info(HydraConfig.get())
        exp_name = exp_info["exp_name"]
        task_name = exp_info["task"]
        exp_dir = pathlib.Path(cfg.work_dir) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger_path = exp_dir / "train.log"
        config_log_dir = exp_dir / "configs"
        config_log_dir.mkdir(exist_ok=True)
        # init wandb
        if cfg.task.trainer.use_wandb and rank == 0:
            import wandb

            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project="geofm-bench",
                name=exp_name,
                config=wandb_cfg,
            )
            cfg["wandb_run_id"] = wandb.run.id
        OmegaConf.save(cfg, config_log_dir / "config.yaml")

    else:
        exp_dir = pathlib.Path(cfg.ckpt_dir)
        exp_name = exp_dir.name
        logger_path = exp_dir / "test.log"
        # load training config
        cfg_path = exp_dir / "configs" / "config.yaml"
        cfg = OmegaConf.load(cfg_path)
        if cfg.task.trainer.use_wandb and rank == 0:
            import wandb

            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project="geofm-bench",
                name=exp_name,
                config=wandb_cfg,
                id=cfg.get("wandb_run_id"),
                resume="allow",
            )

    logger = init_logger(logger_path, rank=rank)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    # FORENSIC: Log dataset config and instantiation
    forensic_path = "/cluster/scratch/reimannj/agbd_logs/forensic_run_dataset.txt"
    # Ensure log directory exists
    os.makedirs(os.path.dirname(forensic_path), exist_ok=True)
    with open(forensic_path, "a") as f:
        f.write(f"[RUN] Instantiating encoder, decoder, and datasets at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"[RUN] Dataset config: {OmegaConf.to_yaml(cfg.dataset)}\n")

    encoder: Encoder = instantiate(cfg.encoder)
    encoder.load_encoder_weights(logger)
    logger.info("Built {}.".format(encoder.model_name))

    # prepare the decoder (segmentation/regression)
    decoder: Decoder = instantiate(
        cfg.decoder,
        encoder=encoder,
    )
    decoder.to(device)
    decoder = torch.nn.parallel.DistributedDataParallel(
        decoder,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=cfg.finetune,
    )
    logger.info(
        "Built {} for with {} encoder.".format(
            decoder.module.model_name, type(encoder).__name__
        )
    )

    modalities = list(encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities)

    # training
    if train_run:
        # get preprocessor
        train_preprocessor = instantiate(
            cfg.preprocessing.train,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )
        val_preprocessor = instantiate(
            cfg.preprocessing.val,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )

        # get datasets
        raw_train_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="train")
        raw_val_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="val")
        with open(forensic_path, "a") as f:
            f.write(f"[RUN] Instantiated raw_train_dataset: {type(raw_train_dataset)}\n")
            f.write(f"[RUN] Instantiated raw_val_dataset: {type(raw_val_dataset)}\n")
        print(f"[FORENSIC] Instantiated raw_train_dataset: {type(raw_train_dataset)}", flush=True)
        print(f"[FORENSIC] Instantiated raw_val_dataset: {type(raw_val_dataset)}", flush=True)

        # --- FORENSIC LOGGING: Inspect raw_train_dataset and raw_val_dataset output before transforms ---
        for split_name, dataset in [("train", raw_train_dataset), ("val", raw_val_dataset)]:
            try:
                sample = dataset[0]
                image = sample['image']
                meta = sample.get('meta', sample.get('metadata', {}))
                logger.info(f"[FORENSIC] First sample from {split_name} dataset:")
                logger.info(f"  meta: {meta}")
                for key, tensor in image.items():
                    logger.info(f"  modality: {key}, shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}, min: {tensor.min().item()}, max: {tensor.max().item()}, mean: {tensor.float().mean().item()}")
                target = sample.get('target', None)
                if target is not None:
                    logger.info(f"  target: shape: {tuple(target.shape)}, dtype: {target.dtype}, min: {target.min().item()}, max: {target.max().item()}, mean: {target.float().mean().item()}")
                # Save RGB image for optical if possible
                if 'optical' in image:
                    import matplotlib.pyplot as plt
                    import numpy as np
                    arr = image['optical'].squeeze().cpu().numpy()
                    if arr.shape[0] >= 3:
                        rgb = np.stack([
                            arr[2],  # B04 (red)
                            arr[1],  # B03 (green)
                            arr[0],  # B02 (blue)
                        ], axis=-1)
                        rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6), 0, 1)
                        out_path = f"/cluster/scratch/reimannj/forensic_{split_name}_sample0_rgb.png"
                        plt.imsave(out_path, rgb)
                        logger.info(f"[FORENSIC] Saved {split_name} RGB image to {out_path}")
            except Exception as e:
                logger.error(f"[FORENSIC] Error logging first sample from {split_name} dataset: {e}")

        if 0 < cfg.limited_label_train < 1:
            indices = get_subset_indices(
                raw_train_dataset,
                task=task_name,
                strategy=cfg.limited_label_strategy,
                label_fraction=cfg.limited_label_train,
                num_bins=cfg.stratification_bins,
                logger=logger,
            )
            raw_train_dataset = GeoFMSubset(raw_train_dataset, indices)

        if 0 < cfg.limited_label_val < 1:
            indices = get_subset_indices(
                raw_val_dataset,
                task=task_name,
                strategy=cfg.limited_label_strategy,
                label_fraction=cfg.limited_label_val,
                num_bins=cfg.stratification_bins,
                logger=logger,
            )
            raw_val_dataset = GeoFMSubset(raw_val_dataset, indices)

        train_dataset = GeoFMDataset(
            raw_train_dataset, train_preprocessor, cfg.data_replicate
        )
        val_dataset = GeoFMDataset(
            raw_val_dataset, val_preprocessor, cfg.data_replicate
        )
        with open(forensic_path, "a") as f:
            f.write(f"[RUN] Instantiated train_dataset: {type(train_dataset)}\n")
            f.write(f"[RUN] Instantiated val_dataset: {type(val_dataset)}\n")
        print(f"[FORENSIC] Instantiated train_dataset: {type(train_dataset)}", flush=True)
        print(f"[FORENSIC] Instantiated val_dataset: {type(val_dataset)}", flush=True)

        logger.info("Built {} dataset.".format(cfg.dataset.dataset_name))

        logger.info(
            f"Total number of train patches: {len(train_dataset)}\n"
            f"Total number of validation patches: {len(val_dataset)}\n"
        )

        # get train val data loaders
        train_loader = DataLoader(
            train_dataset,
            sampler=DistributedSampler(train_dataset),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            # persistent_workers=True causes memory leak
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=get_generator(cfg.seed),
            drop_last=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            sampler=DistributedSampler(val_dataset),
            batch_size=cfg.test_batch_size,
            num_workers=cfg.test_num_workers,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            # generator=g,
            drop_last=False,
            collate_fn=collate_fn,
        )
        with open(forensic_path, "a") as f:
            f.write(f"[RUN] Created train_loader and val_loader at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"[FORENSIC] Created train_loader and val_loader", flush=True)

        criterion = instantiate(cfg.criterion)
        optimizer = instantiate(cfg.optimizer, params=decoder.parameters())
        lr_scheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            total_iters=len(train_loader) * cfg.task.trainer.n_epochs,
        )

        val_evaluator: Evaluator = instantiate(
            cfg.task.evaluator, val_loader=val_loader, exp_dir=exp_dir, device=device
        )
        trainer: Trainer = instantiate(
            cfg.task.trainer,
            model=decoder,
            train_loader=train_loader,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            criterion=criterion,
            evaluator=val_evaluator,
            exp_dir=exp_dir,
            device=device,
        )
        # resume training if model_checkpoint is provided
        if cfg.ckpt_dir is not None:
            trainer.load_model(cfg.ckpt_dir)

        trainer.train()

    # Evaluation
    test_preprocessor = instantiate(
        cfg.preprocessing.test,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )

    # get datasets
    raw_test_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
    # Subset test set for fast debugging if limited_label_test is set
    if hasattr(cfg, "limited_label_test") and 0 < cfg.limited_label_test < 1:
        indices = get_subset_indices(
            raw_test_dataset,
            task=task_name if 'task_name' in locals() else None,
            strategy=cfg.limited_label_strategy,
            label_fraction=cfg.limited_label_test,
            num_bins=cfg.stratification_bins,
            logger=logger,
        )
        raw_test_dataset = GeoFMSubset(raw_test_dataset, indices)
    test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor)

    test_loader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(test_dataset),
        batch_size=cfg.test_batch_size,
        num_workers=cfg.test_num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_evaluator: Evaluator = instantiate(
        cfg.task.evaluator, val_loader=test_loader, exp_dir=exp_dir, device=device
    )

    if cfg.use_final_ckpt:
        model_ckpt_path = get_final_model_ckpt_path(exp_dir)
    else:
        model_ckpt_path = get_best_model_ckpt_path(exp_dir)
    test_evaluator.evaluate(decoder, "test_model", model_ckpt_path)

    if cfg.use_wandb and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
