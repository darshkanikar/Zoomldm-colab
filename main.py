# main_colab.py  (Colab-compatible version)

import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset, default_collate
from functools import partial
from PIL import Image
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_info
# ❌ REMOVE THIS:
# from pytorch_lightning.plugins import DDPPlugin
import wandb

from ldm.util import instantiate_from_config


# === 🔹 COLAB FIXES ===
# Disable wandb interactive popups
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_MODE"] = "offline"
# Avoid permission issues
os.makedirs("logs", exist_ok=True)
# ========================


# =============================
# (keep your helper functions)
# =============================

# ... keep get_parser(), nondefault_trainer_args(), WrappedDataset, DataModuleFromConfig,
# SetupCallback, ImageLogger, CUDACallback as in your original code ...


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%m-%dT%H-%M")
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    # === Colab GPU handling ===
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print("⚠️ No GPU detected — training will run on CPU (slow).")
    else:
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")

    # Handle resume, logdir, etc.
    if opt.name and opt.resume:
        raise ValueError("-n/--name and -r/--resume cannot be used together.")
    if opt.resume:
        if os.path.isfile(opt.resume):
            logdir = os.path.dirname(os.path.dirname(opt.resume))
            ckpt = opt.resume
        else:
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        nowname = os.path.basename(logdir)
    else:
        name = f"_{opt.name}" if opt.name else ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # === Load config ===
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())

        # === Trainer config ===
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # ⚠️ Colab: only 1 GPU, so no DDP
        trainer_config["accelerator"] = "gpu" if use_gpu else "cpu"
        trainer_config["devices"] = 1
        trainer_config["precision"] = 16 if use_gpu else 32
        trainer_config["max_epochs"] = trainer_config.get("max_epochs", 1)  # default 1 epoch for testing

        # merge CLI args
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # === Model ===
        model = instantiate_from_config(config.model)

        # === WandB logger (offline in Colab) ===
        logger_cfg = {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "project": "colab_run",
                "name": opt.wandb_name or nowname,
                "save_dir": logdir,
                "offline": True,  # Colab default
                "id": opt.wandb_id or nowname,
                "resume": None,
                "config": vars(opt),
            },
        }
        trainer_logger = instantiate_from_config(logger_cfg)

        # === Checkpoints ===
        ckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "save_last": True,
                "verbose": True,
            },
        }

        callbacks = [
            instantiate_from_config(ckpt_cfg),
            LearningRateMonitor(logging_interval="step"),
            CUDACallback(),
        ]

        trainer = Trainer.from_argparse_args(
            trainer_opt,
            logger=trainer_logger,
            callbacks=callbacks,
        )

        # === Data ===
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}: {len(data.datasets[k])} samples")

        # === Learning Rate ===
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        model.learning_rate = base_lr
        print(f"Learning rate set to {model.learning_rate:.2e}")

        # === Run training/testing ===
        if opt.train:
            trainer.fit(model, data)
        if not opt.no_test:
            trainer.test(model, data)

    except Exception as e:
        print("❌ Training failed due to:", e)
        raise
