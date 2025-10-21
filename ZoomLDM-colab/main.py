# main.py (Colab-ready, synced with zoomldm_brca.yaml)
import argparse
import os
import sys
import datetime
import glob
import time
import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import WandbLogger
from ldm.util import instantiate_from_config

# --- COLAB / ENV small fixes ---
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_MODE"] = "offline"
os.makedirs("logs", exist_ok=True)

# ---- Helper stubs / compatibility ----
def nondefault_trainer_args(opt):
    """
    Return list of trainer-related args present in opt that are not defaults.
    For now we return empty list to avoid overriding trainer_config unexpectedly.
    """
    return []

class CUDACallback(Callback):
    """Minimal callback stub to mimic original repo callback if missing."""
    def on_fit_start(self, trainer, pl_module):
        # no-op; placeholder for any CUDA-specific logging in original repo
        return

# ---- Argument parser ----
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true", help="Run training")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (Colab uses 0)")
    parser.add_argument("--base", type=str, nargs="+", required=True,
                        help="One or more YAML config files (space-separated)")
    # Common options original repo expects — provide defaults so code won't fail
    parser.add_argument("--name", type=str, default="zoomldm_run", help="Experiment name")
    parser.add_argument("--resume", type=str, default="", help="Resume from logdir or checkpoint")
    parser.add_argument("--postfix", type=str, default="", help="Optional name postfix")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory to save logs/checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_name", type=str, default="", help="WandB name (optional)")
    parser.add_argument("--wandb_id", type=str, default="", help="WandB id (optional)")
    parser.add_argument("--no_test", action="store_true", help="Skip test phase")
    # allow passing extra overrides: --key.subkey value
    return parser

# ---- Main execution ----
if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%m-%dT%H-%M")
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # GPU handling (Colab typically has one GPU on cuda:0)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {device_name}")
    else:
        print("⚠️ No GPU detected — running on CPU (very slow for training)")

    # Handle resume case or fresh run
    if opt.name and opt.resume:
        raise ValueError("-n/--name and --resume cannot be used together.")

    if opt.resume:
        # If resume is a file, treat as checkpoint; else treat as logdir
        if os.path.isfile(opt.resume):
            ckpt = opt.resume
            logdir = os.path.dirname(os.path.dirname(ckpt))
        else:
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        # try to pick up base configs inside the resumed logdir if available
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        if base_configs:
            # extend list of base configs to include those found
            opt.base = base_configs + opt.base
        nowname = os.path.basename(logdir)
    else:
        name = f"_{opt.name}" if opt.name else ""
        nowname = now + name + (opt.postfix or "")
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)

    seed_everything(opt.seed, workers=True)

    try:
        # --- Load and merge configs ---
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)

        # Pull out lightning config if present in YAML
        lightning_config = config.pop("lightning", OmegaConf.create())

        # --- Prepare trainer config for Colab single GPU ---
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        trainer_config["accelerator"] = "gpu" if use_gpu else "cpu"
        trainer_config["devices"] = 1
        trainer_config["precision"] = 16 if use_gpu else 32
        # set a sane default for quick testing; override in YAML to train longer
        trainer_config["max_epochs"] = int(trainer_config.get("max_epochs", 1))

        # Merge any explicit trainer overrides from CLI (we return empty for now)
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)

        # Instantiate model
        model = instantiate_from_config(config.model)
        print("Model instantiated.")

        # Setup WandB logger (offline in Colab)
        wandb_name_final = opt.wandb_name or nowname
        wandb_id_final = opt.wandb_id or None
        logger_cfg = {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "project": "colab_run",
                "name": wandb_name_final,
                "save_dir": logdir,
                "offline": True,
                "id": wandb_id_final,
                "resume": None,
                "config": {},
            },
        }
        try:
            trainer_logger = instantiate_from_config(logger_cfg)
        except Exception as e:
            # fallback: plain WandbLogger instantiation if instantiate_from_config fails
            trainer_logger = WandbLogger(project="colab_run", name=wandb_name_final, save_dir=logdir, offline=True)

        # Checkpoint callback
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

        # Create PyTorch Lightning Trainer using kwargs only (no add_argparse_args)
        # Convert OmegaConf map to normal dict for Trainer(**)
        trainer_kwargs = dict(trainer_config)
        # attach callbacks and logger
        trainer_kwargs["callbacks"] = callbacks
        trainer_kwargs["logger"] = trainer_logger

        # Ensure deterministic and track grads if needed (optional)
        trainer = Trainer(**trainer_kwargs)
        print("Trainer created with config:", trainer_kwargs)

        # --- Data (only if provided in config) ---
        data = None
        if "data" in config and config.data is not None:
            try:
                data = instantiate_from_config(config.data)
                # Some Data Modules expect these calls:
                if hasattr(data, "prepare_data"):
                    data.prepare_data()
                if hasattr(data, "setup"):
                    data.setup()
                print("Data module prepared.")
                if hasattr(data, "datasets"):
                    print("Datasets in data module:")
                    for k in getattr(data, "datasets", {}):
                        try:
                            print(f"  {k}: {len(data.datasets[k])}")
                        except Exception:
                            print(f"  {k}: (length unknown)")
            except Exception as e:
                print("⚠️ Failed to instantiate data from config:", e)
                data = None
        else:
            print("No data config found; skipping data instantiation.")

        # Set learning rate on model if config contains base_learning_rate
        try:
            base_lr = config.model.base_learning_rate
            model.learning_rate = base_lr
            print(f"Learning rate set to {model.learning_rate:.2e}")
        except Exception:
            pass

        # --- Run training / testing ---
        if opt.train:
            if data is None:
                raise RuntimeError("Training requested but no data module available in config.")
            trainer.fit(model, data, ckpt_path=getattr(opt, "resume_from_checkpoint", None))
        if not opt.no_test:
            # run test if available
            try:
                if data is not None:
                    trainer.test(model, data, ckpt_path=getattr(opt, "resume_from_checkpoint", None))
                else:
                    print("Skipping test — no data module.")
            except Exception as e:
                print("Test failed:", e)

        print("=== Done ===")

    except Exception as e:
        print("❌ Training failed due to:", e)
        raise
