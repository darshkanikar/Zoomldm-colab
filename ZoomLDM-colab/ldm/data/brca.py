from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MAG_DICT = {
    "20x": 0,
    "10x": 1,
    "5x": 2,
    "2_5x": 3,
    "1_25x": 4,
    "0_625x": 5,
    "0_3125x": 6,
    "0_15625x": 7,
}

MAG_NUM_IMGS = {
    "20x": 12_509_760,
    "10x": 3_036_288,
    "5x": 752_000,
    "2_5x": 187_280,
    "1_25x": 57_090,
    "0_625x": 20_679,
    "0_3125x": 7_923,
    "0_15625x": 2489,
}


class TCGADataset(Dataset):
    def __init__(self, config=None):
        config = config or {}
        self.root = Path(config.get("root", "."))
        self.mag = config.get("mag", None)
        self.keys = list(MAG_DICT.keys())
        self.feat_target_size = config.get("feat_target_size", -1)
        self.return_image = config.get("return_image", False)
        self.normalize_ssl = config.get("normalize_ssl", False)

    def __len__(self):
        if self.mag:
            return MAG_NUM_IMGS[self.mag]
        return MAG_NUM_IMGS["20x"]

    def __getitem__(self, idx):
        # select magnitude
        if self.mag:
            mag_choice = self.mag
        else:
            mag_choice = np.random.choice(self.keys)
            idx = np.random.randint(0, MAG_NUM_IMGS[mag_choice])

        # determine folder
        folder = str(idx // 1_000_000)
        folder_path = self.root / f"{mag_choice}/{folder}"

        # load VAE features
        try:
            vae_feat = np.load(folder_path / f"{idx}_vae.npy").astype(np.float32)
            if vae_feat.shape != (3, 64, 64):
                raise ValueError(f"Unexpected VAE shape {vae_feat.shape} for idx {idx}")
            vae_feat = torch.tensor(vae_feat)
        except:
            # fallback: pick another index
            return self.__getitem__(np.random.randint(len(self)))

        # load SSL features
        try:
            ssl_feat = np.load(folder_path / f"{idx}_uni_grid.npy").astype(np.float32)
            if ssl_feat.ndim == 1:
                ssl_feat = ssl_feat[:, None]
            h = int(np.sqrt(ssl_feat.shape[1]))
            ssl_feat = torch.tensor(ssl_feat.reshape((-1, h, h)))  # [C,H,W]

            # resize SSL features
            if self.feat_target_size != -1 and h > self.feat_target_size:
                ssl_feat = F.adaptive_avg_pool2d(ssl_feat, (self.feat_target_size, self.feat_target_size))

            # normalize
            if self.normalize_ssl:
                mean = ssl_feat.mean(dim=(1, 2), keepdim=True)
                std = ssl_feat.std(dim=(1, 2), keepdim=True)
                ssl_feat = (ssl_feat - mean) / (std + 1e-8)

        except:
            ssl_feat = torch.zeros((1, 64, 64), dtype=torch.float32)

        # load image (optional)
        if self.return_image:
            try:
                image = np.load(folder_path / f"{idx}_img.npy")
                image = torch.tensor(image.astype(np.float32))
            except:
                image = torch.zeros((3, 64, 64), dtype=torch.float32)
        else:
            image = torch.ones((1, 1, 1, 3), dtype=torch.float32)

        return {
            "image": image,
            "vae_feat": vae_feat,
            "ssl_feat": ssl_feat,
            "idx": idx,
            "mag": MAG_DICT[mag_choice],
        }
