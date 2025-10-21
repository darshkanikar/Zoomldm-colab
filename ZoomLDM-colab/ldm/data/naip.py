from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange

MAG_DICT = {
    "1x": 0,
    "2x": 1,
    "3x": 2,
    "4x": 3,
}

MAG_NUM_IMGS = {
    "1x": 365_119,
    "2x": 94_263,
    "3x": 25_690,
    "4x": 8_772,
}


class NAIPDataset(Dataset):
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
        return sum(MAG_NUM_IMGS.values())

    def __getitem__(self, idx):
        # Select magnitude
        if self.mag:
            mag_choice = self.mag
        else:
            mag_choice = np.random.choice(self.keys)
            idx = np.random.randint(0, MAG_NUM_IMGS[mag_choice])

        folder_path = self.root / f"{mag_choice}/"

        # Load VAE features
        try:
            vae_feat = np.load(folder_path / f"{idx}_vae.npy").astype(np.float32)
            vae_feat = torch.tensor(vae_feat)
        except:
            return self.__getitem__(np.random.randint(len(self)))

        # Load SSL features
        try:
            ssl_feat = np.load(folder_path / f"{idx}_dino_grid.npy").astype(np.float32)
            h = int(np.sqrt(ssl_feat.shape[0]))
            ssl_feat = torch.tensor(rearrange(ssl_feat, "(h1 h2) dim -> dim h1 h2", h1=h))

            # Resize SSL features
            if self.feat_target_size != -1 and h > self.feat_target_size:
                ssl_feat = F.adaptive_avg_pool2d(ssl_feat, (self.feat_target_size, self.feat_target_size))

            # Normalize SSL features
            if self.normalize_ssl:
                mean = ssl_feat.mean(dim=(1, 2), keepdim=True)
                std = ssl_feat.std(dim=(1, 2), keepdim=True)
                ssl_feat = (ssl_feat - mean) / (std + 1e-8)
        except:
            ssl_feat = torch.zeros((1, 64, 64), dtype=torch.float32)

        # Load image (optional)
        if self.return_image:
            try:
                image = np.array(folder_path / f"{idx}.jpg", dtype=np.uint8)
            except:
                image = np.zeros((3, 64, 64), dtype=np.uint8)
        else:
            image = torch.ones((1, 1, 1, 3), dtype=torch.float32)

        return {
            "image": image,
            "vae_feat": vae_feat,
            "ssl_feat": ssl_feat,
            "idx": idx,
            "mag": MAG_DICT[mag_choice],
            "img_path": str(folder_path / f"{idx}.jpg"),
        }
