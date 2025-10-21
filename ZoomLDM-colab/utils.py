import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict

def show(img_lis, names):
    """
    Displays a list of images with names using matplotlib.
    Assumes img_lis contains PIL Images or NumPy arrays.
    """
    fig, axs = plt.subplots(nrows=len(img_lis), ncols=1, squeeze=False, figsize=(20, 4))
    for ax, name, img in zip(axs, names, img_lis):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
        ax[0].imshow(np.asarray(img))
        ax[0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        ax[0].set_ylabel(name)
    fig.tight_layout()

def collate_fn(batch):
    """
    Custom collate function to combine dict-based samples into a batch tensor.
    Expected keys: image, ssl_feat, ssl_feat_unpooled, mag
    """
    outputs = defaultdict(list)

    for item in batch:
        # Convert image to tensor
        pil_img = item['image'].convert("RGB")
        img_array = np.array(pil_img)
        outputs['image'].append(torch.from_numpy(img_array).permute(2, 0, 1))  # CHW format

        # Handle optional fields
        for k in ['ssl_feat', 'ssl_feat_unpooled', 'mag']:
            if k in item:
                value = item[k]
                if not torch.is_tensor(value):
                    value = torch.tensor(value)
                outputs[k].append(value)

    # Stack lists into batch tensors
    for k, v in outputs.items():
        try:
            outputs[k] = torch.stack(v)
        except Exception:
            print(f"Warning: Could not stack key '{k}' — leaving as list.")

    return outputs
