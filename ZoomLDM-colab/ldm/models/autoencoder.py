import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer


class VQModel(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, n_embed, embed_dim, ckpt_path=None, ignore_keys=[],
                 image_key="image", colorize_nlabels=None, monitor=None,
                 batch_resize_range=None, scheduler_config=None, lr_g_factor=1.0,
                 remap=None, sane_index_shape=False, use_ema=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if colorize_nlabels is not None:
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        self.batch_resize_range = batch_resize_range
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)

        if ckpt_path:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context:
                print(f"{context}: Switched to EMA weights")
        try:
            yield
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0,3,1,2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range:
            lower, upper = self.batch_resize_range
            size = upper if self.global_step <= 4 else np.random.choice(np.arange(lower, upper+16,16))
            if size != x.shape[2]:
                x = F.interpolate(x, size=size, mode="bicubic")
        return x

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor * self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()), lr=lr_g, betas=(0.5,0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr_d, betas=(0.5,0.9))
        if self.scheduler_config:
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler_list = [
                {"scheduler": LambdaLR(opt_ae, lr_lambda=scheduler.schedule), "interval": "step"},
                {"scheduler": LambdaLR(opt_disc, lr_lambda=scheduler.schedule), "interval": "step"},
            ]
            return [opt_ae, opt_disc], scheduler_list
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
