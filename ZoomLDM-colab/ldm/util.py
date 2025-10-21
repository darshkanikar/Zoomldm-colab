import importlib
import torch
from torch import optim
import numpy as np
from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont

# ---------------------
# Helper decorators
# ---------------------
def autocast(f):
    """Automatically uses GPU AMP if available"""
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available(),
            dtype=torch.get_autocast_gpu_dtype() if torch.cuda.is_available() else torch.float32,
            cache_enabled=torch.is_autocast_cache_enabled() if torch.cuda.is_available() else False,
        ):
            return f(*args, **kwargs)
    return do_autocast

# ---------------------
# Logging text as image
# ---------------------
def log_txt_as_img(wh, xc, size=10):
    """
    wh: tuple (width, height)
    xc: list of strings
    Returns tensor: (B, C, H, W)
    """
    b = len(xc)
    txts = []
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except:
            font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start+nc] for start in range(0, len(xc[bi]), nc))
        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("⚠️ Cannot encode string for logging. Skipping.")
        arr = np.array(txt).transpose(2,0,1)/127.5 - 1.0
        txts.append(arr)
    txts = torch.tensor(np.stack(txts))
    return txts

# ---------------------
# Tensor type checks
# ---------------------
def ismap(x):
    return isinstance(x, torch.Tensor) and len(x.shape) == 4 and x.shape[1] > 3

def isimage(x):
    return isinstance(x, torch.Tensor) and len(x.shape) == 4 and x.shape[1] in [1,3]

# ---------------------
# Utility functions
# ---------------------
def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else (d() if isfunction(d) else d)

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def count_params(model, verbose=False):
    total = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total*1.e-6:.2f} M params.")
    return total

# ---------------------
# Instantiate objects from config
# ---------------------
def instantiate_from_config(config):
    if "target" not in config:
        if config in ["__is_first_stage__", "__is_unconditional__"]:
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))

def get_obj_from_str(string, reload=False):
    module_name, cls_name = string.rsplit(".",1)
    module = importlib.import_module(module_name)
    if reload:
        importlib.reload(module)
    return getattr(module, cls_name)

# ---------------------
# AdamW with EMA of params
# ---------------------
class AdamWwithEMAandWings(optim.Optimizer):
    """AdamW optimizer with EMA for parameters."""
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-2,
                 amsgrad=False, ema_decay=0.9999, ema_power=1.0, param_names=()):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
            ema_decay=ema_decay, ema_power=ema_power, param_names=param_names
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad, grads, exp_avgs, exp_avg_sqs, ema_params_with_grad, state_steps = [], [], [], [], [], []
            max_exp_avg_sqs = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]
            ema_decay = group["ema_decay"]
            ema_power = group["ema_power"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["param_exp_avg"] = p.detach().float().clone()
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                ema_params_with_grad.append(state["param_exp_avg"])
                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                state["step"] += 1
                state_steps.append(state["step"])

            optim._functional.adamw(
                params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
                amsgrad=amsgrad, beta1=beta1, beta2=beta2,
                lr=group["lr"], weight_decay=group["weight_decay"], eps=group["eps"]
            )

            cur_ema_decay = min(ema_decay, 1 - state["step"] ** -ema_power)
            for param, ema_param in zip(params_with_grad, ema_params_with_grad):
                ema_param.mul_(cur_ema_decay).add_(param.float(), alpha=1 - cur_ema_decay)

        return loss
