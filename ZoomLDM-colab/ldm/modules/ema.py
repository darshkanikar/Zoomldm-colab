import torch
from torch import nn

class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_updates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.m_name2s_name = {}
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer(
            "num_updates",
            torch.tensor(0, dtype=torch.int) if use_num_updates else torch.tensor(-1, dtype=torch.int)
        )

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove '.'-character as it is not allowed in buffer names
                s_name = name.replace(".", "")
                self.m_name2s_name[name] = s_name
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

    @torch.no_grad()
    def forward(self, model):
        decay = self.decay
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates.item()) / (10 + self.num_updates.item()))

        one_minus_decay = 1.0 - decay
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())

        for key, param in m_param.items():
            if param.requires_grad:
                s_name = self.m_name2s_name[key]
                shadow_params[s_name] = shadow_params[s_name].to(param.device)
                shadow_params[s_name].sub_(one_minus_decay * (shadow_params[s_name] - param))
            else:
                assert key not in self.m_name2s_name

    @torch.no_grad()
    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key, param in m_param.items():
            if param.requires_grad:
                s_name = self.m_name2s_name[key]
                param.data.copy_(shadow_params[s_name].to(param.device).data)
            else:
                assert key not in self.m_name2s_name

    def store(self, parameters):
        """Save the current parameters for restoring later."""
        self.collected_params = [param.clone().detach() for param in parameters]

    def restore(self, parameters):
        """Restore parameters stored with `store`."""
        for stored, param in zip(self.collected_params, parameters):
            param.data.copy_(stored.data)
