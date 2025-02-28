import math

import torch
from torch import nn


# slightly modified implementation from
# https://github.com/jxhe/unify-parameter-efficient-tuning/blob/3222ce2c0079566a28043e22380eb4ab6ad14389/petl/petl_factory.py#L396
class Adapter_Layer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.n_embd = config.n_embd
        self.down_size = config.adapter_bottleneck_dim
        self.adapter_dropout = config.adapter_dropout
        self.adapter_scale = config.adapter_scale
        self.adapter_layernorm_option = getattr(config, 'adapter_layernorm_option', 'in')
        # self.non_linearity = args.non_linearity  # use ReLU by default

        self.adapter_layer_norm_before = None
        if self.adapter_layernorm_option == "in" or self.adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if self.adapter_scale == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(self.adapter_scale)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        if self.adapter_dropout > 0:
            self.dropout = nn.Dropout(p=self.adapter_dropout)
        else:
            self.lora_dropout = lambda x: x

        # init params with lora init
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = self.dropout(down)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output
