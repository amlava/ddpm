import math

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, n_groups):
        super().__init__()

        assert (in_channels % n_groups == 0) and (out_channels % n_groups == 0)

        self.act = nn.SiLU()

        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')

        self.time_proj = nn.Sequential(nn.Linear(time_dim, out_channels), nn.SiLU())

        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        t: (B, time_dim)
        x: (B, C_in, H, W)

        returns: (B, C_out, H, W)
        """
        out = self.conv1(self.act(self.norm1(x))) # (B, C_out, H, W)

        # add time embedding
        t = self.time_proj(t)[:, :, None, None] # (B, C_out, 1, 1)
        out = t + out

        out = self.conv2(self.act(self.norm2(out)))

        return out + self.skip(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim)
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """
        t: (B,) integer timesteps
        returns: (B, dim)
        """
        t = t.float()
        device = t.device
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / half
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb
    
    def forward(self, t: torch.Tensor):
        """
        t: (B,)
        returns: (B, dim) 
        """
        return self.mlp(TimeEmbedding.timestep_embedding(t, self.dim))


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels, base_channels=8, time_dim=32, n_groups=4):
        super().__init__()
        self.config = dict(
            in_channels=in_channels,
            base_channels=base_channels,
            time_dim=time_dim,
            n_groups=n_groups
        )

        self.time_dim = time_dim
        self.time_embedding = TimeEmbedding(time_dim)

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding='same')

        self.down1 = ResBlock(base_channels, 2 * base_channels, time_dim, n_groups)
        self.down2 = ResBlock(2 * base_channels, 2 * base_channels, time_dim, n_groups)

        self.mid = ResBlock(2 * base_channels, 2 * base_channels, time_dim, n_groups)

        self.up2 = ResBlock(4 * base_channels, base_channels, time_dim, n_groups)
        self.up1 = ResBlock(base_channels, base_channels, time_dim, n_groups)

        self.norm_out = nn.GroupNorm(n_groups, base_channels)
        self.conv_out = nn.Conv2d(base_channels, in_channels, 1, padding='same')

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_embedding(t)
        x = self.conv_in(x)

        x1 = self.down1(x, t)                          # (B, 2C, H, W)
        x2 = self.down2(F.avg_pool2d(x1, 2), t)        # (B, 2C, H/2, W/2)

        x_mid = self.mid(x2, t)                        # (B, 2C, H/2, W/2)

        x = F.interpolate(x_mid, scale_factor=2, mode="nearest")  # (B, 2C, H, W)
        x = self.up2(torch.cat([x, x1], dim=1), t)                 # (B, C, H, W)

        x = self.up1(x, t)

        return self.conv_out(self.norm_out(x))
