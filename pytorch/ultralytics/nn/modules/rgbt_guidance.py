from __future__ import annotations

import torch
import torch.nn as nn


class DualModalRefine(nn.Module):
    """Light per-modal residual refinement with channel reweighting."""

    def __init__(self, c1: int, c2: int | None = None, se_ratio: int = 16, gamma_init: float = 0.10):
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c1 != c2:
            raise ValueError(f"DualModalRefine requires c1 == c2, got {c1} and {c2}")
        hidden = max(c1 // max(se_ratio, 1), 8)
        self.pre = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(x)
        y = y * self.se(self.pool(y))
        return x + torch.tanh(self.gamma) * y


class TargetAwareGuide(nn.Module):
    """Use IR targetness to modulate VIS features."""

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        gate_reduction: int = 4,
        beta_init: float = 0.05,
        heatmap_hidden: int = 64,
        spatial_mode: bool = False,
        channel_mode: bool = False,
    ):
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c1 != c2:
            raise ValueError(f"TargetAwareGuide requires c1 == c2, got {c1} and {c2}")
        heatmap_hidden = max(int(heatmap_hidden), 16)
        self.spatial_mode = bool(spatial_mode)
        self.channel_mode = bool(channel_mode)
        self.ir_heatmap_head = nn.Sequential(
            nn.Conv2d(c1, heatmap_hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(heatmap_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(heatmap_hidden, 1, 1, bias=True),
        )
        if self.channel_mode:
            self.channel_mlp = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c1, heatmap_hidden, 1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(heatmap_hidden, c1, 1, bias=True),
                nn.Sigmoid(),
            )
        self.beta = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("TargetAwareGuide expects [vis_feature, ir_feature].")
        vis, ir = x
        if vis.shape != ir.shape:
            raise ValueError(f"TargetAwareGuide expects same shape for VIS/IR, got {tuple(vis.shape)} vs {tuple(ir.shape)}")

        channel_mode = bool(getattr(self, "channel_mode", False))
        spatial_mode = bool(getattr(self, "spatial_mode", False))

        if channel_mode:
            if not hasattr(self, "channel_mlp"):
                raise AttributeError("TargetAwareGuide channel_mode=True requires channel_mlp to be initialized.")
            p_ir = self.channel_mlp(ir)
        else:
            heatmap = torch.sigmoid(self.ir_heatmap_head(ir))
            if spatial_mode:
                p_ir = heatmap
            else:
                p_ir = heatmap.mean(dim=(2, 3), keepdim=True)

        return vis * (1.0 + self.beta * p_ir)


class CrossModalGuide(nn.Module):
    """VIS-dominant cross-modal attention guide."""

    def __init__(self, c1: int, c2: int | None = None, num_heads: int = 4, alpha_init: float = 0.08):
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c1 != c2:
            raise ValueError(f"CrossModalGuide requires c1 == c2, got {c1} and {c2}")
        if c1 % num_heads != 0:
            raise ValueError(f"channels ({c1}) must be divisible by heads ({num_heads})")
        self.channels = c1
        self.num_heads = num_heads
        self.head_dim = c1 // num_heads
        self.q = nn.Conv2d(c1, c1, 1, bias=False)
        self.k = nn.Conv2d(c1, c1, 1, bias=False)
        self.v = nn.Conv2d(c1, c1, 1, bias=False)
        self.proj = nn.Sequential(nn.Conv2d(c1, c1, 1, bias=False), nn.BatchNorm2d(c1))
        self.alpha_logit = nn.Parameter(torch.tensor(float(alpha_init)).clamp(1e-4, 0.95).logit())

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return x.view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("CrossModalGuide expects [vis_feature, ir_feature].")
        vis, ir = x
        if vis.shape != ir.shape:
            raise ValueError(f"CrossModalGuide expects same shape for VIS/IR, got {tuple(vis.shape)} vs {tuple(ir.shape)}")
        q = self._reshape_heads(self.q(vis))
        k = self._reshape_heads(self.k(ir))
        v = self._reshape_heads(self.v(ir))
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (self.head_dim**0.5), dim=-1)
        ctx = attn @ v
        b, _, h, w = vis.shape
        ctx = ctx.transpose(2, 3).contiguous().view(b, self.channels, h, w)
        alpha = torch.sigmoid(self.alpha_logit)
        return vis + alpha * self.proj(ctx)


class IGSA(nn.Module):
    """IR-Guided Split Attention.

    IGSA replaces the single-modal C2PSA context block at P5. The VIS feature
    keeps the spatial path, while IR only provides channel-wise modulation for
    the attention branch. This keeps the fusion conservative under target-level
    RGB/IR misalignment.
    """

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        n: int = 1,
        e: float = 0.5,
        se_ratio: int = 16,
        delta_init: float = 0.05,
    ):
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c1 != c2:
            raise ValueError(f"IGSA requires c1 == c2, got {c1} and {c2}")
        self.c = int(c1 * e)
        if self.c <= 0:
            raise ValueError(f"IGSA hidden channels must be positive, got {self.c}")

        from ultralytics.nn.modules.block import PSABlock
        from ultralytics.nn.modules.conv import Conv

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=max(1, self.c // 64)) for _ in range(int(n)))
        )

        hidden = max(self.c // max(int(se_ratio), 1), 8)
        self.ir_pool = nn.AdaptiveAvgPool2d(1)
        self.ir_se = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.c, 1, bias=True),
            nn.Sigmoid(),
        )
        self.cv2 = Conv(2 * self.c, c2, 1, 1)
        self.delta = nn.Parameter(torch.tensor(float(delta_init), dtype=torch.float32))

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("IGSA expects [vis_feature, ir_feature].")
        vis, ir = x
        if vis.shape != ir.shape:
            raise ValueError(f"IGSA expects same shape for VIS/IR, got {tuple(vis.shape)} vs {tuple(ir.shape)}")

        a, b = self.cv1(vis).split((self.c, self.c), dim=1)
        b = self.m(b)
        w_ir = self.ir_se(self.ir_pool(ir))
        b = b * (1.0 + torch.tanh(self.delta) * w_ir)
        return self.cv2(torch.cat((a, b), dim=1))
