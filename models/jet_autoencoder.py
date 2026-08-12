"""
sonic.models.jet_autoencoder
============================
EdgeConv‑based graph autoencoder with a physics‑coupled decoder.
"""

import torch
import torch.nn as nn


class EdgeConv(nn.Module):
    """Dynamic graph convolution block with residual skip."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 20):
        super().__init__()
        self.k = k
        self.conv_xc = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv_diff = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, idx=None):
        b, c, n = x.shape
        k = min(self.k, n)

        if idx is None:
            inner = -2 * torch.bmm(x.transpose(1, 2), x)
            xx = x.pow(2).sum(1, keepdim=True)
            dist = -(xx + inner + xx.transpose(1, 2))
            idx = dist.topk(k, -1)[1]

        x_trans = x.transpose(1, 2)
        batch_idx = torch.arange(b, device=x.device).view(-1, 1, 1)
        nbrs = x_trans[batch_idx, idx].permute(0, 3, 1, 2)
        xc = x.unsqueeze(3)

        out = self.conv_xc(xc) + self.conv_diff(nbrs - xc)
        out = self.bn(out)
        out = self.act(out)
        return out.mean(3) + self.skip(x), idx


class JetAE_V12(nn.Module):
    """
    Graph autoencoder: 4‑layer EdgeConv encoder → FC latent → physics‑branched decoder.
    """

    N_FEAT = 8

    def __init__(
        self,
        n_part: int = 30,
        latent: int = 128,
        ch: int = 128,
        static_graph: bool = False,
    ):
        super().__init__()
        self.n_part = n_part
        self.latent = latent
        self.static_graph = static_graph

        self.e1 = EdgeConv(self.N_FEAT, ch // 2, k=20)
        self.e2 = EdgeConv(ch // 2, ch, k=20)
        self.e3 = EdgeConv(ch, ch * 2, k=16)
        self.e4 = EdgeConv(ch * 2, ch * 2, k=8)

        self.fc_z = nn.Sequential(
            nn.Linear(ch * 4, latent * 2),
            nn.BatchNorm1d(latent * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(latent * 2, latent),
        )

        self.dec_backbone = nn.Sequential(
            nn.Linear(latent, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
        )

        self.out_spatial = nn.Linear(512, n_part * 4)
        self.out_energy = nn.Linear(512, n_part * 2)
        self.softplus = nn.Softplus()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h, idx1 = self.e1(x.transpose(1, 2))
        h, _ = self.e2(h, idx1 if self.static_graph else None)
        h, _ = self.e3(h, idx1 if self.static_graph else None)
        h, _ = self.e4(h, idx1 if self.static_graph else None)
        return self.fc_z(torch.cat([h.max(2)[0], h.mean(2)], 1))

    def forward(self, x: torch.Tensor):
        b = x.size(0)
        z = self.encode(x)
        latent_features = self.dec_backbone(z)

        spatial = self.out_spatial(latent_features).view(b, self.n_part, 4)
        energy = self.out_energy(latent_features).view(b, self.n_part, 2)
        energy = self.softplus(energy)

        recon = torch.cat(
            [spatial[:, :, 0:2], energy, spatial[:, :, 2:4]], dim=2
        )
        return recon, z
