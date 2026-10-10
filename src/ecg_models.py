"""E0 ResNet and the configured physical-amplitude local branch."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def masked_bce(logits, labels, weights=None):
    """Unknown labels are NaN; normalize each task by its known labels only."""
    loss = logits.sum() * 0
    weights = [1.] * logits.shape[1] if weights is None else weights
    for task, weight in enumerate(weights):
        known = torch.isfinite(labels[:, task])
        if known.any():
            loss = loss + weight * F.binary_cross_entropy_with_logits(logits[known, task], labels[known, task])
    return loss


class Residual(nn.Module):
    def __init__(self, cin, cout, kernel=7, stride=2):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(cin, cout, kernel, stride, kernel // 2), nn.GELU(),
                                  nn.Conv1d(cout, cout, kernel, 1, kernel // 2))
        self.skip = nn.Conv1d(cin, cout, 1, stride) if cin != cout or stride != 1 else nn.Identity()

    def forward(self, x):
        return F.gelu(self.body(x) + self.skip(x))


class ResNet(nn.Module):
    def __init__(self, targets=("OMI",)):
        super().__init__()
        self.targets = tuple(targets)
        self.encoder = nn.Sequential(nn.Conv1d(12, 32, 15, 2, 7), nn.GELU(),
                                     Residual(32, 48), Residual(48, 64), Residual(64, 96), Residual(96, 128))
        self.heads = nn.ModuleList([nn.Linear(256, 1) for _ in targets])

    def forward(self, x):
        x = self.encoder(x)
        z = torch.cat((x.mean(2), x.amax(2)), 1)
        return torch.cat([head(z) for head in self.heads], 1)


class TemporalResidual(nn.Module):
    def __init__(self, width, kernel, dilation):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(width, width, kernel, padding=dilation * (kernel // 2),
                                           dilation=dilation, groups=width), nn.GELU(), nn.Conv1d(width, width, 1))

    def forward(self, x):
        return x + self.body(x)


class LocalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config["local_branch"]
        stem = c["stem"]
        self.scale = c["input_scale_mv"]
        self.stem = nn.ModuleList([nn.Conv1d(1, stem["channels_each"], k, stem["stride"], k // 2)
                                  for k in stem["kernels"]])
        r = c["residual_downsampling"]
        width = r["channels"]
        widths = [len(self.stem) * stem["channels_each"]] + [width] * r["num_stages"]
        self.down = nn.Sequential(*[Residual(a, b, r["kernel"], r["stride_each"])
                                    for a, b in zip(widths[:-1], widths[1:])])
        attn = c["lead_attention"]
        self.use_attention = attn.get("enabled", True)
        self.lead_id = nn.Parameter(torch.zeros(12, width))
        nn.init.normal_(self.lead_id, std=.02)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, attn["heads"], batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, attn["ffn_dim"]), nn.GELU(), nn.Linear(attn["ffn_dim"], width))
        temporal = c["temporal_convolution"]
        self.temporal = nn.Sequential(*[TemporalResidual(width, temporal["kernel"], d)
                                        for d in temporal["dilations"]])
        self.use_summary = c["amplitude_summary"].get("enabled", True)
        self.register_buffer("summary_mean", torch.zeros(48))
        self.register_buffer("summary_std", torch.ones(48))
        self.project = nn.Sequential(nn.Linear(c["pooled_dim"] + (48 if self.use_summary else 0), c["output_dim"]),
                                     nn.GELU(), nn.Dropout(config["fusion"]["dropout"]))

    @staticmethod
    def amplitude_summary(x):
        x = x.float()
        q = torch.quantile(x, torch.tensor([.05, .95], device=x.device), dim=2)
        return torch.stack((q[0], q[1], x.std(2, correction=0), x.diff(dim=2).std(2, correction=0)), 2).flatten(1)

    def forward(self, x):
        batch = x.shape[0]
        summary = (self.amplitude_summary(x) - self.summary_mean) / self.summary_std if self.use_summary else None
        h = (x / self.scale).reshape(batch * 12, 1, 5000)
        h = self.down(torch.cat([F.gelu(conv(h)) for conv in self.stem], 1))
        width, time = h.shape[1:]
        if self.use_attention:
            h = h.reshape(batch, 12, width, time).permute(0, 3, 1, 2).reshape(batch * time, 12, width)
            h = h + self.lead_id
            normalized = self.norm(h)
            h = h + self.attention(normalized, normalized, normalized, need_weights=False)[0]
            h = h + self.ffn(self.ffn_norm(h))
            h = h.reshape(batch, time, 12, width).permute(0, 2, 3, 1).reshape(batch * 12, width, time)
        h = self.temporal(h)
        pooled = torch.cat((h.mean(2), h.amax(2)), 1).reshape(batch, -1)
        if summary is not None:
            pooled = torch.cat((pooled, summary), 1)
        return self.project(pooled)


class LocalModel(nn.Module):
    def __init__(self, config, targets=("OMI",), phenotype_targets=()):
        super().__init__()
        self.local = LocalEncoder(config)
        self.heads = nn.ModuleList([nn.Linear(config["local_branch"]["output_dim"], 1) for _ in targets])
        self.phenotype = nn.Linear(config["local_branch"]["output_dim"], len(phenotype_targets)) if phenotype_targets else None

    def forward(self, x):
        z = self.local(x)
        return torch.cat([head(z) for head in self.heads], 1)


class FusionModel(LocalModel):
    """Accept ONLY provenance-validated cached vectors from a frozen backbone."""
    def __init__(self, config, targets=("OMI",)):
        super().__init__(config, targets)
        g, f = config["global_branch"], config["fusion"]
        self.global_project = nn.Sequential(nn.Linear(g["input_projection_dim"], g["output_dim"]),
                                            nn.GELU(), nn.Dropout(f["dropout"]))
        self.fuse = nn.Sequential(nn.Linear(f["input_dim"], f["output_dim"]), nn.GELU(), nn.Dropout(f["dropout"]))

    def forward(self, x, embedding):
        z = self.fuse(torch.cat((self.global_project(embedding), self.local(x)), 1))
        return torch.cat([head(z) for head in self.heads], 1)


def architecture():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/model_architecture_v1.json").read_text(encoding="utf-8"))


def make_model(name, targets):
    if name == "resnet":
        return ResNet(targets)
    if name == "local":
        return LocalModel(architecture(), targets)
    raise ValueError("supported trainable comparisons: resnet, local; foundation models require audited checkpoints")
