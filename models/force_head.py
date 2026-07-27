"""
Force estimation head.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool


class ForceHead(nn.Module):
    """Predicts a scalar contact force from per-point tissue/displacement features.

    This is intentionally a separate, optional module: it can be trained
    jointly with `DisplacementNet`, trained separately against a frozen
    displacement net, or even trained against ground-truth displacements
    instead of predicted ones. The paper reports comparable force accuracy
    across these setups, so if you don't need per-point displacement at all,
    you can drop `DisplacementNet` entirely and feed this head from any other
    displacement source.

    Parameters
    ----------
    in_dim : int
        Input size of the point-wise MLP: 3 (relative displacement) +
        384 (3 flattened, 128-wide relative feature channels) = 387 by default.
    hidden_dim : int
        Hidden width of the point-wise MLP; the pooled feature going into
        `force_mlp` is `2 * hidden_dim` (max-pool concatenated with mean-pool).

    Forward
    -------
    input_points : [B*N, >=2]
        Point coordinates; only the first two columns (x, y) are used.
    batch : [B*N]
        Batch index per point (PyG convention).
    displacement : [B*N, 3]
        Predicted (or ground-truth) per-point displacement.
    cond : [B, 6]
        Per-sample probe condition, `[start_xyz, end_xyz]`.
    point_feat : [B*N, 2, 128]
        Per-point tissue profile features; only channel 0 is used.
    cond_feat : [B, 4, 128]
        Per-sample probe profile features.

    Returns
    -------
    force : [B]
        Predicted scalar contact force per sample.
    """

    def __init__(self, in_dim: int = 387, hidden_dim: int = 128):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.force_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, input_points, batch, displacement, cond, point_feat, cond_feat):
        cond_end = cond[:, 3:][batch]                       # [B*N, 3]
        rel_disp = displacement - cond_end                    # [B*N, 3]

        cond_feat = cond_feat[:, :3, :][batch]               # [B*N, 3, 128]

        point_feat = point_feat[:, 0:1, :]                    # [B*N, 1, 128] tissue channel
        x_coord = input_points[:, 0].view(-1, 1, 1).expand(-1, 1, 128)
        y_coord = input_points[:, 1].view(-1, 1, 1).expand(-1, 1, 128)
        point_feat = torch.cat([point_feat, x_coord, y_coord], dim=1)  # [B*N, 3, 128]

        rel_feats = (point_feat - cond_feat).reshape(point_feat.size(0), -1)  # [B*N, 384]
        fused = torch.cat([rel_disp, rel_feats], dim=1)        # [B*N, 387]

        x = self.point_mlp(fused)
        pooled = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        return self.force_mlp(pooled).squeeze(-1)
