"""
Displacement prediction network built from stacked cEGCL layers.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MLP

from .layers import cEGCL


class DisplacementNet(nn.Module):
    """Predicts per-point surface displacement from a conditioned point cloud.

    Four `cEGCL` message-passing steps are applied in sequence. By default
    all four steps **share one set of weights** (this matches the model
    reported in the paper — effectively a 4-step recurrent block rather than
    4 independently-parameterized layers). Set `share_weights=False` to give
    each step its own independent parameters instead.

    Parameters
    ----------
    k : int
        Number of nearest neighbors for the k-NN graph built at every layer.
    aggr : str
        Aggregation mode forwarded to each `cEGCL`.
    dropout : float
        Dropout used in the final MLP head.
    feature_dim : int
        Per-point feature dimensionality. Must match the flattened size of
        the `feat` input below (default 256 = 2 * 128, see `feat` shape).
    hidden_dim : int
        Hidden width of the small MLPs inside each `cEGCL`.
    share_weights : bool
        Whether the 4 message-passing steps share one set of weights
        (default, matches the paper) or each gets independent parameters.

    Forward
    -------
    input_points : [B*N, 3]
        Point positions.
    batch : [B*N]
        Batch index per point (PyG convention).
    cond : [B, 6]
        Per-sample probe condition, `[start_xyz, end_xyz]`.
    cond_feat : [B, 4, 128]
        Per-sample condition profile (tissue + xyz channels along the probe).
    point_feat : [B*N, 2, 128]
        Per-point tissue profile features.

    Returns
    -------
    displacement : [B*N, 3]
        Predicted per-point displacement.
    """

    def __init__(self, k: int, aggr: str = "add", dropout: float = 0.0,
                 feature_dim: int = 256, hidden_dim: int = 64,
                 share_weights: bool = True):
        super().__init__()
        out_channels = 3

        def make_mlps():
            phi_e = nn.Sequential(
                nn.Linear(feature_dim * 2 + 1, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            )
            phi_x = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            phi_h = nn.Sequential(
                nn.Linear(hidden_dim + feature_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, feature_dim),
            )
            phi_cond_s = nn.Sequential(
                nn.Linear(3, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, out_channels),
            )
            phi_cond_e = nn.Sequential(
                nn.Linear(3, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, out_channels),
            )
            return phi_e, phi_x, phi_h, phi_cond_s, phi_cond_e

        if share_weights:
            shared_mlps = make_mlps()
            per_layer_mlps = [shared_mlps] * 4
        else:
            per_layer_mlps = [make_mlps() for _ in range(4)]

        self.layers = nn.ModuleList([
            cEGCL(k, *mlps, aggr=aggr) for mlps in per_layer_mlps
        ])

        # Small MLP summarizing the [B, 4, 128] condition profile into 6 dims.
        self.cond_mlp = MLP([128 * 4, 64, 6], dropout=0.2)

        # Final regression head: 4 layers' positions (3 each) + raw probe
        # condition (6) + flattened condition profile (4*128).
        self.head = MLP(
            [3 * 4 + 6 + 4 * 128, 512, 256, 128, out_channels],
            dropout=dropout, norm=None,
        )

    def forward(self, input_points, batch, cond, cond_feat, feat):
        feat = feat.reshape(feat.size(0), -1)          # [B*N, 2, 128] -> [B*N, 256]
        cond_per_point = cond[batch]                    # [B*N, 6]
        cond_feat_flat = cond_feat.reshape(cond_feat.size(0), -1)             # [B, 4, 128] -> [B, 512]
        cond_feat_flat_per_point = cond_feat_flat[batch]               # [B*N, 512]
        cond_feat_final = self.cond_mlp(cond_feat_flat)[batch]          # [B*N, 6]

        cond_s = cond_per_point[:, :3]  # start of the probe
        cond_e = cond_per_point[:, 3:]  # end of the probe

        h = feat
        layer_positions = []
        for layer in self.layers:
            input_points, h = layer(input_points, batch, h, cond_s, cond_e)
            layer_positions.append(input_points)

        head_input = torch.cat([*layer_positions, cond_per_point, cond_feat_flat_per_point], dim=1)
        displacement = self.head(head_input)

        return displacement
