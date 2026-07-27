"""
Combined displacement + force model.
"""
import torch.nn as nn

from .displacement_net import DisplacementNet
from .force_head import ForceHead


class SoftTissueNet(nn.Module):
    """One model wrapping `DisplacementNet` and an optional `ForceHead`.

    Most users just want a single `forward()` that returns both outputs;
    this class provides that, while still letting you:
      - drop the force branch entirely (`predict_force=False`) if you only
        care about displacement, or
      - train the force head separately later — it stays a standalone
        module (`self.force_head`) either way, so you can also freeze
        `self.displacement_net` and fine-tune just the force head, or
        swap it out for one trained on ground-truth displacements.

    Parameters
    ----------
    k, aggr, dropout, feature_dim, hidden_dim, share_weights :
        Forwarded to `DisplacementNet` — see its docstring.
    predict_force : bool
        If True (default), also builds and runs a `ForceHead`. If False,
        the force branch (and its parameters) is never created.

    Forward
    -------
    input_points, batch, cond, cond_feat, point_feat :
        See `DisplacementNet.forward`.

    Returns
    -------
    displacement : [B*N, 3]
    force : [B] or None
        `None` whenever `predict_force=False`.
    """

    def __init__(self, k: int, aggr: str = "add", dropout: float = 0.0,
                 feature_dim: int = 256, hidden_dim: int = 64,
                 share_weights: bool = True, predict_force: bool = True):
        super().__init__()
        self.displacement_net = DisplacementNet(
            k=k, aggr=aggr, dropout=dropout, feature_dim=feature_dim,
            hidden_dim=hidden_dim, share_weights=share_weights,
        )
        self.force_head = ForceHead() if predict_force else None

    def forward(self, input_points, batch, cond, cond_feat, point_feat):
        displacement = self.displacement_net(input_points, batch, cond, cond_feat, point_feat)

        force = None
        if self.force_head is not None:
            force = self.force_head(input_points, batch, displacement, cond, point_feat, cond_feat)

        return displacement, force
