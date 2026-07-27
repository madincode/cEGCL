"""
Conditional Equivariant Graph Convolution Layer (cEGCL).

Extends the E(n)-Equivariant Graph Neural Network message-passing scheme of
Satorras et al., "E(n) Equivariant Graph Neural Networks", 2021
(https://arxiv.org/pdf/2102.09844) with two additional conditioning terms —
the start and end location of the applied probe — added directly to the
coordinate update (their Eq. 4).

Shapes below use:
    B - batch size
    N - number of points per sample
    k - number of nearest neighbors per point
    F - per-point feature dimension
    H - hidden dimension of the layer's internal MLPs
    E - number of directed edges in the k-NN graph (= B*N*k)
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_cluster import knn


class cEGCL(MessagePassing):
    """One conditional, equivariant message-passing step.

    Builds a k-NN graph over the current point positions and performs one
    round of EGNN-style message passing (Eqs. 3-6 of Satorras et al., 2021),
    with the coordinate update further conditioned on two fixed 3D locations
    `cond_s` and `cond_e` (the start and end of the applied probe).

    Parameters
    ----------
    k : int
        Number of nearest neighbors used to build the graph at every call.
    phi_e : nn.Module
        Edge/message MLP.               [E, 2F+1] -> [E, H]
    phi_x : nn.Module
        Coordinate-update MLP.          [E, H]    -> [E, 1]
    phi_h : nn.Module
        Feature-update MLP.             [B*N, H+F] -> [B*N, F]
    phi_cond_s, phi_cond_e : nn.Module
        Map a (point - condition) offset to a coordinate-update contribution.
                                         [B*N, 3] -> [B*N, 3]
    aggr : str
        Forwarded to `MessagePassing.__init__` for API compatibility; the
        actual aggregation is implemented explicitly in `aggregate()` below.
    """

    def __init__(self, k: int, phi_e: nn.Module, phi_x: nn.Module, phi_h: nn.Module,
                 phi_cond_s: nn.Module, phi_cond_e: nn.Module,
                 aggr: str = "add", **kwargs):
        super().__init__(aggr=aggr, flow="source_to_target", **kwargs)
        self.phi_e = phi_e
        self.phi_x = phi_x
        self.phi_h = phi_h
        self.phi_cond_s = phi_cond_s
        self.phi_cond_e = phi_cond_e
        self.k = k
        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialize sub-modules that expose their own `reset_parameters`.

        Note: `phi_e`/`phi_x`/`phi_h` are typically plain `nn.Sequential`
        containers, which do not implement `reset_parameters` themselves
        (only the individual `nn.Linear` layers inside them do). This method
        is therefore a no-op for the default MLPs built in `DisplacementNet`
        — kept only for API compatibility / for custom MLP modules that do
        implement it.
        """
        for mlp in (self.phi_e, self.phi_x, self.phi_h, self.phi_cond_s, self.phi_cond_e):
            if hasattr(mlp, "reset_parameters"):
                mlp.reset_parameters()

    def forward(self, x: torch.Tensor, batch: torch.Tensor, x_feat: torch.Tensor,
                cond_s: torch.Tensor, cond_e: torch.Tensor):
        """
        Parameters
        ----------
        x : [B*N, 3]        current point positions
        batch : [B*N]       batch index per point (PyG convention)
        x_feat : [B*N, F]   current per-point features
        cond_s, cond_e : [B*N, 3]
            Start/end probe location, already broadcast to one row per point.

        Returns
        -------
        x_new : [B*N, 3]   updated point positions
        h_new : [B*N, F]   updated point features
        """
        edge_index = knn(x, x, self.k, batch, batch).flip([0])
        x_update, h_update = self.propagate(
            edge_index, x=(x, x), batch=batch, x_feat=x_feat,
            cond_s=cond_s, cond_e=cond_e,
        )
        return x + x_update, h_update

    def message(self, x_i, x_j, x_feat_i, x_feat_j,
                cond_s_i, cond_s_j, cond_e_i, cond_e_j):
        """Eqs. 3-4: per-edge message and its coordinate-update contribution."""
        diff = x_i - x_j
        dist_sq = diff.pow(2).sum(dim=-1, keepdim=True)  # [E, 1]

        edge_attr = torch.cat([x_feat_i, x_feat_j, dist_sq], dim=-1)  # [E, 2F+1]
        m_ij = self.phi_e(edge_attr)     # [E, H]
        weight = self.phi_x(m_ij)        # [E, 1]
        x_update = diff * weight          # [E, 3]
        return x_update, m_ij

    def aggregate(self, message_outputs, index, dim_size):
        """Sum coordinate updates and messages per target node (Eqs. 4-5)."""
        x_update, m_ij = message_outputs
        norm = 1.0 / (self.k - 1)

        x_sum = torch.zeros(dim_size, x_update.size(-1), device=x_update.device)
        x_sum.scatter_add_(0, index.unsqueeze(-1).expand_as(x_update), x_update)

        m_sum = torch.zeros(dim_size, m_ij.size(-1), device=m_ij.device)
        m_sum.scatter_add_(0, index.unsqueeze(-1).expand_as(m_ij), m_ij)

        return norm * x_sum, m_sum

    def update(self, aggregated, x, x_feat, cond_s, cond_e):
        """Eq. 4 (+ probe conditioning) and Eq. 6."""
        x, _ = x  # `propagate()` was given the pair (x, x); both halves are equal
        x_update, m_i = aggregated

        to_start = x - cond_s
        to_end = x - cond_e
        x_update = self.phi_cond_s(to_start) + self.phi_cond_e(to_end) + x_update

        h_update = x_feat + self.phi_h(torch.cat([x_feat, m_i], dim=-1))
        return x_update, h_update
