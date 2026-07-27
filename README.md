# cEGNN — Conditional Equivariant GNN for soft tissue deformation & force estimation

Model code for:

> Kojanazarova, M., El Hadramy, S., Wilkie, J., Rauter, G., Cattin, P. C.
> **Soft Tissue Simulation and Force Estimation From Heterogeneous Structures
> Using Equivariant Graph Neural Networks.**
> *Healthcare Technology Letters* 12(1):e70042, 2025.
> https://doi.org/10.1049/htl2.70042 · https://arxiv.org/abs/2509.10125

## Based on

The message-passing scheme (`cEGCL`) extends the E(n)-Equivariant Graph
Neural Network of:

> Satorras, V. G., Hoogeboom, E., & Welling, M.
> **E(n) Equivariant Graph Neural Networks.** 2021.
> https://arxiv.org/pdf/2102.09844

Each `cEGCL` step is a standard EGNN message-passing update, with the
coordinate update additionally conditioned on two fixed 3D locations — the
start and end of the applied probe — so the same layer can be reused across
different contact locations without retraining.

## Requirements

```
torch>=2.0
torch_geometric>=2.3
torch_cluster>=1.6
```

## Structure

```
models/
  layers.py            cEGCL — one conditional, equivariant message-passing step
  displacement_net.py  DisplacementNet — 4 stacked cEGCL steps + regression head
  force_head.py        ForceHead — optional scalar contact-force regressor
  soft_tissue_net.py   SoftTissueNet — convenience wrapper combining the two
```

## Inputs and outputs

All three models are PyTorch Geometric–style: a batch of `B` samples, each
with `N` points, is passed as flat tensors of shape `[B*N, ...]` plus a
`batch` index tensor (the standard PyG convention), rather than a separate
leading batch dimension per point tensor.

**`DisplacementNet.forward(x, batch, cond, cond_feat, point_feat)`**

| name           | shape           | meaning                                                |
|----------------|-----------------|--------------------------------------------------------|
| `input_points` | `[B*N, 3]`      | point positions                                        |
| `batch`        | `[B*N]`         | batch index per point (PyG convention)                 |
| `cond`         | `[B, 6]`        | probe condition, `[start_xyz, end_xyz]`                |
| `cond_feat`    | `[B, 4, 128]`   | per-sample condition profile (tissue + xyz along probe)|
| `point_feat`   | `[B*N, 2, 128]` | per-point tissue profile features (tissue + z)         |

→ returns `displacement [B*N, 3]`, `latent [B*N, 24]` (the latent is exposed
for reuse, e.g. by a force head, and can be ignored otherwise).

**`ForceHead.forward(input_points, batch, displacement, cond, point_feat, cond_feat)`**

| name           | shape            | meaning                                   |
|----------------|------------------|-------------------------------------------|
| `input_points` | `[B*N, >=2]`     | point coordinates (only xy used)          |
| `batch`        | `[B*N]`          | batch index per point                     |
| `displacement` | `[B*N, 3]`       | predicted or ground-truth displacement    |
| `cond`         | `[B, 6]`         | probe condition, `[start_xyz, end_xyz]`   |
| `point_feat`   | `[B*N, 2, 128]`  | per-point tissue profile (channel 0 used) |
| `cond_feat`    | `[B, 4, 128]`    | per-sample probe profile                  |

→ returns `force [B]`, a scalar contact force per sample.

**`SoftTissueNet.forward(input_points, batch, cond, poke_feat, point_feat)`**
combines both — same inputs as above — and returns `(displacement, force)`,
with `force` being `None` if the model was built with `predict_force=False`.

## Usage

```python
from cegnn import SoftTissueNet

# Displacement + force, jointly:
model = SoftTissueNet(k=5, predict_force=True)
displacement, force = model(input_points, batch, cond, cond_feat, point_feat)

# Displacement only — no force branch is even created:
model = SoftTissueNet(k=5, predict_force=False)
displacement, _ = model(input_points, batch, cond, cond_feat, point_feat)

# Or use the two pieces independently, e.g. to train the force head
# separately (against a frozen or already-trained displacement net):
from cegnn import DisplacementNet, ForceHead

displacement_net = DisplacementNet(k=5)
force_head = ForceHead()

displacement = displacement_net(x, batch, cond, cond_feat, point_feat)
force = force_head(input_points, batch, displacement, cond, point_feat, cond_feat)
```

## Notes 

Displacement and force are architecturally decoupled on purpose: if you
don't need per-point force, `predict_force=False` skips that branch
entirely; if you do need it, it can be trained jointly or separately from
the displacement net.

`DisplacementNet`'s 4 `CEGCL` steps
share one set of weights by default (`share_weights=True`), matching the
model reported in the paper. Set `share_weights=False` for 4
independently-parameterized layers instead.
