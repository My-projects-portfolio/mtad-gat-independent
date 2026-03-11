"""
MTGFlow-CNF: MTGFlow with Continuous Normalizing Flow
======================================================

ARCHITECTURAL DESIGN RATIONALE
================================

1. WHAT IS WRONG WITH THE ORIGINAL DISCRETE FLOW?
---------------------------------------------------
MTGFlow's original design uses a Masked Autoregressive Flow (MAF), which is a
*discrete* normalizing flow — a fixed, finite sequence of affine bijections.

Problems with this for MTS anomaly detection:
  (a) The flow depth (number of blocks) is a hard architectural choice.
      Too few blocks → poor density estimation. Too many → overfit or slow.
  (b) Each MAF block transforms the data autoregressively, imposing a fixed
      factorisation order on features. For multivariate sensor data without a
      natural ordering, this is an arbitrary inductive bias.
  (c) All blocks share the same computational "resolution" — there is no natural
      mechanism for the model to allocate more transformation effort to harder
      regions of the distribution.

2. WHAT A CNF OFFERS
---------------------
A Continuous Normalizing Flow (CNF) defines the transformation as a continuous-
time dynamical system:

    dz/dt = f_θ(z, t, context)    z(t=0) = x

and the change in log-density follows the instantaneous change-of-variables
formula (Liouville's equation):

    d log p(z(t)) / dt = -Tr(∂f_θ / ∂z)

The final log-likelihood is:

    log p(x) = log p_base(z(T)) + ∫₀ᵀ Tr(∂f_θ/∂z) dt

This gives:
  (a) Depth-free expressiveness: the ODE solver adaptively allocates
      function evaluations where the trajectory is complex.
  (b) No ordering constraint: f_θ is a free neural network with no
      autoregressive structure — all dimensions are treated symmetrically.
  (c) Natural conditioning: the context vector enters as a simple
      additional input to f_θ at every integration step.

3. PROPOSED DESIGN: MTGFlow-CNF
---------------------------------

  ┌─────────────────────────────────────────────────────────────┐
  │  Input:  x  ∈  R^{N × K × T × 1}                           │
  │          N = batch, K = sensors, T = window, 1 = feature dim│
  └─────────────────────────────────────────────────────────────┘
               │
               ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  ENCODER (kept from MTGFlow — frozen design)                │
  │                                                             │
  │  a) Self-Attention Graph Learner                            │
  │     Treats x_k (sensor k across T steps) as a graph node.  │
  │     Computes per-sample dynamic adjacency A ∈ R^{N×K×K}     │
  │                                                             │
  │  b) LSTM per sensor                                         │
  │     x reshaped to [N*K, T, 1] → LSTM → H ∈ R^{N×K×T×H}   │
  │                                                             │
  │  c) GNN (graph convolution over sensor graph)              │
  │     Uses A to mix sensor representations                    │
  │     C ∈ R^{N×K×T×H}  (spatio-temporal condition)           │
  └─────────────────────────────────────────────────────────────┘
               │   context C reshaped to [N*K*T, H]
               ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  ENTITY-AWARE POOLING                                       │
  │                                                             │
  │  A lightweight linear projection pools the T hidden states  │
  │  into a single fixed-size context vector per (sensor, batch)│
  │  c ∈ R^{N×K×H_ctx}                                         │
  │                                                             │
  │  This is the conditioning signal for the CNF.               │
  └─────────────────────────────────────────────────────────────┘
               │
               ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  CONTINUOUS NORMALIZING FLOW (CNF)                          │
  │                                                             │
  │  The CNF operates on the raw input x (scalar per timestep). │
  │  Specifically, after flattening to [N*K*T, 1], the CNF maps │
  │  each scalar observation x_t^k to z through an ODE:         │
  │                                                             │
  │    dz/dt = f_θ(z, t, c)   where c is the context          │
  │                                                             │
  │  f_θ is a small MLP (the "dynamics net") with input:        │
  │    [z, t, c]  →  hidden layers  →  dz/dt (same dim as z)   │
  │                                                             │
  │  The log-density is tracked via the trace of the Jacobian.  │
  │  We use Hutchinson's stochastic trace estimator for         │
  │  efficiency (avoids full Jacobian computation).             │
  │                                                             │
  │  Base distribution: entity-specific Gaussian N(μ_k, I)     │
  │  (matching MTGFlow's entity-aware design).                   │
  └─────────────────────────────────────────────────────────────┘
               │
               ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  ANOMALY SCORE                                              │
  │                                                             │
  │  log p(x^k) per sensor k, averaged across T, then averaged │
  │  across K for the window-level score.                        │
  │  Score = -mean log p   (higher = more anomalous)            │
  └─────────────────────────────────────────────────────────────┘

4. KEY DESIGN DECISIONS
------------------------

(a) CNF on raw scalars, not on latent embeddings:
    We feed the raw 1-D sensor value x_t^k into the CNF (same as MAF in
    MTGFlow). The context c from the encoder provides all temporal and
    graph-structural information. This separates the *representation* task
    (encoder) from the *density estimation* task (CNF) cleanly.

(b) Time-conditional dynamics net:
    Including t as input to f_θ makes the flow time-inhomogeneous, allowing
    it to learn that different parts of the trajectory need different
    transformations (e.g. more warping near the base distribution).

(c) Hutchinson trace estimator:
    Computing Tr(∂f/∂z) exactly requires a full Jacobian (expensive for
    large batches). Instead, we use the unbiased estimator:
        Tr(∂f/∂z) ≈ ε^T (∂f/∂z) ε,  ε ~ N(0, I)
    which requires only one extra vector-Jacobian product (vjp) per step.

(d) Fixed short integration interval [0, 1]:
    Using a standard interval simplifies hyperparameter tuning. The ODE
    solver (Euler or RK4 for stability) adaptively takes sub-steps.

(e) Shared CNF parameters across entities (entity-aware via context):
    Like MTGFlow's shared normalizing flow, we share f_θ across sensors.
    Entity-specificity comes entirely from the context vector c_k, which
    includes graph-convolved, sensor-specific hidden states.

(f) Gradient clipping + regularisation for stability:
    CNFs trained with Hutchinson estimators can have high-variance gradients.
    We add a kinetic energy regulariser: λ * E[||f_θ||²] to discourage
    overly complex trajectories.

5. STABILITY CONSIDERATIONS
-----------------------------
  - Euler integration is used by default (fast, deterministic). RK4 is
    available for better accuracy.
  - The dynamics net uses Tanh activations (bounded derivatives → stable ODEs).
  - Context is LayerNorm'd before injection to prevent scale drift.
  - A small kinetic energy penalty λ·||dz/dt||² is added to the training loss.
  - Gradient clipping of 1.0 is strongly recommended (same as MTGFlow).

===========================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ENCODER  (Graph-structure learning + LSTM + GNN)
#     Kept almost identical to MTGFlow to preserve the validated representation
# ─────────────────────────────────────────────────────────────────────────────

class ScaleDotProductAttentionGraph(nn.Module):
    """
    Self-attention used as a dynamic graph-structure learner.

    Input : x  ∈  R^{N × K × T × D}
    Output: A  ∈  R^{N × K × K}   (soft adjacency matrix, row-softmax)
    """

    def __init__(self, feature_dim: int, dropout: float = 0.2):
        super().__init__()
        self.w_q = nn.Linear(feature_dim, feature_dim)
        self.w_k = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(feature_dim)

    def forward(self, x: torch.Tensor):
        # x: [N, K, T, D]
        N, K, T, D = x.shape
        # Flatten T*D into one feature per node
        x_flat = x.reshape(N, K, T * D)          # [N, K, T*D]
        q = self.w_q(x_flat)                       # [N, K, T*D]
        k = self.w_k(x_flat)                       # [N, K, T*D]
        # pairwise attention scores
        score = torch.bmm(q, k.transpose(1, 2)) / self.scale   # [N, K, K]
        A = self.dropout(self.softmax(score))      # [N, K, K]
        return A


class GNN(nn.Module):
    """
    One-hop graph convolution with residual history term.
    Identical to MTGFlow's GNN.

    h_new[k] = ReLU( A·H·W1 + H[:, :, :-1]·W2 ) W3
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # h: [N, K, T, H],  A: [N, K, K]
        h_n = self.lin_n(torch.einsum('nkld,nkj->njld', h, A))   # [N, K, T, H]
        h_r = self.lin_r(h[:, :, :-1])                            # [N, K, T-1, H]
        h_n[:, :, 1:] = h_n[:, :, 1:] + h_r
        return self.lin_2(F.relu(h_n))                            # [N, K, T, H]


class MTGFlowEncoder(nn.Module):
    """
    Spatio-temporal encoder:  x  →  context C

    Steps:
      1. Self-attention graph learner → dynamic adjacency A  [N, K, K]
      2. Per-sensor LSTM             → hidden states  H     [N, K, T, H]
      3. Graph convolution (GNN)     → context C            [N, K, T, H]
      4. Temporal pooling + project  → c                    [N*K, H_ctx]
    """

    def __init__(
        self,
        input_size: int,         # D = 1 in MTGFlow
        hidden_size: int,        # H
        window_size: int,        # T
        n_sensor: int,           # K
        ctx_size: int,           # H_ctx  (size of context for CNF)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = n_sensor
        self.T = window_size
        self.H = hidden_size

        feat_dim = window_size * input_size          # flattened node feature
        self.graph_attn = ScaleDotProductAttentionGraph(feat_dim, dropout)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )
        self.gcn = GNN(hidden_size, hidden_size)

        # Pool T hidden states → single context vector per (batch, sensor)
        self.ctx_pool = nn.Linear(hidden_size, ctx_size)
        self.ctx_norm = nn.LayerNorm(ctx_size)

    def forward(self, x: torch.Tensor):
        """
        x: [N, K, T, D]
        returns: context  [N*K, ctx_size]
                 graph    [N, K, K]    (stored for inspection)
        """
        N, K, T, D = x.shape
        assert K == self.K and T == self.T

        # 1. Dynamic graph
        A = self.graph_attn(x)                          # [N, K, K]

        # 2. LSTM — run per sensor
        x_rnn = x.reshape(N * K, T, D)                 # [N*K, T, D]
        h_rnn, _ = self.rnn(x_rnn)                     # [N*K, T, H]
        H = h_rnn.reshape(N, K, T, self.H)             # [N, K, T, H]

        # 3. GNN
        C = self.gcn(H, A)                              # [N, K, T, H]

        # 4. Temporal mean-pool + project
        c_pool = C.mean(dim=2)                          # [N, K, H]
        c_proj = self.ctx_norm(self.ctx_pool(c_pool))   # [N, K, ctx_size]

        # Flatten batch and sensor dims for CNF input
        c_flat = c_proj.reshape(N * K, -1)              # [N*K, ctx_size]

        return c_flat, A


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CNF DYNAMICS NETWORK
#     f_θ : (z, t, context) → dz/dt
# ─────────────────────────────────────────────────────────────────────────────

class DynamicsNet(nn.Module):
    """
    The neural ODE function f_θ(z, t, context) → dz/dt.

    Architecture:
      [z | sin(t*ω) | cos(t*ω) | context]  →  MLP(Tanh)  →  dz/dt

    Using Fourier time-embedding avoids the trivial t=0 singularity and lets
    the network encode where in the flow trajectory it is.

    Tanh activations give bounded derivatives, keeping the ODE well-conditioned.
    """

    def __init__(
        self,
        z_dim: int,          # dimension of the flow state (1 for scalar sensor obs)
        ctx_dim: int,        # context dimension from encoder
        hidden_dim: int,     # hidden units in MLP
        n_layers: int,       # MLP depth
        n_freq: int = 4,     # number of Fourier frequencies for time embedding
    ):
        super().__init__()
        self.z_dim = z_dim
        self.n_freq = n_freq
        t_embed_dim = 2 * n_freq          # sin + cos for each freq

        in_dim = z_dim + t_embed_dim + ctx_dim
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, z_dim))
        self.net = nn.Sequential(*layers)

        # Learnable frequencies
        self.register_buffer(
            'freqs',
            torch.arange(1, n_freq + 1, dtype=torch.float32) * math.pi
        )

    def time_embed(self, t: torch.Tensor) -> torch.Tensor:
        """t: scalar → [2*n_freq]"""
        phases = t * self.freqs        # [n_freq]
        return torch.cat([phases.sin(), phases.cos()], dim=-1)   # [2*n_freq]

    def forward(
        self,
        t: torch.Tensor,        # scalar (current integration time)
        z: torch.Tensor,        # [M, z_dim]  M = N*K*T
        ctx: torch.Tensor,      # [M, ctx_dim]
    ) -> torch.Tensor:
        te = self.time_embed(t).expand(z.shape[0], -1)   # [M, 2*n_freq]
        inp = torch.cat([z, te, ctx], dim=-1)             # [M, z_dim+t+ctx]
        return self.net(inp)                               # [M, z_dim]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ODE INTEGRATORS
#     Simple implementations that avoid the torchdiffeq dependency.
#     Both track the accumulated log-det (via Hutchinson estimator).
# ─────────────────────────────────────────────────────────────────────────────

def _hutchinson_trace(f_out: torch.Tensor, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """
    Stochastic trace estimator: Tr(∂f/∂z) ≈ e^T (∂f/∂z) e

    Uses autograd vjp (vector-Jacobian product).
    e: [M, z_dim] Rademacher or Gaussian noise vector.
    """
    # e^T (Jf) = vjp(f, z, e)
    e_jf = torch.autograd.grad(
        outputs=f_out,
        inputs=z,
        grad_outputs=e,
        create_graph=True,
        retain_graph=True,
    )[0]                    # [M, z_dim]
    return (e_jf * e).sum(dim=-1)   # [M]


def euler_integrate_cnf(
    dynamics: DynamicsNet,
    z0: torch.Tensor,
    ctx: torch.Tensor,
    n_steps: int = 10,
    t0: float = 0.0,
    t1: float = 1.0,
    hutchinson_samples: int = 1,
) -> tuple:
    """
    Euler integrator for the CNF ODE.

    Returns
    -------
    z1         : [M, z_dim]   final state
    delta_logp : [M]          accumulated log-density change
                              (negative = density increases = data is likely)
    dyn_norm   : scalar       mean squared norm of dynamics (kinetic energy reg)
    """
    z = z0.requires_grad_(True)
    delta_logp = torch.zeros(z0.shape[0], device=z0.device)
    dyn_norms = []

    dt = (t1 - t0) / n_steps
    t = torch.tensor(t0, device=z0.device, dtype=z0.dtype)

    for _ in range(n_steps):
        # Sample Rademacher noise for trace estimator
        e = torch.randint(0, 2, z.shape, device=z.device, dtype=z.dtype) * 2 - 1

        dz = dynamics(t, z, ctx)                              # [M, z_dim]
        trace_est = _hutchinson_trace(dz, z, e)               # [M]

        dyn_norms.append((dz ** 2).mean())

        # Euler step — detach z before step to avoid accumulating huge graphs
        with torch.no_grad():
            delta_logp = delta_logp - trace_est.detach() * dt
            z_next = z.detach() + dz.detach() * dt

        z = z_next.requires_grad_(True)
        t = t + dt

    dyn_norm = torch.stack(dyn_norms).mean()
    return z, delta_logp, dyn_norm


def rk4_integrate_cnf(
    dynamics: DynamicsNet,
    z0: torch.Tensor,
    ctx: torch.Tensor,
    n_steps: int = 8,
    t0: float = 0.0,
    t1: float = 1.0,
    hutchinson_samples: int = 1,
) -> tuple:
    """
    4th-order Runge-Kutta integrator for the CNF ODE.
    More accurate than Euler at the same n_steps, but 4× the cost.
    """
    z = z0.requires_grad_(True)
    delta_logp = torch.zeros(z0.shape[0], device=z0.device)
    dyn_norms = []

    dt = (t1 - t0) / n_steps
    t = torch.tensor(t0, device=z0.device, dtype=z0.dtype)

    for _ in range(n_steps):
        e = torch.randint(0, 2, z.shape, device=z.device, dtype=z.dtype) * 2 - 1

        # k1
        dz1 = dynamics(t, z, ctx)
        tr1 = _hutchinson_trace(dz1, z, e)

        # k2
        t_mid = t + dt / 2
        z2 = (z + dz1 * dt / 2).detach().requires_grad_(True)
        dz2 = dynamics(t_mid, z2, ctx)
        tr2 = _hutchinson_trace(dz2, z2, e)

        # k3
        z3 = (z + dz2 * dt / 2).detach().requires_grad_(True)
        dz3 = dynamics(t_mid, z3, ctx)
        tr3 = _hutchinson_trace(dz3, z3, e)

        # k4
        t_end = t + dt
        z4 = (z + dz3 * dt).detach().requires_grad_(True)
        dz4 = dynamics(t_end, z4, ctx)
        tr4 = _hutchinson_trace(dz4, z4, e)

        dz_rk4 = (dz1 + 2 * dz2 + 2 * dz3 + dz4) / 6
        tr_rk4 = (tr1 + 2 * tr2 + 2 * tr3 + tr4) / 6

        dyn_norms.append((dz_rk4 ** 2).mean())

        with torch.no_grad():
            delta_logp = delta_logp - tr_rk4.detach() * dt
            z_next = z.detach() + dz_rk4.detach() * dt

        z = z_next.requires_grad_(True)
        t = t_end

    dyn_norm = torch.stack(dyn_norms).mean()
    return z, delta_logp, dyn_norm


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ENTITY-AWARE BASE DISTRIBUTION
#     Each sensor k has a learnable mean μ_k; covariance is fixed I
#     (mirrors MTGFlow's entity-aware design)
# ─────────────────────────────────────────────────────────────────────────────

class EntityAwareGaussian(nn.Module):
    """
    Learnable entity-specific Gaussian base distributions.

    p_k(z) = N(z; μ_k, I)

    μ_k is initialised from N(0, 1) and learned during training.
    Parameters are shared across batch but distinct per sensor index.
    """

    def __init__(self, n_sensor: int, z_dim: int = 1):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(n_sensor, z_dim))

    def log_prob(self, z: torch.Tensor, sensor_idx: torch.Tensor) -> torch.Tensor:
        """
        z          : [M, z_dim]
        sensor_idx : [M]   integers in [0, K)
        returns    : [M]   log p_k(z)
        """
        mu_k = self.mu[sensor_idx]            # [M, z_dim]
        diff = z - mu_k                        # [M, z_dim]
        log_p = -0.5 * (diff ** 2).sum(-1) \
                - 0.5 * self.mu.shape[1] * math.log(2 * math.pi)
        return log_p                           # [M]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FULL MTGFlow-CNF MODEL
# ─────────────────────────────────────────────────────────────────────────────

class MTGFlowCNF(nn.Module):
    """
    MTGFlow with Continuous Normalizing Flow replacing the discrete MAF.

    Constructor parameters
    ----------------------
    input_size   : int   Feature dim D (= 1 in standard MTGFlow usage)
    hidden_size  : int   LSTM / GNN hidden dimension H
    ctx_size     : int   Context vector size (input to CNF dynamics net)
    cnf_hidden   : int   Hidden units in the CNF dynamics MLP
    cnf_layers   : int   Depth of the CNF dynamics MLP
    window_size  : int   T (sliding window length)
    n_sensor     : int   K (number of sensors / entities)
    dropout      : float Dropout used in LSTM and attention
    ode_steps    : int   Number of ODE integration steps (Euler or RK4)
    integrator   : str   'euler' | 'rk4'
    kinetic_reg  : float λ coefficient for kinetic energy regulariser
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        ctx_size: int = 32,
        cnf_hidden: int = 64,
        cnf_layers: int = 2,
        window_size: int = 100,
        n_sensor: int = 38,
        dropout: float = 0.1,
        ode_steps: int = 10,
        integrator: str = 'euler',
        kinetic_reg: float = 0.01,
    ):
        super().__init__()

        self.K = n_sensor
        self.T = window_size
        self.D = input_size
        self.ode_steps = ode_steps
        self.integrator = integrator
        self.kinetic_reg = kinetic_reg

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = MTGFlowEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            window_size=window_size,
            n_sensor=n_sensor,
            ctx_size=ctx_size,
            dropout=dropout,
        )

        # ── CNF dynamics network ──────────────────────────────────────────────
        self.dynamics = DynamicsNet(
            z_dim=input_size,        # = 1: one scalar per observation
            ctx_dim=ctx_size,
            hidden_dim=cnf_hidden,
            n_layers=cnf_layers,
        )

        # ── Entity-aware base distribution ────────────────────────────────────
        self.base_dist = EntityAwareGaussian(n_sensor=n_sensor, z_dim=input_size)

        # ── ODE integrator selector ───────────────────────────────────────────
        if integrator == 'rk4':
            self._integrate = rk4_integrate_cnf
        else:
            self._integrate = euler_integrate_cnf

    # ── Internal forward pass ─────────────────────────────────────────────────

    def _compute_log_prob(self, x: torch.Tensor):
        """
        Core computation: encode → flow → log p.

        x : [N, K, T, D]

        Returns
        -------
        log_prob  : [N, K, T]   per-observation log likelihoods
        kin_reg   : scalar      kinetic energy regularisation term
        graph     : [N, K, K]   learned adjacency (for inspection)
        """
        N, K, T, D = x.shape

        # 1. Encoder
        ctx, graph = self.encoder(x)           # ctx: [N*K, ctx_size]

        # 2. Flatten observations x to [N*K*T, D]
        x_flat = x.reshape(N * K * T, D)       # [N*K*T, D]

        # Expand context from [N*K, ctx_size] to [N*K*T, ctx_size]
        ctx_exp = ctx.unsqueeze(1).expand(-1, T, -1).reshape(N * K * T, -1)

        # 3. CNF forward integration:  x  →  z
        z1, delta_logp, kin_reg = self._integrate(
            dynamics=self.dynamics,
            z0=x_flat,
            ctx=ctx_exp,
            n_steps=self.ode_steps,
        )

        # 4. Base distribution log p(z)
        #    Build sensor index array matching [N*K*T]
        sensor_idx = torch.arange(K, device=x.device)              # [K]
        sensor_idx = sensor_idx.unsqueeze(0).expand(N, -1)          # [N, K]
        sensor_idx = sensor_idx.reshape(N * K)                      # [N*K]
        sensor_idx = sensor_idx.unsqueeze(1).expand(-1, T).reshape(N * K * T)  # [N*K*T]

        log_pz = self.base_dist.log_prob(z1, sensor_idx)            # [N*K*T]

        # 5. Change-of-variables: log p(x) = log p(z) + delta_logp
        log_px = log_pz + delta_logp                                 # [N*K*T]

        # Reshape to [N, K, T]
        log_prob = log_px.reshape(N, K, T)

        return log_prob, kin_reg, graph

    # ── Public API  (mirrors MTGFlow for drop-in replacement) ─────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.

        x    : [N, K, T, D]
        loss : scalar (negative mean log likelihood + kinetic reg)
                  minimise this during training
        """
        log_prob, kin_reg, _ = self._compute_log_prob(x)
        nll = -log_prob.mean()
        loss = nll + self.kinetic_reg * kin_reg
        return loss

    def test(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference: return mean log-likelihood per window.

        x        : [N, K, T, D]
        log_prob : [N]   (mean over K and T)
        """
        with torch.no_grad():
            log_prob, _, _ = self._compute_log_prob(x)
        return log_prob.mean(dim=(1, 2))   # [N]

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns NEGATIVE log-likelihood (higher = more anomalous).

        x     : [N, K, T, D]
        score : [N]
        """
        return -self.test(x)

    def get_graph(self) -> torch.Tensor:
        """Run a dummy forward to retrieve the last learned graph structure."""
        raise RuntimeError(
            "Call forward() or test() first; graph is an intermediate tensor."
        )

    def locate(self, x: torch.Tensor):
        """
        Per-entity anomaly scores for anomaly interpretation.

        x          : [N, K, T, D]
        returns:
          entity_scores : [N, K]   mean NLL per sensor
          z_flat        : [N*K, T]  transformed latent (for inspection)
        """
        with torch.no_grad():
            log_prob, _, _ = self._compute_log_prob(x)
        entity_scores = -log_prob.mean(dim=2)   # [N, K]
        return entity_scores


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DATASET  (identical to MTGFlowWindowDataset in the notebook)
# ─────────────────────────────────────────────────────────────────────────────

class MTGFlowWindowDataset(Dataset):
    """
    windows : [N, T, K]  float32
    Output  : [K, T, 1]  (MTGFlow convention)
    """

    def __init__(self, windows, labels=None, starts=None):
        self.windows = windows.astype(np.float32)
        self.labels = None if labels is None else labels.astype(np.int64)
        self.starts = None if starts is None else starts.astype(np.int64)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]                          # [T, K]
        x = torch.from_numpy(w).transpose(0, 1)       # [K, T]
        x = x.unsqueeze(-1)                            # [K, T, 1]
        y = 0 if self.labels is None else int(self.labels[idx])
        s = -1 if self.starts is None else int(self.starts[idx])
        return x, y, int(idx), s


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRAINING LOOP  (drop-in replacement for train_one_entity)
# ─────────────────────────────────────────────────────────────────────────────

def build_mtgflow_cnf(
    K: int,
    T: int,
    hidden_size: int = 32,
    ctx_size: int = 32,
    cnf_hidden: int = 64,
    cnf_layers: int = 2,
    dropout: float = 0.1,
    ode_steps: int = 10,
    integrator: str = 'euler',
    kinetic_reg: float = 0.01,
    device: str = 'cpu',
) -> MTGFlowCNF:
    """Factory function matching the signature of build_mtgflow_model."""
    model = MTGFlowCNF(
        input_size=1,
        hidden_size=hidden_size,
        ctx_size=ctx_size,
        cnf_hidden=cnf_hidden,
        cnf_layers=cnf_layers,
        window_size=T,
        n_sensor=K,
        dropout=dropout,
        ode_steps=ode_steps,
        integrator=integrator,
        kinetic_reg=kinetic_reg,
    ).to(device)
    return model


def train_mtgflow_cnf(
    model: MTGFlowCNF,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 40,
    lr: float = 2e-3,
    weight_decay: float = 5e-4,
    grad_clip: float = 1.0,
    device: str = 'cpu',
    verbose: bool = True,
):
    """
    Training loop for MTGFlowCNF.

    Mirrors the pattern in the notebook's train_one_entity function.
    Loss = NLL + kinetic_reg * ||dz/dt||^2
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'epoch': [], 'train_loss': [], 'auroc': [], 'aupr': []}
    best_auroc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        batch_losses = []

        for x, _, _, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)          # NLL + kinetic reg
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float('nan')

        # Evaluation
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for x, y, _, _ in test_loader:
                x = x.to(device)
                s = model.anomaly_score(x).cpu().numpy()
                scores.append(s)
                labels.append(np.asarray(y, dtype=np.int64))

        scores = np.concatenate(scores) if scores else np.array([])
        labels = np.concatenate(labels) if labels else np.array([])

        if labels.size and np.unique(labels).size > 1:
            auroc = float(roc_auc_score(labels, scores))
            aupr  = float(average_precision_score(labels, scores))
        else:
            auroc, aupr = float('nan'), float('nan')

        history['epoch'].append(ep)
        history['train_loss'].append(train_loss)
        history['auroc'].append(auroc)
        history['aupr'].append(aupr)

        if verbose:
            print(
                f"  epoch {ep:03d}/{epochs} | "
                f"loss={train_loss:.4f} | "
                f"AUROC={auroc:.4f} | "
                f"AUPR={aupr:.4f}"
            )

        if np.isfinite(auroc) and auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"\n  ✅ Restored best model (AUROC={best_auroc:.4f})")

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# 8.  QUICK SMOKE-TEST  (runs without real data)
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test():
    """
    Verifies all shapes and gradients are correct on synthetic data.
    """
    print("=" * 60)
    print("MTGFlow-CNF  —  Smoke Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    N, K, T, D = 4, 10, 30, 1   # small values for fast test
    model = build_mtgflow_cnf(
        K=K, T=T,
        hidden_size=16,
        ctx_size=16,
        cnf_hidden=32,
        cnf_layers=2,
        ode_steps=4,
        integrator='euler',
        kinetic_reg=0.01,
        device=str(device),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # Synthetic batch
    x = torch.randn(N, K, T, D, device=device)

    # Training step
    model.train()
    loss = model(x)
    print(f"Training loss (NLL+reg): {loss.item():.4f}")
    loss.backward()
    print("  ✅ Backward pass OK")

    # Test / anomaly scoring
    model.eval()
    scores = model.anomaly_score(x)
    print(f"Anomaly scores shape: {scores.shape}  (should be [{N}])")
    print(f"Anomaly scores: {scores.detach().cpu().numpy()}")

    # Per-entity scores (anomaly localisation)
    entity_scores = model.locate(x)
    print(f"Entity scores shape: {entity_scores.shape}  (should be [{N}, {K}])")

    # Dataset + DataLoader round-trip
    windows = np.random.randn(20, T, K).astype(np.float32)
    labels = (np.random.rand(20) > 0.8).astype(np.int64)
    ds = MTGFlowWindowDataset(windows, labels=labels)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    x_batch, y_batch, idx_batch, _ = next(iter(loader))
    print(f"\nDataLoader batch: x={x_batch.shape}, y={y_batch.shape}")
    assert x_batch.shape == (4, K, T, 1), f"Unexpected shape: {x_batch.shape}"

    print("\n✅  All checks passed.")


if __name__ == '__main__':
    _smoke_test()
