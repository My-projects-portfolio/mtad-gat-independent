"""
MTGFlow-CNF: Continuous Normalizing Flow extension of MTGFlow
=============================================================

Design overview
---------------
The original MTGFlow pipeline is:
  x [N,K,L,D] → Attention (graph) → LSTM (temporal) → GNN (spatial) → h [N,K,L,H]
                                                                         ↓
                                                               MAF log_prob(x | h)

This redesign replaces the MAF with a latent-space Conditional CNF:

  x [N,K,L,D] → Attention → LSTM → GNN → h [N,K,L,H]
                                          │
                                          ├─► context c [N,H]  (global window summary)
                                          │
                                          ▼
                               CNF: model log p(h | c)
                               ODE: dz/dt = f(z, t, c)  using FiLM conditioning
                               Trace: Hutchinson estimator  (O(H) cost)
                               Solver: dopri5 adaptive (torchdiffeq)
                                          │
                                          ▼
                               anomaly score = -log p(h)

Key design decisions
--------------------
1. The CNF operates on latent GNN embeddings h ∈ ℝᴴ, not on raw scalar
   observations. This is more expressive and avoids the D=1 factorisation.

2. Context c is a mean-pooled window summary injected via FiLM (Feature-wise
   Linear Modulation) at every layer of the ODE network. This is more powerful
   than simple concatenation because c can modulate the entire dynamics.

3. Spectral normalisation on all ODE MLP layers bounds the Lipschitz constant,
   keeping the ODE non-stiff and gradients well-behaved.

4. Hutchinson trace estimator reduces log-det cost from O(H²) to O(H) per step.

5. An optional kinetic-energy regulariser ∫‖f‖² dt encourages smooth transport.

Dependencies
------------
  pip install torchdiffeq

Usage
-----
  model = MTGFlowCNF(
      n_sensor=38, input_size=1, hidden_size=64,
      window_size=100, ode_hidden=128, ode_layers=3,
  )
  log_prob = model.test(x)       # [N]  – use .mean() as training loss
  score    = -model.test(x)      # [N]  – anomaly score (higher = more anomalous)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ---------------------------------------------------------------------------
# Optional: torchdiffeq (required at runtime; import gracefully)
# ---------------------------------------------------------------------------
try:
    from torchdiffeq import odeint, odeint_adjoint
    _TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    _TORCHDIFFEQ_AVAILABLE = False
    # Provide a minimal Euler fallback so the module can still be imported
    # and tested structurally without torchdiffeq installed.
    def odeint(func, y0, t, **kwargs):
        """Minimal fixed-step Euler solver fallback."""
        dt = t[1] - t[0]
        y = y0
        for i in range(len(t) - 1):
            y = y + dt * func(t[i], y)
        return torch.stack([y0, y])

    odeint_adjoint = odeint


# ===========================================================================
# 1. Unchanged MTGFlow feature-extraction components
# ===========================================================================

class ScaleDotProductAttention(nn.Module):
    """
    Sensor-graph builder.
    Input  x: [N, K, L, D]
    Output score: [N, K, K]  – soft adjacency matrix per sample
    """

    def __init__(self, c: int):
        super().__init__()
        self.w_q = nn.Linear(c, c)
        self.w_k = nn.Linear(c, c)
        self.w_v = nn.Linear(c, c)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask=None, e=1e-12):
        shape = x.shape                          # [N, K, L, D]
        x_flat = x.reshape(shape[0], shape[1], -1)   # [N, K, L*D]
        N, K, c = x_flat.size()
        q = self.w_q(x_flat)
        k = self.w_k(x_flat)
        k_t = k.view(N, c, K)
        score = (q @ k_t) / math.sqrt(c)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = self.dropout(self.softmax(score))
        return score, k


class GNN(nn.Module):
    """
    Single-layer graph neural network.
    h: [N, K, L, H]
    A: [N, K, K]
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        h_n = self.lin_n(torch.einsum('nkld,nkj->njld', h, A))
        h_r = self.lin_r(h[:, :, :-1])
        h_n[:, :, 1:] += h_r
        h = self.lin_2(F.relu(h_n))
        return h


# ===========================================================================
# 2. Context aggregator  (new)
# ===========================================================================

class ContextAggregator(nn.Module):
    """
    Aggregates h [N, K, L, H] into a single global context vector c [N, H].

    Strategy: mean-pool over K and L, then pass through a small MLP to allow
    non-linear summarisation.  The MLP has LayerNorm for training stability.
    """

    def __init__(self, hidden_size: int, context_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, context_size),
            nn.LayerNorm(context_size),
            nn.GELU(),
            nn.Linear(context_size, context_size),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [N, K, L, H]
        c = h.mean(dim=(1, 2))          # [N, H]
        c = self.mlp(c)                 # [N, context_size]
        return c


# ===========================================================================
# 3. CNF components  (new)
# ===========================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps scalar t ∈ [0,1] to a vector embedding.
    Uses sinusoidal features (like positional encoding) followed by a small MLP.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim
        half = embed_dim // 2
        # Fixed frequency bands
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) *
                          -(math.log(10000.0) / (half - 1)))
        self.register_buffer('freqs', freqs)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: scalar or [M] tensor
        t = t.reshape(-1) if t.dim() > 0 else t.unsqueeze(0)  # [M]
        args = t.unsqueeze(1) * self.freqs.unsqueeze(0)        # [M, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [M, embed_dim]
        return self.proj(emb)                                         # [M, embed_dim]


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Given a condition vector c, produce scale γ and bias β to modulate h:
        h_out = γ(c) ⊙ h + β(c)
    """

    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, feature_dim)
        self.beta_proj  = nn.Linear(cond_dim, feature_dim)
        # Initialise to identity transform
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # h: [M, feature_dim], c: [M, cond_dim]
        gamma = self.gamma_proj(c)   # [M, feature_dim]
        beta  = self.beta_proj(c)    # [M, feature_dim]
        return gamma * h + beta


class ODEFunc(nn.Module):
    """
    The drift function f(z, t, c) for the CNF ODE:
        dz/dt = f(z, t, c)

    Architecture:
        z → Linear → FiLM(c+t_emb) → tanh → Linear → FiLM → tanh → ... → z_out

    All linear layers have spectral normalisation to bound the Lipschitz
    constant, keeping the ODE non-stiff and gradients well-behaved.

    Parameters
    ----------
    latent_dim   : dimension of z (= hidden_size of GNN)
    hidden_dim   : width of the ODE MLP
    n_layers     : number of hidden layers (>= 2 recommended)
    context_dim  : dimension of the conditioning context c
    time_embed_dim: dimension of the sinusoidal time embedding
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int,
        context_dim: int,
        time_embed_dim: int = 32,
    ):
        super().__init__()
        assert n_layers >= 2, "n_layers must be >= 2"

        self.latent_dim = latent_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        cond_dim = context_dim + time_embed_dim   # combined conditioning signal

        # Build layers with spectral norm for Lipschitz control
        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [latent_dim] + [hidden_dim] * (n_layers - 1) + [latent_dim]
        for i in range(len(dims) - 1):
            lin = spectral_norm(nn.Linear(dims[i], dims[i + 1]))
            self.layers.append(lin)
            self.film_layers.append(FiLMLayer(dims[i + 1], cond_dim))
            self.norms.append(nn.LayerNorm(dims[i + 1]))

        # Track kinetic energy for optional regularisation
        self._kinetic_energy = 0.0
        self._n_evals = 0

    def forward(self, t: torch.Tensor, state: tuple) -> tuple:
        """
        Called by odeint as f(t, state).
        state = (z, log_p, [kinetic])
            z     : [M, latent_dim]
            log_p : [M, 1]           running log-density change
        Returns (dz/dt, d_log_p/dt, [d_kinetic/dt])
        """
        z, log_p = state[0], state[1]
        t_emb = self.time_embedding(t)               # [1, time_embed_dim]
        t_emb = t_emb.expand(z.shape[0], -1)         # [M, time_embed_dim]

        # Context c is stored externally (set before each odeint call)
        c = self._context                            # [M, context_dim]
        cond = torch.cat([c, t_emb], dim=-1)         # [M, cond_dim]

        # Compute dz/dt
        h = z
        for i, (lin, film, norm) in enumerate(
            zip(self.layers, self.film_layers, self.norms)
        ):
            h = lin(h)
            h = film(h, cond)
            h = norm(h)
            if i < len(self.layers) - 1:
                h = torch.tanh(h)   # tanh: smooth + bounded, good for ODEs
        dz = h  # [M, latent_dim]

        # Estimate d log p / dt = -Tr(∂f/∂z) via Hutchinson estimator
        # Sample ε ~ N(0, I), then Tr(J) ≈ εᵀ (J ε)
        with torch.enable_grad():
            z_req = z.requires_grad_(True)
            # Re-run forward with grad tracking for the trace
            h_g = z_req
            for i, (lin, film, norm) in enumerate(
                zip(self.layers, self.film_layers, self.norms)
            ):
                h_g = lin(h_g)
                h_g = film(h_g, cond)
                h_g = norm(h_g)
                if i < len(self.layers) - 1:
                    h_g = torch.tanh(h_g)
            eps = torch.randn_like(z_req)
            # Jacobian-vector product
            vjp = torch.autograd.grad(
                h_g, z_req,
                grad_outputs=eps,
                create_graph=self.training,
                retain_graph=True,
            )[0]
        hutchinson_trace = (vjp * eps).sum(dim=-1, keepdim=True)  # [M, 1]
        d_log_p = -hutchinson_trace                               # [M, 1]

        # Kinetic energy (for regularisation)
        kinetic = dz.pow(2).sum(dim=-1, keepdim=True)             # [M, 1]

        return (dz, d_log_p, kinetic)

    def set_context(self, c: torch.Tensor):
        """Must be called before each odeint invocation."""
        self._context = c


class LatentCNF(nn.Module):
    """
    Conditional Continuous Normalizing Flow operating in the GNN latent space.

    Given:
        z   : [M, latent_dim]   – GNN embeddings to score
        c   : [M, context_dim]  – per-sample conditioning context

    Returns:
        log_prob : [M]           – log p(z | c) under the CNF

    The flow integrates from t=1 → t=0 (data → noise direction) for
    log-likelihood evaluation, which is the standard CNF convention.

    Parameters
    ----------
    latent_dim      : dimension of the latent space (= GNN hidden_size)
    context_dim     : dimension of conditioning context
    ode_hidden_dim  : width of the ODE MLP
    ode_layers      : depth of the ODE MLP
    time_embed_dim  : sinusoidal time embedding dimension
    solver          : ODE solver name ('dopri5', 'rk4', 'euler')
    rtol, atol      : solver tolerances
    adjoint         : whether to use adjoint method for memory-efficient backprop
    kinetic_weight  : coefficient for kinetic energy regularisation (0 = off)
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        ode_hidden_dim: int = 128,
        ode_layers: int = 3,
        time_embed_dim: int = 32,
        solver: str = 'dopri5',
        rtol: float = 1e-4,
        atol: float = 1e-5,
        adjoint: bool = False,
        kinetic_weight: float = 1e-3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint and _TORCHDIFFEQ_AVAILABLE
        self.kinetic_weight = kinetic_weight

        self.ode_func = ODEFunc(
            latent_dim=latent_dim,
            hidden_dim=ode_hidden_dim,
            n_layers=ode_layers,
            context_dim=context_dim,
            time_embed_dim=time_embed_dim,
        )

        # Integration time: t=0 (data) → t=1 (noise) for density estimation
        self.register_buffer('int_time', torch.tensor([0.0, 1.0]))

    def forward(
        self,
        z: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log p(z | c) by integrating the ODE forward (data→noise).

        Returns
        -------
        log_prob   : [M]    log-probability of each z under the CNF
        reg_term   : scalar kinetic-energy regularisation loss
        """
        M = z.shape[0]
        log_p0 = torch.zeros(M, 1, device=z.device, dtype=z.dtype)
        ke0    = torch.zeros(M, 1, device=z.device, dtype=z.dtype)

        self.ode_func.set_context(c)

        state0 = (z, log_p0, ke0)
        solver_fn = odeint_adjoint if self.adjoint else odeint

        options = {}
        if self.solver in ('euler', 'rk4'):
            options['step_size'] = 0.1   # fixed step for simple solvers

        sol = solver_fn(
            self.ode_func,
            state0,
            self.int_time,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            options=options if options else None,
        )
        # sol is a tuple of tensors each shaped [2, M, ...]
        z_T     = sol[0][-1]   # [M, latent_dim]  – mapped noise
        delta_log_p = sol[1][-1]   # [M, 1]
        kinetic     = sol[2][-1]   # [M, 1]

        # Log-probability under base distribution N(0,I)
        log_base = -0.5 * (
            z_T.pow(2).sum(dim=-1) +
            self.latent_dim * math.log(2 * math.pi)
        )  # [M]

        # Change-of-variables: log p_data(z) = log p_base(z_T) + Δlog_p
        log_prob = log_base + delta_log_p.squeeze(-1)   # [M]

        # Kinetic energy regulariser
        reg_term = self.kinetic_weight * kinetic.mean() if self.training else torch.tensor(0.0)

        return log_prob, reg_term


# ===========================================================================
# 4. MTGFlowCNF — the full model
# ===========================================================================

class MTGFlowCNF(nn.Module):
    """
    MTGFlow with a Continuous Normalizing Flow replacing the MAF.

    The CNF operates in the GNN latent space (not raw sensor space), making it
    sensitive to distributional shifts in the learned representation rather than
    individual scalar values.

    Parameters
    ----------
    n_sensor        : number of sensors K
    input_size      : raw feature dimension D (typically 1)
    hidden_size     : LSTM and GNN hidden dimension H
    window_size     : temporal window length L
    context_size    : dimension of the global context vector (default = hidden_size)
    ode_hidden_dim  : width of the ODE network
    ode_layers      : depth of the ODE network
    time_embed_dim  : sinusoidal time embedding size
    dropout         : LSTM dropout
    solver          : ODE solver ('dopri5', 'rk4', 'euler')
    rtol, atol      : ODE solver tolerances
    adjoint         : use adjoint method (memory-efficient, slower compile)
    kinetic_weight  : kinetic energy regularisation coefficient
    """

    def __init__(
        self,
        n_sensor: int,
        input_size: int = 1,
        hidden_size: int = 64,
        window_size: int = 100,
        context_size: int = None,
        ode_hidden_dim: int = 128,
        ode_layers: int = 3,
        time_embed_dim: int = 32,
        dropout: float = 0.1,
        solver: str = 'dopri5',
        rtol: float = 1e-4,
        atol: float = 1e-5,
        adjoint: bool = False,
        kinetic_weight: float = 1e-3,
    ):
        super().__init__()

        if context_size is None:
            context_size = hidden_size

        # ---- Feature extraction (MTGFlow backbone, unchanged) ----
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
        )
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        self.attention = ScaleDotProductAttention(window_size * input_size)

        # ---- New: context aggregator ----
        self.context_aggregator = ContextAggregator(
            hidden_size=hidden_size,
            context_size=context_size,
        )

        # ---- New: latent CNF ----
        self.cnf = LatentCNF(
            latent_dim=hidden_size,
            context_dim=context_size,
            ode_hidden_dim=ode_hidden_dim,
            ode_layers=ode_layers,
            time_embed_dim=time_embed_dim,
            solver=solver,
            rtol=rtol,
            atol=atol,
            adjoint=adjoint,
            kinetic_weight=kinetic_weight,
        )

        self._last_graph = None

    # ------------------------------------------------------------------
    # Internal: feature extraction
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor):
        """
        x: [N, K, L, D]
        Returns:
            h_graph  : [N, K, L, H]  – per-position GNN embeddings
            A        : [N, K, K]     – sensor adjacency
        """
        full_shape = x.shape           # [N, K, L, D]
        N, K, L, D = full_shape

        # Build sensor graph
        A, _ = self.attention(x)       # [N, K, K]
        self._last_graph = A

        # Temporal encoding: reshape to [N*K, L, D]
        x_flat = x.reshape(N * K, L, D)
        h, _ = self.rnn(x_flat)        # [N*K, L, H]

        # Reshape back: [N, K, L, H]
        H = h.shape[-1]
        h = h.reshape(N, K, L, H)

        # Graph message passing
        h_graph = self.gcn(h, A)       # [N, K, L, H]

        return h_graph, A

    # ------------------------------------------------------------------
    # Training forward pass  (returns mean log-prob + reg, scalar per batch)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the *mean* log-likelihood over the batch (for gradient ascent).
        Training loss = -model(x)
        """
        log_prob, reg = self._log_prob_full(x)
        return log_prob.mean() - reg

    # ------------------------------------------------------------------
    # Evaluation forward pass  (returns log-prob per window, shape [N])
    # ------------------------------------------------------------------

    def test(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns log p per window [N].  Anomaly score = -test(x).
        """
        log_prob, _ = self._log_prob_full(x)
        return log_prob

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _log_prob_full(self, x: torch.Tensor):
        """
        Shared computation for train and test.

        x: [N, K, L, D]
        Returns:
            log_prob_per_window : [N]
            reg_term            : scalar
        """
        N, K, L, D = x.shape

        # 1. Encode
        h_graph, A = self._encode(x)          # [N, K, L, H]
        H = h_graph.shape[-1]

        # 2. Global context per window
        c_window = self.context_aggregator(h_graph)   # [N, context_size]

        # 3. Flatten latent embeddings: [N*K*L, H]
        h_flat = h_graph.reshape(N * K * L, H)

        # Broadcast context to match: [N*K*L, context_size]
        c_flat = c_window.unsqueeze(1).unsqueeze(1)   # [N, 1, 1, context_size]
        c_flat = c_flat.expand(N, K, L, -1)           # [N, K, L, context_size]
        c_flat = c_flat.reshape(N * K * L, -1)

        # 4. CNF log-probability
        log_prob_flat, reg_term = self.cnf(h_flat, c_flat)   # [N*K*L], scalar

        # 5. Aggregate: mean over K*L positions per window
        log_prob_per_window = log_prob_flat.reshape(N, K * L).mean(dim=1)  # [N]

        return log_prob_per_window, reg_term

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_graph(self) -> torch.Tensor:
        """Return the last computed sensor adjacency matrix [N, K, K]."""
        return self._last_graph

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: returns NLL per window (higher = more anomalous)."""
        self.eval()
        with torch.no_grad():
            return -self.test(x)

    def set_solver(self, solver: str, rtol: float = None, atol: float = None):
        """
        Switch ODE solver at runtime (e.g. for fast inference vs. accurate training).
        Common usage: model.set_solver('euler') for fast scoring during evaluation.
        """
        self.cnf.solver = solver
        if rtol is not None:
            self.cnf.rtol = rtol
        if atol is not None:
            self.cnf.atol = atol


# ===========================================================================
# 5. Convenience: drop-in wrapper matching the original MTGFlow interface
# ===========================================================================

class MTGFlowCNFWrapper(MTGFlowCNF):
    """
    Drop-in replacement for the original MTGFLOW class in the training notebook.

    Matches the interface expected by the notebook's train_one_entity function:
        loss = -model(x)         # training
        log_p = model.test(x)    # evaluation  → [N]
        A = model.get_graph()    # sensor graph
    """

    def __init__(
        self,
        n_blocks: int,        # kept for API compatibility, not used by CNF
        input_size: int,
        hidden_size: int,
        n_hidden: int,        # kept for API compatibility, maps to ode_layers
        window_size: int,
        n_sensor: int,
        dropout: float = 0.1,
        model: str = "CNF",   # ignored, always CNF
        batch_norm: bool = True,  # ignored by CNF
        # CNF-specific kwargs
        ode_hidden_dim: int = 128,
        solver: str = 'dopri5',
        rtol: float = 1e-4,
        atol: float = 1e-5,
        kinetic_weight: float = 1e-3,
        adjoint: bool = False,
    ):
        super().__init__(
            n_sensor=n_sensor,
            input_size=input_size,
            hidden_size=hidden_size,
            window_size=window_size,
            ode_hidden_dim=ode_hidden_dim,
            ode_layers=max(2, n_hidden + 1),
            dropout=dropout,
            solver=solver,
            rtol=rtol,
            atol=atol,
            adjoint=adjoint,
            kinetic_weight=kinetic_weight,
        )

    def locate(self, x: torch.Tensor):
        """
        Returns per-sensor anomaly scores [N, K] and latent representations.
        Useful for sensor-level fault localisation.
        """
        N, K, L, D = x.shape
        h_graph, A = self._encode(x)
        H = h_graph.shape[-1]

        c_window = self.context_aggregator(h_graph)

        h_flat = h_graph.reshape(N * K * L, H)
        c_flat = c_window.unsqueeze(1).unsqueeze(1).expand(N, K, L, -1)
        c_flat = c_flat.reshape(N * K * L, -1)

        log_prob_flat, _ = self.cnf(h_flat, c_flat)
        log_prob = log_prob_flat.reshape(N, K, L)

        # Per-sensor score: mean over L, then negate for anomaly score
        sensor_scores = -log_prob.mean(dim=2)     # [N, K]

        # Latent: return first timestep embedding per sensor
        z = h_graph[:, :, 0, :].reshape(N * K, H)  # [N*K, H]

        return sensor_scores, z


# ===========================================================================
# 6. Factory function
# ===========================================================================

def build_mtgflow_cnf(
    K: int,
    T: int,
    hidden_size: int = 64,
    ode_hidden_dim: int = 128,
    ode_layers: int = 3,
    dropout: float = 0.0,
    solver: str = 'dopri5',
    rtol: float = 1e-4,
    atol: float = 1e-5,
    kinetic_weight: float = 1e-3,
    adjoint: bool = False,
    device: str = 'cpu',
) -> MTGFlowCNF:
    """
    Factory to build MTGFlowCNF with sensible defaults.

    Parameters
    ----------
    K               : number of sensors
    T               : window size (temporal length)
    hidden_size     : LSTM/GNN hidden dimension
    ode_hidden_dim  : ODE network width
    ode_layers      : ODE network depth
    dropout         : LSTM dropout
    solver          : ODE solver
    rtol, atol      : solver tolerances
    kinetic_weight  : regularisation coefficient
    adjoint         : use adjoint for backprop
    device          : 'cpu' or 'cuda'

    Example
    -------
    >>> model = build_mtgflow_cnf(K=38, T=100, hidden_size=64)
    >>> x = torch.randn(16, 38, 100, 1)
    >>> loss = -model(x)
    >>> loss.backward()
    """
    model = MTGFlowCNF(
        n_sensor=K,
        input_size=1,
        hidden_size=hidden_size,
        window_size=T,
        ode_hidden_dim=ode_hidden_dim,
        ode_layers=ode_layers,
        dropout=dropout,
        solver=solver,
        rtol=rtol,
        atol=atol,
        adjoint=adjoint,
        kinetic_weight=kinetic_weight,
    ).to(device)
    return model
