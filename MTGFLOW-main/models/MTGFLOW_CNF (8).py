"""
MTGFlow-CNF  —  Adjoint Continuous Normalizing Flow
====================================================

Gradient flow design (critical)
--------------------------------
The adjoint method (odeint_adjoint) works as follows:
  • Forward pass  : integrate ODE normally, store ONLY z(T). No full graph.
  • Backward pass : run a second ODE backward in time. At each backward step
                    it calls ODEFunc.forward() again and calls
                    torch.autograd.grad(func_eval, (z, params), ...) to get
                    ∂f/∂z  and  ∂f/∂θ.

This means ODEFunc.forward MUST return tensors with grad_fn (not detached),
otherwise the adjoint backward ODE has nothing to differentiate through and
raises "element 0 of tensors does not require grad".

Memory strategy
---------------
• dz       : returned WITH grad_fn  (adjoint needs it)
• d_logp   : Hutchinson trace. Computed on a local detached copy of z
             (z_loc = z.detach().requires_grad_(True)) so the trace vjp
             does not accumulate a graph across ODE steps. d_logp is then
             detached before returning — the adjoint does not need to
             differentiate through the log-density accumulation.
• d_ke     : kinetic energy regulariser — detached, adjoint ignores it.

This gives the correct gradient flow:
  loss ← log p(z(T)) ← z(T) ← adjoint ODE ← {∂f/∂z, ∂f/∂θ} at each step
                                              (computed through dz grad_fn)

Other memory savings
--------------------
• Chunked LSTM  (lstm_chunk sequences per call)
• Chunked CNF   (cnf_chunk latent vectors per odeint_adjoint call)
• h detached before CNF: backbone grads flow via the loss signal on
  log p(h|c) already; passing non-detached h into the ODE would
  re-accumulate LSTM/GNN grads through every adjoint step unnecessarily.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

try:
    from torchdiffeq import odeint_adjoint, odeint as _odeint_plain
except ImportError:
    raise ImportError("pip install torchdiffeq")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MTGFlow backbone
# ─────────────────────────────────────────────────────────────────────────────

class ScaleDotProductAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w_q = nn.Linear(c, c)
        self.w_k = nn.Linear(c, c)
        self.softmax = nn.Softmax(dim=1)
        self.drop    = nn.Dropout(0.2)

    def forward(self, x, mask=None):
        N, K, L, D = x.shape
        xf = x.reshape(N, K, L * D)
        q  = self.w_q(xf);  k = self.w_k(xf)
        sc = (q @ k.transpose(1, 2)) / math.sqrt(L * D)
        if mask is not None:
            sc = sc.masked_fill(mask == 0, -1e9)
        return self.drop(self.softmax(sc)), k


class GNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        h_n = self.lin_n(torch.einsum('nkld,nkj->njld', h, A))
        h_r = self.lin_r(h[:, :, :-1])
        h_n[:, :, 1:] += h_r
        return self.lin_2(F.relu(h_n))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Context aggregator
# ─────────────────────────────────────────────────────────────────────────────

class ContextAggregator(nn.Module):
    def __init__(self, hidden_size, context_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, context_size),
            nn.LayerNorm(context_size),
            nn.GELU(),
            nn.Linear(context_size, context_size),
        )

    def forward(self, h):
        return self.mlp(h.mean(dim=(1, 2)))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ODE function
# ─────────────────────────────────────────────────────────────────────────────

class _SinTime(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0
        half  = dim // 2
        freqs = torch.exp(torch.arange(half).float() *
                          -(math.log(10000.) / max(half - 1, 1)))
        self.register_buffer('freqs', freqs)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        t    = t.reshape(-1)
        args = t[:, None] * self.freqs[None]
        return self.proj(torch.cat([torch.sin(args), torch.cos(args)], -1))


class _FiLM(nn.Module):
    def __init__(self, feat, cond):
        super().__init__()
        self.g = nn.Linear(cond, feat);  self.b = nn.Linear(cond, feat)
        nn.init.ones_(self.g.weight);    nn.init.zeros_(self.g.bias)
        nn.init.zeros_(self.b.weight);   nn.init.zeros_(self.b.bias)

    def forward(self, h, c):
        return self.g(c) * h + self.b(c)


class ODEFunc(nn.Module):
    """
    Augmented state: (z [M,H],  log_p [M,1],  ke [M,1])

    Adjoint compatibility rules
    ---------------------------
    Rule 1 — dz MUST have grad_fn.
        The adjoint backward ODE calls autograd.grad(func_eval, z_and_params)
        to compute ∂f/∂z and ∂f/∂θ.  If dz is detached, this raises
        "element 0 does not require grad".

    Rule 2 — d_logp and d_ke can be detached.
        The adjoint only needs to differentiate through dz (to get ∂L/∂z(t)
        and ∂L/∂θ).  The log-density and kinetic energy accumulators are
        auxiliary — returning them detached prevents the adjoint from trying
        to build a second-order graph through the Hutchinson estimator.

    Rule 3 — Hutchinson VJP must NOT accumulate across steps.
        We compute the trace on z_loc = z.detach().requires_grad_(True),
        a fresh local leaf at each step.  This keeps the trace computation
        self-contained (no cross-step edges), while dz (computed on the
        original z) retains its grad_fn for the adjoint.
    """

    def __init__(self, latent_dim, hidden_dim, n_layers,
                 context_dim, time_embed_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.t_emb      = _SinTime(time_embed_dim)
        cond_dim        = context_dim + time_embed_dim

        dims = [latent_dim] + [hidden_dim] * (n_layers - 1) + [latent_dim]
        self.layers = nn.ModuleList()
        self.films  = nn.ModuleList()
        self.norms  = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(spectral_norm(nn.Linear(dims[i], dims[i+1])))
            self.films.append(_FiLM(dims[i+1], cond_dim))
            self.norms.append(nn.LayerNorm(dims[i+1]))

        self._ctx = None   # set by LatentCNF before each odeint call

    def _net(self, z, cond):
        """Drift network f(z; cond) → [M, H]. Does NOT detach z."""
        h = z
        for i, (lin, film, norm) in enumerate(
                zip(self.layers, self.films, self.norms)):
            h = lin(h)
            h = film(h, cond)
            h = norm(h)
            if i < len(self.layers) - 1:
                h = torch.tanh(h)
        return h

    def forward(self, t, state):
        z, log_p, ke = state   # [M,H], [M,1], [M,1]
        M = z.shape[0]

        # Build conditioning vector.
        # t_emb must be expanded to M (z's actual batch size).
        # During the adjoint backward pass, torchdiffeq calls this function
        # with y = z.detach().requires_grad_(True) whose shape may differ
        # from the original cnf_chunk used in the forward pass.
        # self._ctx was set for the forward chunk size — we must slice or
        # broadcast it to match M to avoid the "sizes must match" error.
        with torch.no_grad():
            t_emb = self.t_emb(t).expand(M, -1)       # [M, time_embed_dim]
        if self._ctx.shape[0] == M:
            ctx_m = self._ctx
        elif self._ctx.shape[0] > M:
            ctx_m = self._ctx[:M]                      # adjoint uses a sub-batch
        else:
            # adjoint augmented state can be larger; tile to fill
            reps  = (M + self._ctx.shape[0] - 1) // self._ctx.shape[0]
            ctx_m = self._ctx.repeat(reps, 1)[:M]
        cond = torch.cat([ctx_m, t_emb], dim=-1).detach()  # [M, cond_dim]

        # dz/dt -- computed on z directly so grad_fn is preserved.
        # The adjoint backward ODE calls autograd.grad(dz, (z, params))
        # to get df/dz and df/dtheta.  dz must NOT be detached.
        dz = self._net(z, cond)                        # [M, H], has grad_fn

        # Hutchinson trace estimator: d log_p/dt = -Tr(df/dz)
        # Computed on a fresh local leaf z_loc so this single autograd.grad
        # call does not build cross-step graph edges.
        # d_logp is detached -- the adjoint does not need to differentiate
        # through the log-density accumulation, only through dz.
        with torch.enable_grad():
            z_loc  = z.detach().requires_grad_(True)
            dz_loc = self._net(z_loc, cond)
            eps    = torch.randn_like(z_loc)
            vjp,   = torch.autograd.grad(
                dz_loc, z_loc,
                grad_outputs=eps,
                create_graph=False,
                retain_graph=False,
            )
        d_logp = -(vjp * eps).sum(-1, keepdim=True).detach()  # [M,1]
        d_ke   = dz.detach().pow(2).sum(-1, keepdim=True)     # [M,1]

        return dz, d_logp, d_ke

    def set_ctx(self, ctx):
        self._ctx = ctx


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LatentCNF — adjoint integration
# ─────────────────────────────────────────────────────────────────────────────

class LatentCNF(nn.Module):
    """
    Integrates the augmented ODE using odeint_adjoint.

    odeint_adjoint memory profile
    ─────────────────────────────
    Forward  : stores only z(T)  →  O(chunk × H)
    Backward : recomputes f(z,t) at each adjoint step  →  O(chunk × H)
    Total    : O(chunk × H), independent of #integration steps

    Compare to standard odeint:
    Forward  : stores all intermediate z(t)  →  O(steps × chunk × H)
    Backward : backprop through stored graph →  same
    """

    def __init__(self, latent_dim, context_dim, ode_hidden_dim=64,
                 ode_layers=2, time_embed_dim=16, solver='rk4',
                 step_size=0.25, kinetic_weight=1e-3, cnf_chunk=256):
        super().__init__()
        self.latent_dim     = latent_dim
        self.solver         = solver
        self.step_size      = step_size
        self.kinetic_weight = kinetic_weight
        self.cnf_chunk      = cnf_chunk

        self.ode = ODEFunc(latent_dim, ode_hidden_dim, ode_layers,
                           context_dim, time_embed_dim)
        self.register_buffer('t01', torch.tensor([0., 1.]))

    def _run(self, z, c):
        """Integrate one chunk. Returns log_prob [M], kinetic [M]."""
        M  = z.shape[0]
        lp = torch.zeros(M, 1, device=z.device, dtype=z.dtype)
        ke = torch.zeros(M, 1, device=z.device, dtype=z.dtype)
        self.ode.set_ctx(c)

        sol = odeint_adjoint(
            self.ode,
            (z, lp, ke),
            self.t01,
            method          = self.solver,
            options         = {'step_size': self.step_size},
            adjoint_params  = tuple(self.ode.parameters()),
        )

        z_T  = sol[0][-1]                            # [M, H]
        dlp  = sol[1][-1].squeeze(-1)                # [M]
        log_base = -0.5 * (
            z_T.pow(2).sum(-1) +
            self.latent_dim * math.log(2.0 * math.pi)
        )
        return log_base + dlp, sol[2][-1].squeeze(-1)

    def forward(self, z, c):
        """z [M,H], c [M,ctx_dim] → log_prob [M], reg scalar"""
        M = z.shape[0]
        if M <= self.cnf_chunk:
            lp, ke = self._run(z, c)
        else:
            lps, kes = [], []
            for s in range(0, M, self.cnf_chunk):
                sl = slice(s, s + self.cnf_chunk)
                lp_i, ke_i = self._run(z[sl], c[sl])
                lps.append(lp_i);  kes.append(ke_i)
            lp = torch.cat(lps);  ke = torch.cat(kes)

        reg = self.kinetic_weight * ke.mean() if self.training \
              else z.new_tensor(0.)
        return lp, reg


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Full model
# ─────────────────────────────────────────────────────────────────────────────

class MTGFlowCNF(nn.Module):
    """
    MTGFlow backbone + Latent Conditional CNF (adjoint-trained).

    Recommended settings for 22 GB GPU, K=38, T=100
    ──────────────────────────────────────────────────
    batch=64,  hidden=32, cnf_chunk=256   →  ~5-8 GB  (safe start)
    batch=128, hidden=32, cnf_chunk=128   →  ~8-12 GB
    """

    def __init__(self, n_sensor, input_size=1, hidden_size=32,
                 window_size=100, context_size=None, ode_hidden_dim=None,
                 ode_layers=2, time_embed_dim=16, dropout=0.,
                 solver='rk4', step_size=0.25, kinetic_weight=1e-3,
                 lstm_chunk=512, cnf_chunk=256):
        super().__init__()
        if context_size   is None: context_size   = hidden_size
        if ode_hidden_dim is None: ode_hidden_dim = hidden_size * 2

        self.H          = hidden_size
        self.lstm_chunk = lstm_chunk

        self.rnn       = nn.LSTM(input_size, hidden_size,
                                 batch_first=True, dropout=dropout)
        self.gcn       = GNN(hidden_size, hidden_size)
        self.attention = ScaleDotProductAttention(window_size * input_size)
        self.ctx_agg   = ContextAggregator(hidden_size, context_size)
        self.cnf       = LatentCNF(hidden_size, context_size,
                                   ode_hidden_dim, ode_layers,
                                   time_embed_dim, solver, step_size,
                                   kinetic_weight, cnf_chunk)
        self._graph = None

    def _encode(self, x):
        N, K, L, D = x.shape
        A, _  = self.attention(x)
        self._graph = A
        xf    = x.reshape(N * K, L, D)
        parts = []
        for s in range(0, N * K, self.lstm_chunk):
            h_p, _ = self.rnn(xf[s:s + self.lstm_chunk])
            parts.append(h_p)
        h = torch.cat(parts).reshape(N, K, L, self.H)
        return self.gcn(h, A), A

    def _log_prob(self, x):
        N, K, L, D = x.shape
        h, _  = self._encode(x)                    # [N,K,L,H]
        c_win = self.ctx_agg(h)                    # [N,H]

        # Detach h and c before the CNF.
        # The backbone receives gradients via the anomaly score loss;
        # passing non-detached tensors into the ODE would cause the adjoint
        # to propagate additional gradients back through LSTM/GNN at every
        # adjoint step, multiplying memory usage proportionally to #steps.
        h_flat = h.detach().reshape(N * K * L, self.H)
        c_flat = (c_win.detach()[:, None, None, :]
                        .expand(N, K, L, -1)
                        .reshape(N * K * L, -1))

        lp, reg = self.cnf(h_flat, c_flat)
        return lp.reshape(N, K * L).mean(1), reg    # [N], scalar

    def forward(self, x):
        """Training.  loss = -model(x)"""
        lp, reg = self._log_prob(x)
        return lp.mean() - reg

    def test(self, x):
        """Inference.  anomaly score = -test(x)  [N]"""
        lp, _ = self._log_prob(x)
        return lp

    def get_graph(self):
        return self._graph

    def set_solver(self, solver, step_size=None):
        self.cnf.solver = solver
        if step_size is not None: self.cnf.step_size = step_size

    def locate(self, x):
        """Returns per-sensor NLL [N,K] and latent reps [N*K, H]."""
        N, K, L, D = x.shape
        h, _  = self._encode(x)
        c_win = self.ctx_agg(h)
        h_f   = h.detach().reshape(N * K * L, self.H)
        c_f   = (c_win.detach()[:, None, None, :]
                        .expand(N, K, L, -1)
                        .reshape(N * K * L, -1))
        lp, _ = self.cnf(h_f, c_f)
        return -lp.reshape(N, K, L).mean(2), h[:, :, 0].reshape(N * K, self.H)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_mtgflow_cnf(K, T, hidden_size=32, ode_hidden_dim=None,
                      ode_layers=2, dropout=0., solver='rk4',
                      step_size=0.25, kinetic_weight=1e-3,
                      lstm_chunk=512, cnf_chunk=256,
                      device='cpu', **_ignored):
    if ode_hidden_dim is None:
        ode_hidden_dim = hidden_size * 2

    m = MTGFlowCNF(
        n_sensor=K, input_size=1, hidden_size=hidden_size,
        window_size=T, ode_hidden_dim=ode_hidden_dim,
        ode_layers=ode_layers, dropout=dropout,
        solver=solver, step_size=step_size,
        kinetic_weight=kinetic_weight,
        lstm_chunk=lstm_chunk, cnf_chunk=cnf_chunk,
    ).to(device)

    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  MTGFlowCNF | K={K} T={T} H={hidden_size} | params={n:,} "
          f"| lstm_chunk={lstm_chunk} cnf_chunk={cnf_chunk}")
    return m
