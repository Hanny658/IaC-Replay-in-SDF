"""Cortical predictive-coding network, version 1: the MLP-PC core under cortical constraints.

Same d-64-32-1 core and the same predictive-coding energy as MLP-PC (pc.py), trained by
prospective configuration -- relax the value nodes to (near) equilibrium first, then make one
local Hebbian update (Song et al., Nature Neuroscience 2024).  On top of that, five constraints
that cortex is known to obey, each with its own switch so that its cost can be measured alone:

  transport=False  no weight transport.  The top-down error reaches layer l through dedicated
                   feedback synapses B^l, not through (W^{l+1})^T.  B learns with the
                   Kolen-Pollack rule: the same local product as W^{l+1} plus weight decay, so the
                   two converge to each other without either ever reading the other.
  bounded=True     burst-limited error channel.  What travels top-down, and what drives
                   plasticity in hidden layers, is kappa*tanh(eps/kappa): a bounded, saturating
                   function of the apical error, as a burst probability is (Payeur et al. 2021).
  kwta=True        PV-type lateral inhibition.  Only the top `active_frac` of each hidden layer
                   fires; the rest are silenced (k-winners-take-all).  Rates are non-negative
                   (ReLU), because a rate is a rate.
  dale=True        Dale's law.  Each hidden neuron is excitatory (80%) or inhibitory (20%) and all
                   of its outgoing weights carry that sign; weights are projected back onto the
                   allowed sign after every update.  Not applied to the input layer, whose
                   standardised features are signed quantities, nor to feedback synapses.
  homeo=...        homeostasis, two implementations:
                   "threshold" (v1): each neuron nudges its own bias every step toward a target
                       k-WTA win rate.  Kept for reproducibility; it is too fast, fights the
                       error-driven learning, and on SUPPORT2 (17k updates) degrades the model.
                   "scaling" (v2): synaptic scaling (Turrigiano).  Each neuron carries a gain on
                       its basal drive that drifts multiplicatively, with a slow time constant,
                       toward the population-mean activity; a neuron that has gone silent gets a
                       slow threshold rescue instead, since scaling a negative drive cannot revive
                       it.  Neither rule needs the k-WTA competition to be defined.
                   "sleep" (v3): the same synaptic scaling, but offline.  Nothing homeostatic
                       happens during waking updates; between epochs the network is shown a
                       sample of the data with learning off, every neuron's mean rate is compared
                       with a set point that is FIXED on the first night, and its gain is scaled
                       once, multiplicatively.  Both per-step rules degraded with the number of
                       updates because they chased a set point that moved with learning; this is
                       the synaptic-homeostasis-in-sleep picture (Tononi & Cirelli).
  dale_fb=True     Dale's law on the feedback synapses too, with the column (postsynaptic) signs
                   of the forward twin, so that B can still align with W^T under Dale's law.

With every switch off and act="tanh" this is exactly MLP-PC; with every switch off and
act="relu" it is the control for the activation change alone.  Everything is torch.no_grad:
no autograd anywhere.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import ScoringModel
from .torch_utils import HIDDEN, to_t


class CortexNet:
    def __init__(self, sizes, seed=0, act="relu", transport=False, bounded=True, kwta=True,
                 dale=True, homeo="scaling", dale_fb=False, active_frac=0.25, ei_frac=0.8,
                 kappa=1.0, kp_decay=1e-2, opt="adam", adam_eps=1e-8, momentum=0.9,
                 conn_density=1.0, conn_mode="dist", conn_lambda=0.15, regrow=0.0, input_shape=None,
                 burst_gate=False, burst_baseline=False, dream_batches=0, dream_eta=0.1,
                 sweep=False, burst_mult=False, spike_train=0, mirror=0,
                 decoder=False, rem_neg=0.0, rem_margin=1.0):
        homeo = "threshold" if homeo is True else homeo  # v1 spelling
        g = torch.Generator().manual_seed(seed)
        self.sizes = list(sizes)
        self.L = len(sizes) - 1
        self.act_name, self.transport, self.bounded = act, transport, bounded
        self.kwta, self.dale, self.homeo, self.dale_fb = kwta, dale, homeo, dale_fb
        self.active_frac, self.kappa, self.kp_decay = active_frac, kappa, kp_decay
        # Optimiser for the local updates.  Adam's per-synapse normalisation is what blew up the
        # wide k-WTA runs: after weights and gradients had shrunk for thousands of steps, v was
        # tiny, and the discontinuous change of gradient when k-WTA winners flip gave g/sqrt(v)
        # steps an order of magnitude too large for hundreds of steps.  adam_eps caps that
        # amplification; "sgd" (heavy-ball momentum) has no per-synapse state at all, which is
        # also the more defensible biology.
        self.opt, self.adam_eps, self.momentum = opt, adam_eps, momentum

        self.W, self.b = [None], [None]  # forward synapses, index 1..L
        for l in range(1, self.L + 1):
            n_out, n_in = sizes[l], sizes[l - 1]
            self.W.append(torch.randn(n_out, n_in, generator=g) / np.sqrt(n_in))
            self.b.append(torch.zeros(n_out))
        # feedback synapses B^l carry eps^{l+1} down to layer l; same shape as W^{l+1}
        self.B = [None] + [torch.randn(sizes[l + 1], sizes[l], generator=g) / np.sqrt(sizes[l])
                           for l in range(1, self.L)]
        # neuron types of the hidden layers (excitatory +1 / inhibitory -1), fixed at birth
        self.sign = [None]
        for l in range(1, self.L):
            n = sizes[l]
            s = torch.ones(n)
            s[int(round(ei_frac * n)):] = -1.0
            self.sign.append(s[torch.randperm(n, generator=g)])
        if self.dale:
            self._project_dale()
        # homeostasis state: v1 win-rate trace; v2 per-neuron gain on the basal drive plus a slow
        # trace of mean activity
        self.p_win = [None] + [torch.full((sizes[l],), active_frac) for l in range(1, self.L)]
        self.gain = [None] + [torch.ones(sizes[l]) for l in range(1, self.L)]
        self.a_bar = [None] + [None for _ in range(1, self.L)]
        self.set_point = [None] + [None for _ in range(1, self.L)]  # v3: fixed on the first night
        # event-driven cost of the last forward pass (filled by forward)
        self.last_cost = None

        # Phase 4C: self-organised connectivity.  Every layer's neurons sit on a 2-D sheet (the
        # input on its image grid, hidden layers on square grids); a synapse between layers exists
        # with probability exp(-d / lambda), thinned to `conn_density`.  "random" keeps the density
        # but ignores distance, which is the control.  Masks apply to the forward synapses of the
        # hidden layers and to their feedback twins; the output layer stays dense.  `regrow` is the
        # fraction of a layer's synapses that structural plasticity replaces after each epoch:
        # the weakest are pruned and the same number regrown (distance-biased or uniform).
        self.conn_density, self.conn_mode, self.conn_lambda, self.regrow = conn_density, conn_mode, conn_lambda, regrow
        # Phase 5B: two ingredients of burst multiplexing (Payeur et al. 2021; BurstCCN).
        #   burst_gate: a neuron that emits no event cannot burst, so a silent neuron neither learns
        #   from the apical error nor relays it downward.
        #   burst_baseline: the burst signal is read relative to a slow per-neuron baseline
        #   (burst-rate adaptation), which strips any standing offset from the error a synapse sees.
        self.burst_gate, self.burst_baseline = burst_gate, burst_baseline
        self.e_bar = [None] + [torch.zeros(sizes[l]) for l in range(1, self.L)]
        # Phase 5C: sleep as the negative phase.  Each night the network dreams `dream_batches`
        # batches -- hidden patterns settled top-down from random labels with no input -- and makes
        # an anti-Hebbian update on them scaled by dream_eta * eta: the energy of the model's own
        # fantasies is raised, as the Forward-Forward negative phase lowers their goodness.
        self.dream_batches, self.dream_eta = dream_batches, dream_eta
        # Phase 6A: the single-phase core of BurstCCN.  sweep=True replaces the relaxation by one
        # feed-forward pass and one top-down burst sweep (the apical error of each layer is the
        # gated, bounded burst signal of the layer above through the feedback synapses), then the
        # same local update.  burst_mult makes plasticity proportional to the event rate times the
        # burst deviation (the burst RATE, e * p), the multiplicative coding of Payeur et al.
        self.sweep, self.burst_mult = sweep, burst_mult
        # Phase 6B: spike-count noise during training.  Every hidden rate a is replaced by
        # Poisson(a * spike_train) / spike_train while training, so the local rules only ever see
        # spike-count estimates of the rates; evaluation uses the exact rates.
        self.spike_train, self.training = spike_train, True
        # Phase 6C: weight mirroring (Akrout et al. 2019) as the way to align B with W^T.  Each
        # night `mirror` batches of noise are injected into a layer, propagated through W, and B
        # learns the Hebbian product of noise and projection with a matched decay, whose fixed
        # point is B = W (in expectation).  No transport, and no per-data-set decay to tune.
        self.mirror = mirror
        # Phase 7B: a generative path to the input.  B0 decodes layer-1 activity back into input
        # space and is trained by the input layer's own prediction error (the bottom of a
        # predictive-coding generative model), reciprocal to W1's connectivity.  With it a dream
        # is a full sample, and REM can be judged: rem_neg > 0 turns on the reverse-learning
        # (Crick & Mitchison) side with a margin -- only fantasies whose per-layer goodness rises
        # above the running goodness of real input are pushed down, real input is left to the
        # normal learning rule.  Generative rehearsal (dreams replayed as positives) is a run-loop
        # matter and needs nothing here beyond rem_generate().
        self.decoder, self.rem_neg, self.rem_margin = decoder, rem_neg, rem_margin
        if decoder:
            self.B0 = [None, torch.randn(sizes[1], sizes[0], generator=g) / np.sqrt(sizes[1])]
            self.mB0, self.vB0 = [None, torch.zeros_like(self.B0[1])], [None, torch.zeros_like(self.B0[1])]
        self.goodness = [None] + [None for _ in range(1, self.L)]  # EMA of mean(a^2) on real input
        # Imagination synapses: label -> layer-1 activity, a per-class mean and variance learned by
        # a local delta rule on waking activity (semantic prototypes; no raw sample is stored).
        # Settling top-down from a one-hot label through the error-feedback synapses produced
        # hidden states with no class content (they carry errors, not content), so dreams start
        # from these prototypes instead, with noise, and are decoded through B0.
        if decoder:
            self.G = torch.zeros(sizes[1], sizes[self.L])
            self.Gvar = torch.ones(sizes[1], sizes[self.L])
            self.Gseen = torch.zeros(sizes[self.L])
        self.mask = [None] * (self.L + 1)
        if conn_density < 1.0:
            self.coords = [self._sheet(sizes[0], input_shape, g)] + \
                [self._sheet(sizes[l], None, g) for l in range(1, self.L)]
            for l in range(1, self.L):  # masks for W^1..W^{L-1}; W^L (readout) stays dense
                self.mask[l] = self._draw_mask(l, g)
            self._apply_mask()

        # Adam state (the optimiser only rescales the local gradients)
        self.t = 0
        self.mW = [None] + [torch.zeros_like(w) for w in self.W[1:]]
        self.vW = [None] + [torch.zeros_like(w) for w in self.W[1:]]
        self.mb = [None] + [torch.zeros_like(b) for b in self.b[1:]]
        self.vb = [None] + [torch.zeros_like(b) for b in self.b[1:]]
        self.mB = [None] + [torch.zeros_like(w) for w in self.B[1:]]
        self.vB = [None] + [torch.zeros_like(w) for w in self.B[1:]]

    # ---------------------------------------------------------------- neurons
    def _rate(self, x):
        return torch.relu(x) if self.act_name == "relu" else torch.tanh(x)

    def _k(self, n):
        return max(1, int(round(self.active_frac * n)))

    def act(self, l, x):
        """Firing rate of layer l given its value nodes (input layer passes through)."""
        if l == 0:
            return x
        a = self._rate(x)
        if self.spike_train and self.training:  # 6B: the rate is only known through its spike count
            a = torch.poisson(a * self.spike_train) / self.spike_train
        # 8B refractory local sleep: units flagged in `suppress` sit out this competition (they
        # fired last round and are forced offline now), which makes them asleep for this input
        sup = getattr(self, "suppress", None)
        if sup is not None and l < len(sup) and sup[l] is not None:
            a = a * (1.0 - sup[l])
        if self.kwta:
            k = self._k(a.shape[1])
            thr = a.topk(k, dim=1).values[:, -1:]
            a = torch.where(a >= thr, a, torch.zeros_like(a))
        return a

    def dact(self, l, x, a):
        if l == 0:
            return torch.ones_like(x)
        if self.act_name == "relu":
            return (a > 0).float()  # silenced units (k-WTA) pass nothing either way
        return (1 - a ** 2) * ((a != 0).float() if self.kwta else 1.0)

    def _burst(self, eps):
        return self.kappa * torch.tanh(eps / self.kappa) if self.bounded else eps

    # ---------------------------------------------------------------- connectivity (4C)
    @staticmethod
    def _sheet(n, shape, g):
        """Unit-square coordinates for n neurons: an image grid if given, else a square grid."""
        if shape is not None:
            h, w = shape
            ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            return torch.stack([ys.flatten() / max(h - 1, 1), xs.flatten() / max(w - 1, 1)], 1).float()
        side = int(np.ceil(np.sqrt(n)))
        ys, xs = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        c = torch.stack([ys.flatten(), xs.flatten()], 1).float()[:n] / max(side - 1, 1)
        return c

    def _dist_prob(self, l):
        """Connection propensity of W^l (post = layer l, pre = layer l-1) from sheet distance."""
        d = torch.cdist(self.coords[l], self.coords[l - 1])
        return torch.exp(-d / self.conn_lambda)

    def _draw_mask(self, l, g):
        n_out, n_in = self.W[l].shape
        n_keep = int(round(self.conn_density * n_out * n_in))
        if self.conn_mode == "dist":
            p = self._dist_prob(l).flatten()
            idx = torch.multinomial(p / p.sum(), n_keep, replacement=False, generator=g)
        else:
            idx = torch.randperm(n_out * n_in, generator=g)[:n_keep]
        m = torch.zeros(n_out * n_in)
        m[idx] = 1.0
        return m.view(n_out, n_in)

    def _apply_mask(self):
        for l in range(1, self.L):
            if self.mask[l] is not None:
                self.W[l] = self.W[l] * self.mask[l]
                if l >= 2 and not self.transport:
                    self.B[l - 1] = self.B[l - 1] * self.mask[l]

    def structural_plasticity(self, g=None):
        """Prune the weakest `regrow` fraction of existing synapses and regrow as many elsewhere."""
        if self.regrow <= 0:
            return
        g = g or torch.Generator().manual_seed(int(self.t))
        for l in range(1, self.L):
            m = self.mask[l]
            if m is None:
                continue
            alive = m.flatten().nonzero().squeeze(1)
            n_swap = int(round(self.regrow * alive.numel()))
            if n_swap == 0:
                continue
            w = self.W[l].abs().flatten()[alive]
            prune = alive[w.topk(n_swap, largest=False).indices]
            dead = (m.flatten() == 0).nonzero().squeeze(1)
            if self.conn_mode == "dist":
                p = self._dist_prob(l).flatten()[dead]
                born = dead[torch.multinomial(p / p.sum(), n_swap, replacement=False, generator=g)]
            else:
                born = dead[torch.randperm(dead.numel(), generator=g)[:n_swap]]
            flat = m.flatten().clone()
            flat[prune] = 0.0
            flat[born] = 1.0
            self.mask[l] = flat.view_as(m)
            # a new synapse starts small and with a fresh optimiser state
            Wf = self.W[l].flatten().clone()
            Wf[born] = torch.randn(n_swap, generator=g) * 0.01
            self.W[l] = Wf.view_as(self.W[l])
            for state in (self.mW, self.vW):
                sf = state[l].flatten().clone()
                sf[born] = 0.0
                state[l] = sf.view_as(state[l])
        self._apply_mask()
        if self.dale:
            self._project_dale()

    def _project_dale(self):
        for l in range(2, self.L + 1):  # W^l has presynaptic layer l-1, a hidden layer
            s = self.sign[l - 1][None, :]
            self.W[l] = s * torch.clamp(s * self.W[l], min=0.0)
            if self.dale_fb and not self.transport:
                # B^{l-1} mirrors W^l, so its column (postsynaptic) signs follow the same neuron
                # types: an inhibitory neuron receives its feedback with flipped sign, as through
                # a disinhibitory relay.  Without this B cannot align with W under Dale's law.
                self.B[l - 1] = s * torch.clamp(s * self.B[l - 1], min=0.0)

    # ---------------------------------------------------------------- inference
    def _g(self, l):
        return self.gain[l] if l < self.L else 1.0

    def mu(self, l, a_prev):
        return self._g(l) * (a_prev @ self.W[l].T) + self.b[l]

    def forward(self, X):
        # Phase 9C: an optional fixed front-end (dentate-gyrus / granule-cell expansion) between
        # the raw input and layer 1.  It is not a layer of this net: no plasticity, no feedback.
        front = getattr(self, "front", None)
        front_cost = 0.0
        if front is not None:
            X = front(X)
            front_cost = front.cost
        x, a = [X], [X]
        dense, synops = 0, front_cost
        for l in range(1, self.L + 1):
            # event-driven cost: only a non-zero presynaptic rate costs a synaptic operation, and
            # only along synapses that exist (masked fan-out when connectivity is sparse)
            dense += self.sizes[l - 1] * self.sizes[l]
            fan_out = self.sizes[l] if self.mask[l] is None else float(self.mask[l].sum(0).mean())
            synops += float((a[-1] != 0).float().sum(1).mean()) * fan_out
            x.append(self.mu(l, a[-1]))
            a.append(self.act(l, x[-1]))
        self.last_cost = {"dense_macs": dense, "synops": synops,
                          "active": [float((a[l] != 0).float().mean()) for l in range(1, self.L)]}
        return x, a

    def _err_up(self, eps, a, l):
        """The burst signal layer l+1 sends down: bounded, gated or rate-multiplied."""
        err_up = self._burst(eps[l + 1])
        if l + 1 < self.L:
            if self.burst_mult:
                err_up = err_up * a[l + 1]
            elif self.burst_gate:
                err_up = err_up * (a[l + 1] > 0).float()
        return err_up

    def sweep_errors(self, X, Y):
        """6A single phase: one feed-forward pass, one top-down burst sweep, no settling."""
        x, a = self.forward(X)
        eps = [None] * (self.L + 1)
        eps[self.L] = Y - x[self.L]
        for l in range(self.L - 1, 0, -1):
            fb = self.W[l + 1] if self.transport else self.B[l]
            eps[l] = self.dact(l, x[l], a[l]) * ((self._g(l + 1) * self._err_up(eps, a, l)) @ fb)
        x[self.L] = Y
        return x, a, eps

    def relax(self, X, Y, T, gamma):
        """Prospective configuration: settle the hidden value nodes with input and target clamped."""
        if self.sweep:
            return self.sweep_errors(X, Y)
        x, a = self.forward(X)
        x[self.L] = Y
        for _ in range(T):
            eps = [None] + [x[l] - self.mu(l, a[l - 1]) for l in range(1, self.L + 1)]
            for l in range(1, self.L):
                fb = self.W[l + 1] if self.transport else self.B[l]
                top_down = (self._g(l + 1) * self._err_up(eps, a, l)) @ fb
                x[l] = x[l] + gamma * (-eps[l] + self.dact(l, x[l], a[l]) * top_down)
                a[l] = self.act(l, x[l])
        eps = [None] + [x[l] - self.mu(l, a[l - 1]) for l in range(1, self.L + 1)]
        return x, a, eps

    def mirror_phase(self, n, g, eta_m=0.01):
        """6C: one weight-mirror batch per hidden-to-hidden connection.  Noise xi in layer l-1 is
        projected through W^l; B^{l-1} moves toward the Hebbian product of projection and noise
        with a matched decay, so that E[B] -> W^l.  Local (each feedback synapse sees only the
        noise at its postsynaptic end and the projection at its presynaptic end)."""
        if self.transport:
            return
        for l in range(2, self.L + 1):
            xi = torch.randn(n, self.sizes[l - 1], generator=g)
            y = xi @ self.W[l].T
            self.B[l - 1] = (1 - eta_m) * self.B[l - 1] + eta_m * (y.T @ xi) / n
        self._apply_mask()
        if self.dale:
            self._project_dale()

    def imagine(self, n, g, T, gamma, labels=None):
        """Settle the hidden layers top-down from clamped labels with no input (the dream state)."""
        if labels is None:
            labels = torch.randint(0, self.sizes[self.L], (n,), generator=g)
        Y = torch.nn.functional.one_hot(labels, self.sizes[self.L]).float()
        x, a = [None] * (self.L + 1), [None] * (self.L + 1)
        x[1] = 0.1 * torch.randn(n, self.sizes[1], generator=g)
        a[1] = self.act(1, x[1])
        for l in range(2, self.L):
            x[l] = self.mu(l, a[l - 1])
            a[l] = self.act(l, x[l])
        x[self.L] = Y
        for _ in range(T):
            eps = [None, torch.zeros_like(x[1])] + [x[l] - self.mu(l, a[l - 1]) for l in range(2, self.L + 1)]
            for l in range(1, self.L):
                fb = self.W[l + 1] if self.transport else self.B[l]
                top_down = (self._g(l + 1) * self._err_up(eps, a, l)) @ fb
                x[l] = x[l] + gamma * (-eps[l] + self.dact(l, x[l], a[l]) * top_down)
                a[l] = self.act(l, x[l])
        eps = [None, torch.zeros_like(x[1])] + [x[l] - self.mu(l, a[l - 1]) for l in range(2, self.L + 1)]
        return x, a, eps, labels

    def decode(self, a1):
        """Input-space reconstruction of a layer-1 activity through the generative synapses."""
        B0 = self.B0[1] if self.mask[1] is None else self.B0[1] * self.mask[1]
        return a1 @ B0

    def rem_generate(self, n, g, T, gamma, refine=1, source="proto"):
        """A batch of dreams as full samples.  source="proto": sample layer-1 activity from the
        learned class prototypes (mean + noise scaled by the learned variance), apply the layer's
        own nonlinearity and competition, decode through B0, and pass the image once more through
        the recognition path and the decoder to sharpen it.  source="label": the phase-5 style
        top-down settling from a clamped label (kept for comparison; it carries no class content)."""
        if source == "label" or not hasattr(self, "G"):
            x, a, _, labels = self.imagine(n, g, T, gamma)
            a1 = a[1]
        else:
            labels = torch.randint(0, self.sizes[self.L], (n,), generator=g)
            mean, var = self.G[:, labels].T, self.Gvar[:, labels].T
            a1 = self.act(1, mean + var.sqrt() * torch.randn(mean.shape, generator=g))
        x_hat = self.decode(a1)
        for _ in range(refine):
            _, a_fwd = self.forward(x_hat)
            x_hat = self.decode(a_fwd[1])
        return x_hat, labels

    def rem_negative(self, X_hat, eta, lam=None, margin=None):
        """Reverse learning with a margin on a batch of dreams: for every hidden layer, fantasies
        whose goodness mean(a^2) exceeds the running goodness of real input by less than the margin
        are left alone; the rest get their goodness pushed down through the same local product the
        waking rule uses.  Real input never enters here."""
        lam = self.rem_neg if lam is None else lam
        margin = self.rem_margin if margin is None else margin
        _, a = self.forward(X_hat)
        self.t += 1
        n = X_hat.shape[0]
        for l in range(1, self.L):
            if self.goodness[l] is None:
                continue
            G = (a[l] ** 2).mean(1, keepdim=True)
            push = torch.sigmoid(G - self.goodness[l] + margin)  # ~0 for fantasies already "unreal"
            e = -lam * push * a[l] * (a[l] > 0).float()
            gW = e.T @ a[l - 1] / n
            self._adam(self.W, gW, self.mW, self.vW, l, eta)
        if self.dale:
            self._project_dale()
        self._apply_mask()

    def dream(self, n, g, T, gamma, eta, weight_decay=0.0):
        """Phase 5C (kept for the record): anti-Hebbian update on a settled dream, no margin."""
        x, a, eps, _ = self.imagine(n, g, T, gamma)
        self.local_update(x, a, eps, eta * self.dream_eta, weight_decay, sign=-1.0, first=2)

    # ---------------------------------------------------------------- learning
    def _adam(self, p, g, m, v, l, eta, betas=(0.9, 0.999), mask=None):
        """Local update through the optimiser.  With `mask` (same shape as the parameter) only the
        masked synapses move AND only their optimiser state advances, so an update confined to
        sleeping synapses cannot leak into waking ones through the momentum (phase 8A)."""
        if self.opt == "sgd":  # heavy-ball momentum, no second-moment state
            if mask is None:
                m[l] = self.momentum * m[l] + g
                p[l] = p[l] + eta * m[l]
            else:
                m[l] = torch.where(mask > 0, self.momentum * m[l] + g, m[l])
                p[l] = p[l] + eta * mask * m[l]
            return
        if mask is None:
            m[l] = betas[0] * m[l] + (1 - betas[0]) * g
            v[l] = betas[1] * v[l] + (1 - betas[1]) * g * g
            mh = m[l] / (1 - betas[0] ** self.t)
            vh = v[l] / (1 - betas[1] ** self.t)
            p[l] = p[l] + eta * mh / (vh.sqrt() + self.adam_eps)
            return
        on = mask > 0
        m[l] = torch.where(on, betas[0] * m[l] + (1 - betas[0]) * g, m[l])
        v[l] = torch.where(on, betas[1] * v[l] + (1 - betas[1]) * g * g, v[l])
        mh = m[l] / (1 - betas[0] ** self.t)
        vh = v[l] / (1 - betas[1] ** self.t)
        p[l] = p[l] + eta * mask * mh / (vh.sqrt() + self.adam_eps)

    def local_update(self, x, a, eps, eta, weight_decay=0.0, eta_homeo=1e-2, tau=0.05,
                     eta_scale=1e-3, tau_slow=0.01, sign=1.0, first=1, tau_e=0.01,
                     syn_mask=None, bias_mask=None):
        """One local update from a settled state.  sign=-1 makes it anti-Hebbian (the dream
        phase); first=2 leaves the input synapses alone, which a dream has no input for.
        syn_mask / bias_mask (phase 8A, local sleep): per-layer masks over synapses (n_out x n_in)
        and neurons (n_out); only the masked ones move, and their optimiser state alone advances."""
        self.t += 1
        n = x[self.L].shape[0]
        replay = syn_mask is not None
        for l in range(first, self.L + 1):
            e = eps[l] if l == self.L else self._burst(eps[l])  # output error is the loss itself
            if l < self.L and self.burst_baseline:
                if sign > 0 and not replay:  # the baseline tracks waking activity only
                    self.e_bar[l] = (1 - tau_e) * self.e_bar[l] + tau_e * e.mean(0)
                e = e - self.e_bar[l]
            if l < self.L and self.burst_mult:
                e = e * a[l]  # plasticity follows the burst rate: event rate times burst deviation
            elif l < self.L and self.burst_gate:
                e = e * (a[l] > 0).float()  # no event, no burst, no plasticity
            e = sign * self._g(l) * e  # the drive is gain-scaled, so the synapse sees the scaled error
            # the decoupled Kolen-Pollack decay below replaces L2 on the synapses that have a twin
            wd = 0.0 if (l >= 2 and not self.transport) else weight_decay
            gW = e.T @ a[l - 1] / n - wd * self.W[l]
            gb = e.mean(0)
            sm = syn_mask[l] if replay else None
            bm = bias_mask[l] if replay else None
            self._adam(self.W, gW, self.mW, self.vW, l, eta, mask=sm)
            self._adam(self.b, gb, self.mb, self.vb, l, eta, mask=bm)
            if l >= 2 and not self.transport:
                # Kolen-Pollack: B^{l-1} sees the same local product as W^l, and both carry the
                # same decoupled decay, so their initial (random) difference washes out and they
                # converge on each other without either ever reading the other.
                gB = e.T @ a[l - 1] / n
                self._adam(self.B, gB, self.mB, self.vB, l - 1, eta, mask=sm)
                shrink = self.kp_decay if sm is None else self.kp_decay * sm
                self.W[l] = self.W[l] - shrink * self.W[l]
                self.B[l - 1] = self.B[l - 1] - shrink * self.B[l - 1]
        if self.dale:
            self._project_dale()
        self._apply_mask()
        if sign > 0 and first == 1 and not replay:  # waking, with real input: train the decoder, track goodness
            if self.decoder:
                e0 = x[0] - self.decode(a[1])  # the input layer's prediction error
                gB0 = a[1].T @ e0 / n
                self._adam(self.B0, gB0, self.mB0, self.vB0, 1, eta)
                if self.mask[1] is not None:
                    self.B0[1] = self.B0[1] * self.mask[1]
                # imagination synapses: delta rule toward this batch's per-class mean activity,
                # with the pre-nonlinearity value x[1] as the target so that act() can be re-applied
                labels = x[self.L].argmax(1)
                for c in labels.unique().tolist():
                    m = labels == c
                    xc = x[1][m]
                    rate = 0.05 if self.Gseen[c] > 0 else 1.0
                    self.G[:, c] = (1 - rate) * self.G[:, c] + rate * xc.mean(0)
                    self.Gvar[:, c] = (1 - rate) * self.Gvar[:, c] + rate * ((xc - self.G[:, c]) ** 2).mean(0)
                    self.Gseen[c] += 1
            for l in range(1, self.L):
                G = float((a[l] ** 2).mean())
                self.goodness[l] = G if self.goodness[l] is None else 0.99 * self.goodness[l] + 0.01 * G
        if self.homeo == "threshold":  # v1
            for l in range(1, self.L):
                won = (a[l] > 0).float().mean(0)
                self.p_win[l] = (1 - tau) * self.p_win[l] + tau * won
                self.b[l] = self.b[l] + eta_homeo * (self.active_frac - self.p_win[l])
        elif self.homeo == "scaling":  # v2
            for l in range(1, self.L):
                mean_a = a[l].mean(0)
                if self.a_bar[l] is None:
                    self.a_bar[l] = mean_a
                else:
                    self.a_bar[l] = (1 - tau_slow) * self.a_bar[l] + tau_slow * mean_a
                rho = self.a_bar[l].mean()  # set point = the population's own mean activity
                drift = (rho - self.a_bar[l]) / (rho + 1e-6)
                self.gain[l] = torch.clamp(self.gain[l] * torch.exp(eta_scale * drift), 0.1, 10.0)
                silent = self.a_bar[l] < 0.05 * rho  # scaling cannot revive a negative drive
                self.b[l] = self.b[l] + eta_scale * silent.float()

    def sleep(self, X, beta=0.1, rescue=0.05, max_n=4096):
        """v3 homeostasis: one offline scaling step, to be called between epochs (homeo="sleep").

        Learning is off; the network only looks.  Each hidden neuron's mean rate over the sample is
        compared with the layer's set point, which is the population mean measured on the first
        night and never moved again, and its gain is scaled once by exp(beta * relative gap).  A
        neuron that has fallen silent gets a small threshold rescue instead, since scaling a
        negative drive cannot revive it.
        """
        _, a = self.forward(X[:max_n])
        for l in range(1, self.L):
            a_bar = a[l].mean(0)
            if self.set_point[l] is None:
                self.set_point[l] = a_bar.mean()
            rho = self.set_point[l]
            if self.homeo == "sleep_guard":
                # Guard-rail form: a neuron inside the tolerance band [lo, hi] x set point is left
                # alone; only rates outside it are pulled back to the nearest edge.  The equaliser
                # form above ("sleep") pulls every neuron toward the mean, which under k-WTA is a
                # positive feedback loop -- boosted losers start winning, squeezed winners lose
                # their function -- and it collapsed 2 of 3 MNIST seeds.  Biologically the scaling
                # set point comes with a tolerance range (Turrigiano; Hengen et al. 2016).
                lo, hi = 0.25 * rho, 4.0 * rho
                drift = torch.where(a_bar < lo, (lo - a_bar) / (rho + 1e-6),
                                    torch.where(a_bar > hi, (hi - a_bar) / (rho + 1e-6), torch.zeros_like(a_bar)))
                self.gain[l] = torch.clamp(self.gain[l] * torch.exp(beta * drift), 0.25, 4.0)
            else:
                drift = (rho - a_bar) / (rho + 1e-6)
                self.gain[l] = torch.clamp(self.gain[l] * torch.exp(beta * drift), 0.2, 5.0)
            silent = a_bar < rescue * rho
            self.b[l] = self.b[l] + 0.1 * beta * silent.float()


class CorticalPC(ScoringModel):
    name, family = "MLP-Cortex", "local-learning"

    def __init__(self, seed=0, epochs=300, eta=1e-3, gamma=0.1, T=20, batch_size=128, weight_decay=1e-3,
                 variant="full", hidden=HIDDEN, **net_kwargs):
        super().__init__(seed)
        self.epochs, self.eta, self.gamma, self.T, self.batch_size, self.weight_decay = \
            epochs, eta, gamma, T, batch_size, weight_decay
        self.variant, self.hidden, self.net_kwargs = variant, tuple(hidden), net_kwargs
        self.name = "MLP-Cortex" if variant == "full" else f"MLP-Cortex[{variant}]"

    def fit(self, X, y):
        g = torch.Generator().manual_seed(self.seed)
        Xt, Yt = to_t(X), to_t(y).unsqueeze(1)
        self.net = CortexNet([X.shape[1], *self.hidden, 1], seed=self.seed, **self.net_kwargs)
        n = len(Xt)
        self.hist = []
        with torch.no_grad():
            for _ in range(self.epochs):
                perm = torch.randperm(n, generator=g)
                tot = 0.0
                for i in range(0, n, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    x, a, eps = self.net.relax(Xt[idx], Yt[idx], self.T, self.gamma)
                    self.net.local_update(x, a, eps, self.eta, self.weight_decay)
                    tot += (eps[self.net.L] ** 2).sum().item()
                self.hist.append(tot / n)
                if self.net.homeo in ("sleep", "sleep_guard"):
                    self.net.sleep(Xt[perm])
                self.net.structural_plasticity(g)
                for _ in range(self.net.mirror):
                    self.net.mirror_phase(self.batch_size, g)
        self.net.training = False
        return self

    def decision_scores(self, X):
        with torch.no_grad():
            return self.net.forward(to_t(X))[0][-1].squeeze(1).numpy()

    def describe(self):
        net = self.net
        on = [f for f, v in (("no-transport", not net.transport), ("bounded", net.bounded),
                             (f"kwta@{net.active_frac}", net.kwta), ("dale", net.dale),
                             ("dale_fb", net.dale_fb), (f"homeo:{net.homeo}", bool(net.homeo))) if v]
        c = net.last_cost or {}
        cost = (f" cost: dense_macs={c.get('dense_macs')} synops={c.get('synops', 0):.0f} "
                f"active={[round(v, 3) for v in c.get('active', [])]}") if c else ""
        return f"arch={self.hidden} act={net.act_name} T={self.T} on={on} final_out_err={self.hist[-1]:.3f}{cost}"
