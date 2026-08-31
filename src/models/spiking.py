"""Spiking inference for a trained CortexNet (phase 4A): rate network in, spike counts out.

Training is left to the predictive-coding rules; here the fitted rate network is run as a network
of integrate-and-fire neurons for T_s time steps, which is the standard rate-to-spike conversion
(Diehl et al. 2015, Rueckauer et al. 2017), so that the energy axis is measured in real synaptic
events rather than in "non-zero rate" proxies.

  * Input: every pixel is a Bernoulli spike source with probability equal to its intensity p in
    [0, 1].  The network was trained on standardised pixels x = (p - mu) / sd, so that affine map is
    folded into the first layer once (W' = W / sd, b' = b - W mu / sd); the expected input current
    per step is then exactly the rate network's first-layer drive, and the 81% of MNIST pixels that
    are zero never emit a spike -- the input layer becomes event-driven for free.
  * Hidden layers: non-leaky integrate-and-fire with soft reset (subtract threshold), thresholds
    from data-based normalisation (the 99.9th percentile of the rate activation on a calibration
    batch), and the layer's k-WTA kept as a per-step spiking budget: only the k largest membrane
    potentials above threshold may spike in a step, which is the PV-inhibition picture again.
    The homeostatic gain of the rate model is folded into the weights as well.
  * Output: the ten output currents are accumulated over the steps (no spiking); argmax decides.

Cost is counted as synaptic events: every spike costs one operation per outgoing synapse
(dense fan-out here; masked synapses are not counted when a connectivity mask is present).
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def calibrate_thresholds(net, X_std, q=0.999):
    """Per-layer firing thresholds: a high percentile of each hidden layer's rate activation."""
    _, a = net.forward(X_std)
    thr = [None]
    for l in range(1, net.L):
        v = a[l][a[l] > 0]
        thr.append(float(torch.quantile(v, q)) if v.numel() else 1.0)
    return thr


@torch.no_grad()
def spike_eval(net, X_raw, mu, sd, y, T_s, thr, seed=0, batch=2000):
    """Run the network on Bernoulli input spikes for T_s steps.  Returns (accuracy, per-layer spike
    counts per sample, synaptic events per sample)."""
    g = torch.Generator().manual_seed(seed)
    sd_t = torch.as_tensor(sd, dtype=torch.float32)
    mu_t = torch.as_tensor(mu, dtype=torch.float32)
    # fold standardisation and homeostatic gain into the first layer
    g1 = net.gain[1] if net.L > 1 else torch.ones(net.sizes[1])
    W1 = (g1[:, None] * net.W[1]) / sd_t[None, :]
    b1 = net.b[1] - W1 @ mu_t
    masks = getattr(net, "mask", None)
    correct, spikes, events = 0, np.zeros(net.L - 1), 0.0
    for i in range(0, len(y), batch):
        P = torch.as_tensor(X_raw[i:i + batch], dtype=torch.float32)
        n = P.shape[0]
        V = [None] + [torch.zeros(n, net.sizes[l]) for l in range(1, net.L)]
        out = torch.zeros(n, net.sizes[net.L])
        in_spikes = 0.0
        for _ in range(T_s):
            s_in = (torch.rand(P.shape, generator=g) < P).float()
            in_spikes += s_in.sum().item()
            cur = s_in @ W1.T + b1
            for l in range(1, net.L):
                V[l] = V[l] + cur
                fire = (V[l] >= thr[l]).float()
                if net.kwta:  # spiking budget: only the k largest suprathreshold potentials fire
                    k = net._k(net.sizes[l])
                    top = torch.zeros_like(fire)
                    idx = V[l].topk(k, dim=1).indices
                    top.scatter_(1, idx, 1.0)
                    fire = fire * top
                V[l] = V[l] - fire * thr[l]  # soft reset
                spikes[l - 1] += fire.sum().item()
                fan_out = net.sizes[l + 1] if masks is None or masks[l + 1] is None \
                    else float(masks[l + 1].sum(0).mean())
                events += fire.sum().item() * fan_out
                gl = net.gain[l + 1] if l + 1 < net.L else 1.0
                # with soft reset a spike stands for `thr` units of the rate activation, so that
                # spikes * thr ~ T_s * rate; every layer then integrates T_s times its rate drive
                cur = gl * ((fire * thr[l]) @ net.W[l + 1].T) + net.b[l + 1]
            out += cur  # output layer integrates its input current
        fan_in1 = net.sizes[1] if masks is None or masks[1] is None else float(masks[1].sum(0).mean())
        events += in_spikes * fan_in1
        correct += int((out.argmax(1).numpy() == y[i:i + batch]).sum())
    N = len(y)
    return correct / N, (spikes / N).tolist(), events / N
