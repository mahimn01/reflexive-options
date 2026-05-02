# Abstract

We study a continuous-time Heston-like stochastic-volatility model in which
the underlying spot price is coupled to its own option market through a
dealer-gamma feedback channel, derived from the open-interest grid via a
Gârleanu–Pedersen–Poteshman (2009, *Review of Financial Studies* 22(10),
4259–4299) demand-pressure mapping. The model state is three-dimensional —
spot $S_t$, instantaneous variance $v_t$, and a low-pass-filtered log-spot
memory variable $z_t$ — and reduces to standard time-dependent Heston
(Heston 1993, *RFS* 6(2), 327–343) at zero coupling. Our contribution is
the *synthesis* of three previously separate ingredients into a single
analytical and computational object: (i) a continuous-time SV backbone with
explicit dealer-gamma drift, (ii) a formal Hopf bifurcation calculus on
the deterministic skeleton with first Lyapunov coefficient $\ell_1$ and
Khasminskii-style stochastic shift $\Lambda$, and (iii) a pre-registered
evaluation protocol for RL-trained hedging agents trained inside the
simulator. For a representative dimensionless calibration we obtain a
critical coupling $\kappa^\star \approx 0.8964$, angular frequency
$\omega^\star \approx 0.5724$ rad/yr, and first Lyapunov coefficient
$\ell_1 \approx -0.025 < 0$, so the bifurcation is supercritical: an
attracting limit cycle in volatility is born for $\kappa > \kappa^\star$.
The 2D reduction (without $z_t$) has an upper-triangular Jacobian and
admits only a saddle-node onset, recovering the discrete-time recursion of
Dai (arXiv:2511.22766, 2025); the 3D extension is what unlocks the Hopf.
The closest model-structure precedent (Halperin–Itkin Marketron,
arXiv:2508.09863, 2025) does not perform the bifurcation analysis we
provide. Each evaluation ingredient — sliced-Wasserstein-2 distance, block
bootstrap, arbitrage filter, commit-anchored pre-registration — is
borrowed; the configuration is, to our knowledge, not previously published
in this combination. A complete comparison against precedents is in
`related_work.md`; threats to validity in `threats_to_validity.md`.
