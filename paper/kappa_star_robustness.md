# Robustness of $\kappa^\star$ to OI distribution mis-specification

*Self-contained writeup for integration into `paper/main.tex`. Suggested home: a new §3.6 "Calibration tolerance for the closed-form Hopf threshold", or as an extension of §3.5 immediately after the closed-form Eq. 18. Companion code: `src/reflexive_options/theory/robustness.py`; figures: `paper/figures/kappa_star_robustness_heatmap.pdf`, `paper/figures/kappa_star_misspecification_curve.pdf`; experiment: `src/reflexive_options/experiments/kappa_star_robustness.py`.*

The closed-form Hopf threshold (paper §3.5, Eq. 18)

$$
\kappa^\star(G_y, G_v) \;=\; \frac{G_y A^2 - G_v L \;-\; \sqrt{(G_v L - G_y A^2)^2 - 4 G_y^2 A (M A - L/2)}}{2 G_y^2 A},
\qquad A := \alpha + \kappa_v,\; M := \alpha\kappa_v,\; L := \beta\gamma,
$$

assumes a log-normal open-interest density $q(\log K) = \mathcal{N}(\mu_q, \sigma_q^2)$. Phase-4 calibration estimates $(\hat\mu_q, \hat\sigma_q)$ from the empirical SPX OI grid; the resulting predicted $\kappa^\star$ is only as accurate as the calibration. We quantify the robustness in three steps.

### A. Analytical elasticities at the canonical regime

Implicit differentiation of $H(\kappa^\star; G_y, G_v) = 0$ on the closed-form Routh–Hurwitz quadratic yields

$$
\frac{\partial \kappa^\star}{\partial G_y} \;=\; -\frac{2 G_y A \kappa^{\star 2} - A^2 \kappa^\star}{2 A_2 \kappa^\star + A_1},
\qquad
\frac{\partial \kappa^\star}{\partial G_v} \;=\; -\frac{L \kappa^\star}{2 A_2 \kappa^\star + A_1}, \tag{R1}
$$

with $A_2 = G_y^2 A$, $A_1 = G_v L - G_y A^2$. Composing with the analytic OI-parameter partials of the closed-form $G$-derivatives (computed exactly via `G_lognormal_oi_partials` and 5-point central FD on $(\mu_q, \sigma_q)$ — both pieces smooth, machine-precision agreement):

$$
\frac{\partial \kappa^\star}{\partial \mu_q}
\;=\; \frac{\partial \kappa^\star}{\partial G_y}\frac{\partial G_y}{\partial \mu_q}
+ \frac{\partial \kappa^\star}{\partial G_v}\frac{\partial G_v}{\partial \mu_q},
\qquad
\frac{\partial \kappa^\star}{\partial \sigma_q}
\;=\; \frac{\partial \kappa^\star}{\partial G_y}\frac{\partial G_y}{\partial \sigma_q}
+ \frac{\partial \kappa^\star}{\partial G_v}\frac{\partial G_v}{\partial \sigma_q}. \tag{R2}
$$

At the canonical specification $(\mu_q = \log 100,\, \sigma_q = 0.10,\, T_{\mathrm{eff}} = 0.25,\, \kappa_v = 2,\, \alpha = 0.05,\, \beta = 1,\, \gamma = 1,\, a^\star = \log 100,\, v^\star = 0.04)$ the result is:

| Quantity | Value |
|---|---|
| $\kappa^\star$ | $17.806$ |
| $\omega^\star$ | $1.177$ rad/yr |
| $G_y$ | $-3.524 \times 10^{-2}$ |
| $G_v$ | $-1.769 \times 10^{-1}$ |
| $\partial \kappa^\star/\partial \sigma_q$ | $-281.85$ |
| $\partial \kappa^\star/\partial \mu_q$ | $+2{,}717.58$ |
| $\eta_{\sigma_q} := (\sigma_q/\kappa^\star) \partial \kappa^\star/\partial \sigma_q$ | $\boxed{\;-1.583\;}$ |
| $\eta_{\mu_q} := (\mu_q/\kappa^\star) \partial \kappa^\star/\partial \mu_q$ | $\boxed{\;+702.83\;}$ |
| $(1/\kappa^\star) \partial \kappa^\star/\partial \mu_q$ (per absolute log-strike unit) | $+152.6$, i.e. $+15{,}262\%$ per log-strike unit |

**Interpretation.** $|\eta_{\sigma_q}| = 1.58$ means a $1\%$ increase in $\sigma_q$ changes $\kappa^\star$ by roughly $1.6\%$ — moderate sensitivity. The $\mu_q$ elasticity looks dramatic only because $\mu_q$ is dimensional (a log-strike level $\sim \log 100 \approx 4.6$); the operative number for Phase-4 is the absolute partial $\partial\kappa^\star/\partial\mu_q \approx 2{,}717$, equivalently $1/\kappa^\star \cdot \partial\kappa^\star/\partial\mu_q \approx 152.6$ per log-strike unit. A mis-centered OI mean by $0.001$ log-strike units shifts $\kappa^\star$ by $\sim\!15\%$. Validated by 4th-order central FD on $\kappa^\star(\mu_q, \sigma_q)$ to $<10^{-4}$ relative (`tests/test_kappa_star_robustness.py::test_dkappa_d{muq,sigma_q}_matches_richardson_fd`).

### B. Numerical sensitivity surface

Sweeping $(\mu_q, \sigma_q)$ over $\pm 30\%$ of canonical (so $\sigma_q \in [0.07, 0.13]$ and $\mu_q \in [3.224, 5.987]$ log-strike) and computing $\kappa^\star$ at each cell via Eq. 18 yields a quasi-monotone surface (Fig.~\ref{fig:kappa-star-robustness-heatmap}). Tracing iso-contours of fractional deviation $|\kappa^\star/\kappa^\star_{\mathrm{canon}} - 1|$:

| Sweep direction | $\pm 10\%$ | $\pm 30\%$ |
|---|---|---|
| $\sigma_q$-only | $|\Delta \kappa^\star/\kappa^\star| \approx 19.6\%$ | $\approx 48.0\%$ |
| $\mu_q$-only (at canonical $\sigma_q$) | $\gg 100\%$ — the surface is essentially vertical in $\mu_q$ | $\gg 100\%$ |

The $\mu_q$ direction is much steeper than $\sigma_q$: a $30\%$ shift in $\mu_q$ exits the supercritical Hopf region entirely (cells return NaN — no positive Hopf root). For practical purposes the Phase-4 calibration tolerance is dominated by the $\mu_q$ axis.

### C. Mis-specification: what if true OI is not log-normal?

Empirical SPX OI is not log-normal — concentration at round-number strikes and calendar-effect spikes near ATM produce multi-modal grids. We model the worst-case via a symmetric mixture of two log-normals at $\mu_q \pm \Delta/2$ with equal weight and component spread $\sigma_{\mathrm{comp}} = 0.07$ (so the moment-matched single log-normal sits at $\hat\mu = \mu_q$, $\hat\sigma = \sqrt{\sigma_{\mathrm{comp}}^2 + (\Delta/2)^2}$, ranging from $0.070$ at $\Delta = 0$ to $0.260$ at $\Delta = 0.50$). For each separation $\Delta$ we compute (i) $\kappa^\star_{\mathrm{cf}}$ via the closed form at $(\hat\mu, \hat\sigma)$, and (ii) $\kappa^\star_{\mathrm{true}}$ via numerical FD on the Jacobian of the deterministic skeleton with $G$ built by direct quadrature against the actual mixture density (`kappa_star_brute_force_from_G`). The relative error $|\kappa^\star_{\mathrm{cf}} - \kappa^\star_{\mathrm{true}}|/\kappa^\star_{\mathrm{true}}$ as a function of $\Delta$ is shown in Fig.~\ref{fig:kappa-star-misspec-curve}; selected anchors:

| Bimodal separation $\Delta$ (log-strike) | $\hat\sigma$ | $\kappa^\star_{\mathrm{cf}}$ | $\kappa^\star_{\mathrm{true}}$ | Rel.\ error |
|---:|---:|---:|---:|---:|
| $0.00$ (single mode) | $0.070$ | $26.36$ | $26.36$ | $0.01\%$ |
| $0.05$ | $0.074$ | $25.20$ | $25.14$ | $0.23\%$ |
| $0.10$ | $0.086$ | $21.87$ | $20.86$ | $4.83\%$ |
| $0.20$ | $0.122$ | $12.46$ | $5.68$ | $119.5\%$ |
| $0.30$ | $0.166$ | $7.81$ | $2.46$ | $217.7\%$ |

**Headline finding.** The closed form is robust ($\le 5\%$ error) for bimodal separations up to $\Delta \sim 0.10$ log-strike units (corresponding to a relative strike-range of $\sim 10\%$). Beyond $\Delta \approx 0.15$, the moment-matched single log-normal under-spreads the bimodal mass (the BS-gamma kernel concentrates the mass around the modes, not around the moment-matched mean) and the closed-form $\kappa^\star_{\mathrm{cf}}$ overestimates the true threshold by a factor of $2$–$3$. **The closed form is fragile to severe bimodality.** For empirical SPX, where calendar-effect bimodality is typically $\Delta \lesssim 0.05$–$0.10$ (3M monthly OI clusters around ATM and the next quarterly), the closed form is in the safe zone.

### D. Phase-4 calibration tolerance recommendation

Combining (R2) with an independent-error budget allocation $\tfrac{1}{\sqrt 2}$ to each of $\sigma_q$ and $\mu_q$:

$$
\Bigl(\frac{\Delta\kappa^\star}{\kappa^\star}\Bigr)^2 \;=\; \eta_{\sigma_q}^2 \Bigl(\frac{\Delta\sigma_q}{\sigma_q}\Bigr)^2
\;+\; \Bigl(\frac{1}{\kappa^\star}\frac{\partial\kappa^\star}{\partial\mu_q}\Bigr)^2 (\Delta\mu_q)^2,
$$

at the canonical regime, the calibration tolerance for a target $|\Delta\kappa^\star/\kappa^\star|$ budget is:

| Target $|\Delta\kappa^\star/\kappa^\star|$ | $\sigma_q$ tolerance | $\mu_q$ tolerance (log-strike) |
|---:|---:|---:|
| $5\%$ | $\pm 2.23\%$ relative | $\pm 0.00023$ |
| $10\%$ | $\pm 4.47\%$ relative | $\pm 0.00046$ |
| $25\%$ | $\pm 11.17\%$ relative | $\pm 0.00116$ |

In words: **to predict $\kappa^\star$ within $\pm 10\%$ at Phase 4, we need to estimate $\sigma_q$ to $\pm 4.5\%$ relative and the OI mean $\mu_q$ to $\pm 5 \times 10^{-4}$ log-strike units (i.e.\ $\pm 0.05\%$ relative strike error).** The $\mu_q$ tolerance is the binding constraint and corresponds to centering the empirical OI mean on ATM to within $\sim 5$bp — well within the precision of an SPX OI grid quoted on a $5$-strike-wide ladder. The $\sigma_q$ tolerance ($\pm 4.5\%$ relative, i.e.\ a $\sigma_q$ of $0.0955$–$0.1045$ at canonical $0.10$) is the harder calibration target; for bimodal SPX OI the moment-matched $\hat\sigma_q$ has a structural bias that grows with the bimodality severity (Section C), so the $\pm 10\%$ budget should be reduced accordingly when the empirical OI shows clear bimodality at $\Delta > 0.05$.

### Reproducibility

```bash
python -m reflexive_options.experiments.kappa_star_robustness          # full sweep
python -m reflexive_options.experiments.kappa_star_robustness --quick  # CI smoke test
pytest tests/test_kappa_star_robustness.py                              # 9 tests
```

The figures `paper/figures/kappa_star_robustness_heatmap.pdf` and `paper/figures/kappa_star_misspecification_curve.pdf` are byte-stable under `SOURCE_DATE_EPOCH=0`. Metrics persist to `runs/kappa_star_robustness/<ts>/metrics.json`.
