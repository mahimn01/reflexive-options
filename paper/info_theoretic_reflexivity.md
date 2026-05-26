# Information-Theoretic Reflexivity (LaTeX-ready section)

This file is a self-contained, LaTeX-friendly writeup of the §3.10 Theorem-5 contribution, intended for direct integration into `paper/main.tex` (as either a new §3.10 immediately after the Hawkes-SV equivalence §3.9, or as Appendix B). All math uses `$...$` / `$$...$$`; citations use `\citep{...}` against the keys added to `paper/references.bib`; cross-references use `\cref{...}` against the existing label scheme.

Length: ~1100 words. Replaces nothing in the current main.tex; additive only.

---

## §3.10. Information-Theoretic Characterisation of the Critical Edge

\cref{thm:hopf} (Theorem~1) and \cref{thm:hawkes-sv} (Theorem~4) characterise the Hopf threshold $\kappa^\star$ spectrally: $\kappa^\star$ is the smallest $\kappa$ at which $\mathrm{Re}\,\lambda_{\max}(J(\kappa)) = 0$, equivalently $n_{\mathrm{SV}}(\kappa^\star) = 1$ in the Bacry--Delattre--Hoffmann--Muzy diffusive language \citep{bacrydelattrehoffmannmuzy2013, bacrymastromatteomuzy2015}. We now add an *information-theoretic* characterisation that complements these spectral statements: how many nats of information does the dealer-gamma channel leak from past spot to future returns?

### Setup.

Let $y_t = \log(S_t / S^\star)$ be the log-deviation of spot, $R_\tau := y_\tau - y_0 = \int_0^\tau dS_s/S_s$ the integrated future log-return, and $\mathcal{F}_{(-\infty, 0]}^y$ the past spot history. Define the *excess entropy at horizon $\tau$* (in the Crutchfield--Feldman \citep{crutchfieldfeldman2003} sense, conditioned on the present non-spot state):
\begin{equation}
E_\tau(\kappa) \;:=\; I\bigl(\mathcal{F}_{(-\infty, 0]}^y;\, R_\tau \,\bigm|\, v_0, z_0\bigr).
\label{eq:excess-entropy-def}
\end{equation}
Conditioning on $(v_0, z_0)$ removes the contribution of past spot already encoded in the present non-spot state, isolating the *direct* information flow through the dealer-gamma drift $\kappa G(\cdot)$. At $\kappa = 0$ the SDE \eqref{eq:reflexive-sde} collapses to standard Heston with Markov closure: future returns are $v_0$-conditional independent of past spot, so $E_\tau(0) = 0$. For $\kappa > 0$, past spot enters via $G$, and $E_\tau(\kappa) > 0$.

For the linearised 3D OU $dx = J(\kappa) x\,dt + \Sigma\,dW$ around the equilibrium (with the constant-vol surrogate of §3.4 and $\Sigma = \mathrm{diag}(\sqrt{\theta_v}, \xi\sqrt{\theta_v}, 0)$), Gaussian conditioning + the Markov property of the full 3D state give the closed form (proof in \cref{app:excess-entropy-proof}):
\begin{equation}
E_\tau(\kappa) \;=\; \tfrac{1}{2}\,\log\!\left(1 \;+\; \frac{v_1^2 \cdot \sigma^2_{y\mid u, z}}{m_y(\tau)}\right),
\label{eq:excess-entropy-closed-form}
\end{equation}
with
\begin{align*}
v_1 &:= \bigl[e^{J(\kappa)\tau} - I\bigr]_{11}, \\
\sigma^2_{y\mid u, z} &:= P_{11} - P_{1,(2,3)}\, P_{(2,3),(2,3)}^{-1}\, P_{(2,3),1}, \\
m_y(\tau) &:= \bigl[P - e^{J\tau} P\, e^{J^\top \tau}\bigr]_{11},
\end{align*}
where $P$ solves the Lyapunov equation $J P + P J^\top + \Sigma\Sigma^\top = 0$ (well-defined whenever $J(\kappa)$ is Hurwitz, i.e. for $\kappa \in (0, \kappa^\star)$).

### Theorem 5.

\begin{theorem}[Critical excess entropy at the Hopf boundary]\label{thm:excess-entropy}
Assume the conditions of \cref{thm:hopf} hold, so $J(\kappa)$ is Hurwitz on $(\kappa_{\mathrm{NS}}, \kappa^\star)$ with a complex pair $\lambda_\pm(\kappa) = \alpha(\kappa) \pm i\omega(\kappa)$ crossing the imaginary axis at $\kappa^\star$ with $\partial\alpha/\partial\kappa\rvert_{\kappa^\star} > 0$. Let $E_\tau(\kappa)$ be defined by \eqref{eq:excess-entropy-def} and computed via \eqref{eq:excess-entropy-closed-form}. Then for any fixed $\tau > 0$:
\begin{enumerate}
\item (Markov closure) $\displaystyle\lim_{\kappa \to 0^+} E_\tau(\kappa) = 0$.
\item (Finite saturation at criticality) $E_\tau(\kappa^\star) := \lim_{\kappa \uparrow \kappa^\star} E_\tau(\kappa)$ exists and is finite, with $E_\tau(\kappa^\star) > 0$.
\item (Mean-field critical exponent $\beta = 1$) The approach to the saturation is linear,
\[
E_\tau(\kappa^\star) - E_\tau(\kappa) \;=\; C_\tau \cdot (\kappa^\star - \kappa) \;+\; O\bigl((\kappa^\star - \kappa)^2\bigr),
\]
with $C_\tau > 0$.
\item (Local monotonicity) There exists $\delta > 0$ such that $\partial E_\tau / \partial \kappa > 0$ on $(\kappa^\star - \delta, \kappa^\star)$.
\end{enumerate}
\end{theorem}

\begin{proof}[Proof sketch]
\emph{(1)} The first row of $J(0)$ is identically zero in the constant-vol surrogate, so $e^{J(0)\tau}$ has first row $(1, 0, 0)$, giving $v_1 = 0$ and $E_\tau(0) = 0$. Continuity in $\kappa$ extends this to the limit.

\emph{(2)} As $\kappa \uparrow \kappa^\star$ the leading complex pair's real part $\alpha(\kappa) \to 0$ and $\lVert P(\kappa) \rVert \to \infty$ in the direction of the slow-mode eigenvector $q = (q_y, q_u, q_z)$, with $\lVert P \rVert \sim 1/|\alpha(\kappa)|$. Crucially, this divergence is *coherent* across $(y, u, z)$ — it lives in a 1D subspace. The Schur complement $\sigma^2_{y\mid u, z}$ extracts the component orthogonal to $\mathrm{span}\{e_2, e_3\}$ in the metric induced by the bounded part of $P$; standard low-rank Schur-complement asymptotics \citep[§3]{carlson1986} give the closed-form limit
\[
\sigma^2_{y\mid u, z}(\kappa^\star) \;=\; \frac{q_y^2}{q_u^2 \beta_u + q_z^2 \beta_z}
\]
for positive constants $\beta_u, \beta_z$ determined by the bounded part of $P$. Both $v_1$ and $m_y(\tau)$ stay bounded as $\alpha \to 0$ at fixed $\tau$, so $E_\tau(\kappa^\star) < \infty$.

\emph{(3)} Each ingredient in \eqref{eq:excess-entropy-closed-form} is real-analytic in $\kappa$ on a punctured neighbourhood of $\kappa^\star$ (the Lyapunov solution is analytic in any Hurwitz matrix, \citealp{bhatia1997}). A first-order Taylor expansion of the smooth function $\log(1 + v_1^2 \sigma^2/m)$ gives the linear scaling claimed.

\emph{(4)} The leading coefficient $C_\tau$ in (3) is positive because (i) $\sigma^2_{y\mid u, z}$ increases as $\delta = \kappa^\star - \kappa \to 0$ (the sub-leading Lyapunov terms $\beta_u, \beta_z$ decrease with $\delta$), and (ii) $v_1, m_y(\tau)$ have only sub-leading corrections. The ratio $v_1^2 \sigma^2/m$ is monotonically increasing in $\kappa$ on $(\kappa^\star - \delta, \kappa^\star)$.
\qed
\end{proof}

\textbf{Honest scope.} The proof gives \emph{local} monotonicity on a one-sided neighbourhood of $\kappa^\star$. The \emph{global} monotonicity claim $\partial E_\tau/\partial\kappa \geq 0$ on the entire stable interval $(0, \kappa^\star)$ is checked numerically (see below) but is not asserted as part of \cref{thm:excess-entropy} — the sign argument relies on Schur-complement asymptotics that do not extend across the node-spiral transition $\kappa_{\mathrm{NS}}$.

### Numerical anchor — §3.4 canonical regime.

Implementation: \texttt{src/reflexive\_options/theory/info\_theoretic.py}. Evaluated on a 101-point $\kappa$-grid over $(10^{-4}, \kappa^\star)$ at the §3.4 regime with the canonical Heston diffusion ($\theta_v = 0.04$, $\xi = 0.3$):

\begin{table}[h]
\centering
\begin{tabular}{lrrr}
\toprule
Quantity & $\tau = 0.1$~yr & $\tau = 1$~yr & $\tau = 5$~yr \\
\midrule
$E_\tau(10^{-4})$ (Markov anchor) & $2.5 \times 10^{-10}$ & $2.5 \times 10^{-9}$ & $1.3 \times 10^{-8}$ \\
$E_\tau(\kappa^\star^-)$ (saturation) & $0.0168$ & $0.0530$ & $0.4135$ \\
Ratio (crit-edge enhancement) & $\sim 2 \times 10^8$ & $\sim 2 \times 10^8$ & $\sim 3 \times 10^8$ \\
Global monotone? & yes & yes & yes \\
Fitted $\hat\beta$ at boundary & $0.998$ & $0.998$ & $1.000$ \\
\bottomrule
\end{tabular}
\caption{Numerical anchor for \cref{thm:excess-entropy} at the §3.4 canonical regime. Critical exponent $\hat\beta \approx 1$ matches the mean-field prediction. Figure: \texttt{paper/figures/excess\_entropy\_curve.pdf}.}
\end{table}

### Structural insight — why $E_\tau(\kappa^\star)$ does NOT diverge.

The naïve intuition from the Crutchfield--Feldman \citep{crutchfieldfeldman2003, crutchfield2012complexity} statistical-complexity-at-criticality literature would predict $E_\tau(\kappa^\star) = \infty$. This fails in our setting for a structural reason: the slow-mode collapse at $\kappa^\star$ is *coherent across $(y, u, z)$* — it lives in a 1D eigenvector subspace spanned by $q = (q_y, q_u, q_z)$. Conditioning on the present $(u_0, z_0)$ projects out precisely this coherent mode; what remains is the orthogonal component, governed by the bounded part of $P$. The saturation $E_\tau(\kappa^\star)$ is therefore capped by the eigenvector's "spot-purity ratio" $q_y^2 / (q_u^2 + q_z^2)$ at the boundary — a parametric prediction tying qualitative critical behaviour to spot-vs-vol mode partition.

### Phase-4 testable prediction.

\begin{corollary}\label{cor:transfer-entropy}
For an SPX market window with calibrated dealer-gamma series $\hat G_t$ and observed log-returns $\hat r_{t+1}$, the empirical Schreiber-2000 \citep{schreiber2000transferentropy} transfer entropy
\[
\hat T_{G \to r} \;:=\; \widehat{H(r_{t+1} \mid r_t)} \;-\; \widehat{H(r_{t+1} \mid r_t, G_t)}
\]
should be statistically significant ($p < 0.05$) under an IAAFT-surrogate null on the source series, AND should be larger on event windows where the system is conjectured to sit closer to $\kappa^\star$ (Volmageddon Feb~2018, COVID Mar~2020, Yen carry Aug~2024) than on quiescent windows.
\end{corollary}

The corollary has two pieces — (i) directional significance under IAAFT, (ii) event-window dependence — logically independent of \cref{thm:hawkes-sv}'s $n_{\mathrm{SV}}(\kappa_0) \approx 1$ prediction. The IAAFT null preserves $G$'s marginal and linear ACF while randomising its nonlinear cross-coupling to $r$ (matching the pre-registration amendment A5 detector convention), so the test asks whether the nonlinear feedback channel is informative beyond what $G$'s own autocorrelation explains. Implementation: \texttt{transfer\_entropy\_iaaft\_pvalue} in \texttt{info\_theoretic.py}.

### Relation to \cref{thm:hawkes-sv}.

\cref{thm:hawkes-sv} characterises $\kappa^\star$ spectrally via $n_{\mathrm{SV}}(\kappa^\star) = 1$. \cref{thm:excess-entropy} characterises the *same* boundary information-theoretically via the finite saturation $E_\tau(\kappa^\star) > 0$. The two are independent characterisations, not consequences of each other: $n_{\mathrm{SV}}$ depends only on the leading eigenvalue of $J(\kappa)$, while $E_\tau$ depends on the full Lyapunov-equation solution and the noise covariance. Two markets with identical $\lambda_{\max}(\kappa)$ but different noise structures will have different $E_\tau$ curves. The two theorems are therefore complementary diagnostics: \cref{thm:hawkes-sv} measures *how close to the boundary* the system is; \cref{thm:excess-entropy} measures *how much past-spot information* the system actually leaks at that distance.

Relation to \citep{lizier2012local}'s "active information storage" framework: the conditioning on $(v_0, z_0)$ in \eqref{eq:excess-entropy-def} is precisely Lizier--Prokopenko--Zomaya's "fixed-history baseline" — \cref{thm:excess-entropy} can be read as a Hopf-boundary application of that framework to a continuous-time SDE channel.
