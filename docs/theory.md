# Theoretical Background for Qubit-Native Weak-Measurement Entanglement Estimation

This document summarizes the mathematical model implemented in the repository. It is intended to be a stable public technical companion to the code and artifacts: it defines the benchmark state family, the three-qubit weak-measurement protocol, the postselected feature map, the benchmark-family identifiability argument, and the assumptions behind the conformal uncertainty layer.

Operational run history, queue notes, and environment-specific debugging details are intentionally kept out of this file. Public execution outcomes are summarized in `artifacts/summary_sim.md` and `artifacts/summary_ibm.md`.

## 1. Problem Setup and Notation

The repository studies two-qubit entanglement estimation using a qubit-native weak-measurement protocol. Qubits `A` and `B` carry the target state, and qubit `P` acts as a discrete pointer ancilla.

The benchmark family is
\[
\rho_{AB}(p,\theta)=p|\psi_\theta\rangle\langle\psi_\theta|+(1-p)\frac{I_A}{2}\otimes\rho_B^\theta,
\quad
|\psi_\theta\rangle=\cos\theta|00\rangle+\sin\theta|11\rangle.
\]

The primary target is concurrence `C`, with negativity `N` reported as a secondary entanglement summary in the repository outputs. A single measurement design setting is written as
\[
s = (g,\mathrm{basis}_P,\phi_B),
\]
where `g` is the interaction strength, `\mathrm{basis}_P` is the pointer readout basis, and `\phi_B` is an optional local rotation on qubit `B`.

## 2. Qubit-Native Weak-Measurement Protocol

For a fixed design setting `s`, the experiment proceeds as follows.

1. Prepare `\rho_{AB}(p,\theta)` on system qubits `A` and `B`.
2. Initialize the pointer ancilla `P` in `|0\rangle`.
3. Apply the weak interaction
   \[
   U(g)=\exp\left(-i\frac{g}{2}X_B\otimes Y_P\right).
   \]
4. Apply a local basis change on `A` with `H_A`, then measure `A` in the computational basis and postselect on `A=0`. In the original basis, this corresponds to postselection onto `|+\rangle`.
5. Optionally rotate `B` by `R_y(\phi_B)` before measuring `B` in the computational basis. The default workflow uses `\phi_B = 0`.
6. Rotate the pointer into the requested readout basis before terminal Z measurement:
   - `basis_P = Z`: no pre-rotation,
   - `basis_P = X`: apply `H`,
   - `basis_P = Y`: apply `S^\dagger` followed by `H`.

For Runtime and Aer execution, the interaction is implemented through a standard basis-change decomposition:

- map `X_B \otimes Y_P` into `Z_B \otimes Z_P`,
- apply `CX(B,P) -> RZ(g)_P -> CX(B,P)`,
- undo the basis change.

This provides a gate-model realization of weak-to-strong coupling as `g` sweeps from `0` toward `\pi/2`.

## 3. Postselected Observables and Feature Construction

From terminal counts over `(A,B,P)`, the repository extracts two classes of observables.

First, it records postselection rates:
\[
P(A=0,B=0),\qquad P(A=0,B=1).
\]

Second, it computes conditional pointer expectations in the rotated pointer basis:
\[
m_k^{(X)} = \mathbb E[(-1)^P\mid A=0,B=k,\text{pointer in X}],
\]
\[
m_k^{(Y)} = \mathbb E[(-1)^P\mid A=0,B=k,\text{pointer in Y}],
\qquad k\in\{0,1\}.
\]

These quantities are qubit-pointer analogs of the real and imaginary weak-value quadratures used in the earlier photonic setting. Across a design grid, the resulting statistics are concatenated into the feature representation consumed by the estimator and calibration layers.

## 4. Identifiability on the Benchmark Family

The benchmark family is useful because the postselected local state on `B` admits a simple closed form. Define
\[
\Gamma_A = |+\rangle\langle+|,\qquad
\rho_B^{(+)} = \frac{\langle +|\rho_{AB}|+\rangle}{\operatorname{Tr}(\langle +|\rho_{AB}|+\rangle)}.
\]

For the family above,
\[
\rho_B^{(+)} =
\begin{pmatrix}
\cos^2\theta & p\cos\theta\sin\theta\\
p\cos\theta\sin\theta & \sin^2\theta
\end{pmatrix}.
\]

Its matrix elements are therefore
\[
\rho_{00}=\cos^2\theta,\qquad
\rho_{11}=\sin^2\theta,\qquad
\rho_{10}=\rho_{01}=p\cos\theta\sin\theta.
\]

Using the weak-value-style ratios
\[
w_0 = \frac{\rho_{10}}{\rho_{00}},\qquad
w_1 = \frac{\rho_{01}}{\rho_{11}},
\]
one obtains
\[
w_0 = p\tan\theta,\qquad
w_1 = p\cot\theta.
\]

Hence
\[
p = \sqrt{w_0 w_1},\qquad
\tan^2\theta = \frac{w_0}{w_1}.
\]

Except on degenerate boundaries, `(p,\theta)` is therefore identifiable from postselected local weak information. Once `(p,\theta)` is identified, the entanglement targets reported by the repository are determined by the standard calculations implemented in `src/metrics.py`.

This is the core algebraic reason the benchmark family is a sensible testbed for compressed entanglement estimation.

## 5. Finite-Sample Uncertainty via Conformal Calibration

The repository does not stop at a point estimate. It also constructs calibrated uncertainty intervals for concurrence using split conformal prediction.

For a fixed predictor `\hat f` and a calibration set of size `n_{cal}`, define residual scores
\[
r_i = |y_i-\hat f(x_i)|.
\]

Let
\[
q_{1-\alpha}=\text{the }\left\lceil (n_{cal}+1)(1-\alpha)\right\rceil\text{-th order statistic of }\{r_i\}.
\]

The split-conformal interval is then
\[
\hat C(x) \pm q_{1-\alpha},
\]
which has marginal coverage at least `1-\alpha` under exchangeability of the train, calibration, and test samples.

The repository also supports a locally scaled variant with score
\[
r_i^{(s)} = \frac{|y_i-\hat f(x_i)|}{\hat s(x_i)},
\]
leading to interval half-width
\[
q_{1-\alpha}^{(s)}\hat s(x).
\]

The key public claim here is limited and precise: the guarantee is finite-sample and marginal, not conditional, and it depends on the exchangeability assumption remaining plausible in deployment.

## 6. Assumptions, Regime of Validity, and Failure Modes

The public interpretation of the repository should be read with the following assumptions in mind.

- Calibration and test samples should be exchangeable if the conformal intervals are to retain their nominal marginal validity.
- The postselection events `A=0,B=k` must carry enough probability mass for the conditional moments to be numerically stable.
- The implemented coupling strength `g` and basis rotations must match the intended circuit model closely enough for the estimator to transfer.

Within that regime, the main failure modes are straightforward.

- **Low postselection probability:** `m_k^{(X)}` and `m_k^{(Y)}` become unstable when the conditioning event is too rare.
- **Interaction miscalibration:** a mismatch between intended and realized `g` breaks the assumed measurement model.
- **Hardware drift:** nonstationary readout or gate errors can degrade transfer from simulation-trained models to device data.
- **Domain shift:** if the deployment distribution differs materially from the calibration distribution, empirical coverage can drop below nominal levels.
- **Ambiguous inverse map:** if different parameter settings produce similar weak-measurement features, the posterior can become broad or multimodal.

These are not edge cases to hide; they are the reasons the repository also exposes calibration, shift, and abstention diagnostics.

## 7. Reliability Logic and Abstention

The implementation includes an explicit abstention pathway. In practical terms, the system can decline to report a confident entanglement estimate when either of the following happens:

- the conformal interval half-width exceeds a configured tolerance, or
- posterior or shift diagnostics indicate unreliable inference, for example through large KL shift or strongly multimodal particle beliefs.

This turns uncertainty from a reporting afterthought into part of the experimental logic. On real hardware, that distinction matters: a method that knows when it should not trust its own output is more useful than one that produces a narrow but unearned answer.
