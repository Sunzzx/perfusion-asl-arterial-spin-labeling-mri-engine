# Arterial Spin Labeling (ASL) MRI Perfusion Engine

[![CI Pipeline](https://github.com/Sunzzx/perfusion-asl-arterial-spin-labeling-mri-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Sunzzx/perfusion-asl-arterial-spin-labeling-mri-engine/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-pink.svg)](https://github.com/sponsors/Sunzzx)
[![Commercial License](https://img.shields.io/badge/Commercial%20License-Buy%20Now-green.svg)](https://buy.stripe.com/test_perfusion_asl)

> **Quantitative Hemodynamic Neuroimaging**: Buxton General Kinetic Model (GKM), Multi-Post-Label-Delay (PLD) non-linear fitting, and digital neuro-perfusion phantom benchmarking for non-contrast Cerebral Blood Flow (CBF) and Arterial Transit Time (ATT) quantification.

---

## 🌍 Humanity & Real-World Clinical Impact

Stroke is the second-leading cause of death worldwide and a primary contributor to long-term adult disability. Rapid identification of the **ischemic penumbra**—salvageable brain tissue that is hypoperfused but not yet irreversibly infarcted—is decisive in guiding reperfusion therapies (such as intravenous thrombolysis and endovascular mechanical thrombectomy).

Conventional perfusion imaging (such as Dynamic Susceptibility Contrast MRI and CT Perfusion) mandates exogenous gadolinium or iodinated contrast agents. These pose severe risks to patients with acute renal impairment (inducing nephrogenic systemic fibrosis or contrast-induced nephropathy), pediatric patients, and those needing repeated longitudinal monitoring.

**Arterial Spin Labeling (ASL)** utilizes magnetically labeled endogenous blood water protons as a completely non-invasive, contrast-free tracer. This engine provides real-time quantitative deconvolution of Cerebral Blood Flow ($f$) and Arterial Transit Delay ($\Delta t$), enabling contrast-free acute stroke triage and neurovascular assessment at zero tracer expenditure.

---
## 📐 Theoretical Foundations & Mathematical Formulations

### 1. General Kinetic Model (Buxton GKM)

The difference magnetization $\Delta M(t) = M_{\text{control}}(t) - M_{\text{label}}(t)$ in pseudo-continuous ASL (pCASL) obeys the Buxton convolution integral:

$$
\Delta M(t) = 2 M_0 \alpha f \int_0^t c(t') r(t - t') m(t - t') dt'
$$

Where:
- $M_0$: Equilibrium longitudinal magnetization of blood
- $\alpha$: Inversion labeling efficiency ($\approx 0.85$ for pCASL)
- $f$: Cerebral Blood Flow (CBF) in units of $\text{mL}/(\text{g} \cdot \text{s})$
- $c(t)$: Arterial input bolus delivery function
- $r(t) = \exp(-t / T_{1,\text{tissue}})$: Residue wash-out function
- $m(t) = \exp(-t / T_{1,\text{blood}})$: Longitudinal relaxation of blood water

Under standard blood-brain barrier partition assumptions, the piecewise analytic solution for post-label delay (PLD) $w$ and labeling duration $\tau$ is:

$$
\Delta M(w) = \begin{cases} 
0, & w < \Delta t \\ 
2 M_0 \alpha f T_{1,\text{blood}} e^{-\Delta t / T_{1,\text{blood}}} \left(1 - e^{-(w - \Delta t) / T_{1,\text{blood}}}\right), & \Delta t \le w \le \Delta t + \tau \\ 
2 M_0 \alpha f T_{1,\text{blood}} e^{-\Delta t / T_{1,\text{blood}}} \left(1 - e^{-\tau / T_{1,\text{blood}}}\right) e^{-(w - \Delta t - \tau) / T_{1,\text{blood}}}, & w > \Delta t + \tau
\end{cases}
$$

### 2. Multi-Delay Non-Linear Inversion

Given a sequence of measurements $\{w_k, \Delta M_k\}_{k=1}^K$, the engine performs joint parameter estimation for $[f, \Delta t]$ via Levenberg-Marquardt non-linear least-squares minimization:

$$
\min_{f, \Delta t} \sum_{k=1}^K \left| \Delta M_k - \Delta M(w_k; f, \Delta t) \right|^2
$$

---

## ⚡ Algorithmic Complexity

- **Single Voxel Multi-PLD Inversion**: $\mathcal{O}(K \cdot I)$ where $K$ is the number of PLDs ($K \sim 5-10$) and $I$ is iteration count ($I \le 20$).
- **Full 3D Brain Volume**: $\mathcal{O}(N_x \cdot N_y \cdot N_z \cdot K \cdot I)$ with spatial independence enabling perfect embarrassed parallelization across multi-core processors.
- **Space Complexity**: $\mathcal{O}(N_x \cdot N_y \cdot N_z \cdot K)$ memory footprint.

---

## 🚀 Quickstart & Interactive Demo

```bash
# Install dependencies
pip install -e .

# Run comprehensive test suite
pytest -q

# Run ASCII Perfusion Phantom Demo
python -m asl_mri.demo
```

---

## 💼 Enterprise Support & Commercial Licensing
For clinical trial integration, OEM MRI console firmware plugins, or custom DICOM pipeline support:
- **Commercial License**: [Stripe Checkout](https://buy.stripe.com/test_perfusion_asl)
- **GitHub Sponsorship**: [Sponsor @Sunzzx](https://github.com/sponsors/Sunzzx)
- **Enterprise Contact**: `ayufrom03@gmail.com`
