"""Buxton General Kinetic Model for Arterial Spin Labeling (ASL) MRI.

Implements the pulsed ASL (PASL) and pseudo-continuous ASL (pCASL) kinetic models
der standard blood-brain barrier permeability conditions according to:
Buxton et al., Magnetic Resonance in Medicine 40:383-396 (1998).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ASLTimingParams:
    tau: float = 1.8  # Labeling duration (seconds) - for pCASL or PASL
    t1_blood: float = 1.65  # Longitudinal relaxation time of blood at 3T (seconds)
    t1_tissue: float = 1.30  # Longitudinal relaxation time of grey matter at 3T (seconds)
    alpha: float = 0.85  # Inversion labeling efficiency (0.85 for pCASL, 0.98 for PASL)
    lambda_blood_brain: float = 0.90  # Blood-brain partition coefficient (mL/g)


@dataclass
class ASLTissueParams:
    cbf: float = 60.0  # Cerebral Blood Flow in mL / 100g / min
    att: float = 1.20  # Arterial Transit Time / Bolus Arrival Time in seconds


class BuxtonModel:
    """General Kinetic Model (GKM) for ASL Difference Signal $\Delta M(t)$."""

    def __init__(self, timing: ASLTimingParams = None):
        self.timing = timing or ASLTimingParams()

    def delta_m_pcasl(
        self, 
        post_label_delay: float | np.ndarray, 
        cbf: float | np.ndarray, 
        att: float | np.ndarray, 
        m0: float | np.ndarray = 1.0
    ) -> float | np.ndarray:
        """Compute pCASL difference magnetization signal \Delta M as a function of PLD.

        CBF is provided in mL / 100g / min. Internally converted to s^-1:
        f = (CBF / 6000.0) * lambda_blood_brain
        """
        pld = np.asarray(post_label_delay, dtype=float)
        # Convert CBF (mL/100g/min) to flow in (mL blood)/(mL tissue * s)
        # 1 mL/100g/min = 1 / (100 * 60) s^-1 = 1 / 6000 s^-1
        f = (cbf / 6000.0)
        alpha = self.timing.alpha
        t1_blood = self.timing.t1_blood
        tau = self.timing.tau

        # Standard Buxton / Consensus paper (Alsop et al. MRM 2015) equation:
        # For t = PLD + tau
        # When PLD < ATT (bolus has not arrived yet): signal = 0
        # When ATT <= PLD <= ATT + tau (during bolus delivery):
        #   Delta M = 2 * M0 * alpha * f * T1_blood * exp(-ATT / T1_blood) * (1 - exp(-(PLD - ATT) / T1_blood))
        # When PLD > ATT + tau (after bolus has cleared delivery):
        #   Delta M = 2 * M0 * alpha * f * T1_blood * exp(-ATT / T1_blood) * (1 - exp(-tau / T1_blood)) * exp(-(PLD - ATT - tau) / T1_blood)

        delta_m = np.zeros_like(pld, dtype=float)

        # Condition 1: PLD < ATT
        # delta_m remains 0

        # Condition 2: ATT <= PLD <= ATT + tau
        mask_mid = (pld >= att) & (pld <= att + tau)
        if np.any(mask_mid):
            pld_m = pld[mask_mid] if isinstance(pld, np.ndarray) else pld
            delta_m[mask_mid] = (
                2.0 * m0 * alpha * f * t1_blood * 
                np.exp(-att / t1_blood) * 
                (1.0 - np.exp(-(pld_m - att) / t1_blood))
            )

        # Condition 3: PLD > ATT + tau
        mask_late = pld > (att + tau)
        if np.any(mask_late):
            pld_l = pld[mask_late] if isinstance(pld, np.ndarray) else pld
            delta_m[mask_late] = (
                2.0 * m0 * alpha * f * t1_blood * 
                np.exp(-att / t1_blood) * 
                (1.0 - np.exp(-tau / t1_blood)) * 
                np.exp(-(pld_l - att - tau) / t1_blood)
            )

        return delta_m if isinstance(post_label_delay, np.ndarray) else float(delta_m)

    def single_delay_cbf_pcasl(
        self, 
        delta_m: float | np.ndarray, 
        pld: float, 
        m0: float | np.ndarray = 1.0
    ) -> float | np.ndarray:
        """Quantify CBF (mL/100g/min) from single-delay pCASL assuming PLD > ATT.
        
        Alsop et al. (2015) White Paper formula:
        CBF = (6000 * lambda * DeltaM * exp(PLD / T1_blood)) / 
              (2 * alpha * M0 * T1_blood * (1 - exp(-tau / T1_blood)))
        """
        t1b = self.timing.t1_blood
        lam = self.timing.lambda_blood_brain
        alpha = self.timing.alpha
        tau = self.timing.tau

        numerator = 6000.0 * lam * delta_m * np.exp(pld / t1b)
        denominator = 2.0 * alpha * m0 * t1b * (1.0 - np.exp(-tau / t1b))
        
        # Avoid zero division
        with np.errstate(divide='ignore', invalid='ignore'):
            cbf = np.where(denominator > 0, numerator / denominator, 0.0)
        return cbf if isinstance(delta_m, np.ndarray) else float(cbf)
