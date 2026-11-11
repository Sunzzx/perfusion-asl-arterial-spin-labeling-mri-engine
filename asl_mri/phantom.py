"""Synthetic Digital Brain Perfusion Phantom Generator for ASL MRI Benchmarking."""

import numpy as np
from .buxton import BuxtonModel, ASLTimingParams


class DigitalSheppLoganASLPhantom:
    """Constructs a 2D synthetic neurological phantom with realistic perfusion regimes.
    
    Tissues:
    - Background: CBF = 0, ATT = 0
    - White Matter (WM): CBF = 20 mL/100g/min, ATT = 1.4s
    - Grey Matter (GM): CBF = 65 mL/100g/min, ATT = 1.0s
    - Focal Ischemic Penumbra: CBF = 15 mL/100g/min, ATT = 2.2s (delayed arrival)
    - Hyperperfusion Core: CBF = 110 mL/100g/min, ATT = 0.8s (early shunt)
    """

    def __init__(self, size: int = 64):
        self.size = size
        self.cbf_ground_truth = np.zeros((size, size), dtype=float)
        self.att_ground_truth = np.zeros((size, size), dtype=float)
        self.m0_ground_truth = np.ones((size, size), dtype=float)
        self._synthesize()

    def _synthesize(self):
        s = self.size
        y, x = np.ogrid[:s, :s]
        cy, cx = s / 2.0, s / 2.0

        # Outer skull / brain boundary (ellipse)
        brain_mask = (((x - cx) / (0.42 * s)) ** 2 + ((y - cy) / (0.46 * s)) ** 2) <= 1.0
        self.cbf_ground_truth[brain_mask] = 22.0  # White matter background
        self.att_ground_truth[brain_mask] = 1.45

        # Cortex / Grey matter ring
        gm_mask = brain_mask & (
            (((x - cx) / (0.42 * s)) ** 2 + ((y - cy) / (0.46 * s)) ** 2) >= 0.55
        )
        self.cbf_ground_truth[gm_mask] = 68.0
        self.att_ground_truth[gm_mask] = 1.05

        # Deep grey matter nuclei
        left_nucleus = ((x - (cx - 0.16 * s)) ** 2 + (y - cy) ** 2) <= (0.09 * s) ** 2
        right_nucleus = ((x - (cx + 0.16 * s)) ** 2 + (y - cy) ** 2) <= (0.09 * s) ** 2
        self.cbf_ground_truth[left_nucleus | right_nucleus] = 75.0
        self.att_ground_truth[left_nucleus | right_nucleus] = 0.95

        # Focal Ischemia lesion (delayed ATT, severely reduced CBF)
        ischemia = ((x - (cx + 0.22 * s)) ** 2 + (y - (cy + 0.18 * s)) ** 2) <= (0.10 * s) ** 2
        self.cbf_ground_truth[ischemia] = 12.0
        self.att_ground_truth[ischemia] = 2.40

        # Hyperperfusion / Arteriovenous Malformation (early ATT, elevated CBF)
        hyper = ((x - (cx - 0.20 * s)) ** 2 + (y - (cy - 0.18 * s)) ** 2) <= (0.08 * s) ** 2
        self.cbf_ground_truth[hyper] = 115.0
        self.att_ground_truth[hyper] = 0.75

        self.m0_ground_truth[brain_mask] = 1.0
        self.m0_ground_truth[~brain_mask] = 0.05


def generate_multi_ti_dataset(
    phantom: DigitalSheppLoganASLPhantom,
    plds: np.ndarray = None,
    snr: float = 30.0,
    timing: ASLTimingParams = None,
    seed: int = 42
) -> np.ndarray:
    """Generates multi-PLD difference magnetization series \Delta M(pld, y, x) with Rician/Gaussian noise."""
    if plds is None:
        plds = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    
    model = BuxtonModel(timing=timing)
    np.random.seed(seed)
    
    series = []
    for pld in plds:
        clean_slice = np.zeros_like(phantom.cbf_ground_truth)
        for y in range(phantom.size):
            for x in range(phantom.size):
                c = phantom.cbf_ground_truth[y, x]
                a = phantom.att_ground_truth[y, x]
                m = phantom.m0_ground_truth[y, x]
                if c > 0:
                    clean_slice[y, x] = model.delta_m_pcasl(pld, c, a, m0=m)
        
        # Add Gaussian thermal noise commensurate with clinical SNR
        noise_sigma = (np.max(clean_slice) / snr) if np.max(clean_slice) > 0 else 1e-4
        noisy_slice = clean_slice + np.random.normal(0, noise_sigma, clean_slice.shape)
        series.append(np.clip(noisy_slice, 0, None))
        
    return np.array(series)
