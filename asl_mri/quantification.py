"""Quantitative Perfusion & Arterial Transit Time (ATT) Mapping."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit
from .buxton import BuxtonModel, ASLTimingParams


@dataclass
class PerfusionMap:
    cbf: np.ndarray  # Cerebral Blood Flow map (mL/100g/min)
    att: np.ndarray  # Arterial Transit Time map (s)
    residuals: np.ndarray  # Sum of squared residuals


class ASLQuantifier:
    """Multi-PLD nonlinear least-squares estimator for quantitative CBF & ATT."""

    def __init__(self, timing: ASLTimingParams = None):
        self.timing = timing or ASLTimingParams()
        self.buxton = BuxtonModel(timing=self.timing)

    def fit_voxel(self, plds: np.ndarray, delta_m_curve: np.ndarray, m0: float = 1.0) -> tuple[float, float, float]:
        """Fit a single voxel multi-PLD timecourse to obtain (CBF, ATT, residual)."""
        if np.all(delta_m_curve <= 0) or np.max(delta_m_curve) < 1e-6:
            return 0.0, 0.0, 0.0

        def model_func(pld_arr, cbf_val, att_val):
            res = []
            for p in pld_arr:
                res.append(self.buxton.delta_m_pcasl(p, cbf_val, att_val, m0=m0))
            return np.array(res)

        p0 = [50.0, 1.2]
        bounds = ([0.0, 0.1], [250.0, 3.5])

        try:
            popt, _ = curve_fit(model_func, plds, delta_m_curve, p0=p0, bounds=bounds, maxfev=1000)
            fitted = model_func(plds, *popt)
            residual = float(np.sum((delta_m_curve - fitted) ** 2))
            return float(popt[0]), float(popt[1]), residual
        except Exception:
            return 0.0, 0.0, 0.0

    def fit_volume(self, plds: np.ndarray, multi_pld_series: np.ndarray, m0_map: np.ndarray = None) -> PerfusionMap:
        """Fit full 2D or 3D multi-PLD volume.

        multi_pld_series shape: (n_plds, H, W) or (n_plds, D, H, W)
        """
        shape = multi_pld_series.shape[1:]
        cbf_map = np.zeros(shape, dtype=float)
        att_map = np.zeros(shape, dtype=float)
        res_map = np.zeros(shape, dtype=float)

        # Flatten spatial dimensions for efficient looping
        flat_series = multi_pld_series.reshape(multi_pld_series.shape[0], -1)
        flat_cbf = np.zeros(flat_series.shape[1], dtype=float)
        flat_att = np.zeros(flat_series.shape[1], dtype=float)
        flat_res = np.zeros(flat_series.shape[1], dtype=float)

        flat_m0 = np.ones(flat_series.shape[1], dtype=float) if m0_map is None else m0_map.flatten()

        for i in range(flat_series.shape[1]):
            curve = flat_series[:, i]
            if np.max(curve) > 1e-5:
                c, a, r = self.fit_voxel(plds, curve, m0=flat_m0[i])
                flat_cbf[i] = c
                flat_att[i] = a
                flat_res[i] = r

        cbf_map = flat_cbf.reshape(shape)
        att_map = flat_att.reshape(shape)
        res_map = flat_res.reshape(shape)

        return PerfusionMap(cbf=cbf_map, att=att_map, residuals=res_map)
