"""Unit and Invariance Test Suite for ASL MRI Perfusion Engine."""

import numpy as np
import pytest
from asl_mri.buxton import BuxtonModel, ASLTimingParams, ASLTissueParams
from asl_mri.quantification import ASLQuantifier
from asl_mri.phantom import DigitalSheppLoganASLPhantom, generate_multi_ti_dataset


def test_buxton_forward_signal_invariance():
    """Verify Buxton GKM signal increases monotonically during bolus inflow and decays after."""
    timing = ASLTimingParams(tau=1.8, t1_blood=1.65, alpha=0.85)
    model = BuxtonModel(timing=timing)
    att = 1.0
    cbf = 60.0
    
    # Prior to arrival (PLD < ATT), signal must be exactly 0
    pld_pre = 0.5
    sig_pre = model.delta_m_pcasl(pld_pre, cbf, att)
    assert sig_pre == 0.0
    
    # During bolus arrival
    pld_mid1 = 1.2
    pld_mid2 = 1.8
    sig_mid1 = model.delta_m_pcasl(pld_mid1, cbf, att)
    sig_mid2 = model.delta_m_pcasl(pld_mid2, cbf, att)
    assert sig_mid2 > sig_mid1 > 0.0
    
    # After bolus completion (PLD > ATT + tau)
    pld_late = 3.5
    sig_late = model.delta_m_pcasl(pld_late, cbf, att)
    assert 0.0 < sig_late < sig_mid2


def test_single_delay_cbf_reconstruction():
    """Test analytical inversion for single-delay pCASL under consensus white paper assumptions."""
    timing = ASLTimingParams(tau=1.8, t1_blood=1.65, alpha=0.85)
    model = BuxtonModel(timing=timing)
    
    true_cbf = 60.0
    att = 1.0
    pld = 2.0  # PLD > ATT, bolus in brain
    
    delta_m = model.delta_m_pcasl(pld, true_cbf, att)
    # Single delay formula assumes PLD > ATT and complete arrival approximation
    recon_cbf = model.single_delay_cbf_pcasl(delta_m, pld)
    
    # Discrepancy due to exp(-ATT/T1b) factor vs exp(-PLD/T1b)
    assert recon_cbf > 0
    assert np.isclose(recon_cbf, true_cbf, rtol=0.25)


def test_phantom_geometry_and_perfusion_contrasts():
    """Ensure digital phantom creates differentiated tissue perfusion profiles."""
    phantom = DigitalSheppLoganASLPhantom(size=32)
    assert phantom.cbf_ground_truth.shape == (32, 32)
    
    # Verify distinct tissue regions exist
    assert np.any(np.isclose(phantom.cbf_ground_truth, 68.0))  # Grey matter
    assert np.any(np.isclose(phantom.cbf_ground_truth, 22.0))  # White matter
    assert np.any(np.isclose(phantom.cbf_ground_truth, 12.0))  # Ischemia
    assert np.any(np.isclose(phantom.cbf_ground_truth, 115.0)) # Hyperperfusion


def test_multi_pld_quantification_accuracy():
    """Verify multi-PLD non-linear fit accurately recovers CBF and ATT."""
    timing = ASLTimingParams(tau=1.8, t1_blood=1.65, alpha=0.85)
    quantifier = ASLQuantifier(timing=timing)
    
    plds = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    true_cbf = 55.0
    true_att = 1.1
    
    model = BuxtonModel(timing=timing)
    synthetic_curve = np.array([model.delta_m_pcasl(p, true_cbf, true_att) for p in plds])
    
    fitted_cbf, fitted_att, res = quantifier.fit_voxel(plds, synthetic_curve)
    
    assert np.isclose(fitted_cbf, true_cbf, rtol=0.05)
    assert np.isclose(fitted_att, true_att, rtol=0.05)
    assert res < 1e-4


def test_noise_resilience():
    """Check that dataset generation with noise maintains positive SNR."""
    phantom = DigitalSheppLoganASLPhantom(size=16)
    data = generate_multi_ti_dataset(phantom, plds=np.array([1.5, 2.0]), snr=20.0)
    assert data.shape == (2, 16, 16)
    assert np.all(data >= 0.0)
