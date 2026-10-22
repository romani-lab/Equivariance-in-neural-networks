"""
Simulations of the equivariance analysis (src/simulations.ipynb): synthetic populations with
known structure, run through the identical analysis pipeline as the recorded data.

1. Low-dimensional simulations (ED Fig 8, SI Fig 7; `grid`, `run_cell`, `run_cell_cached`):
   the parameter-recovery benchmark on populations with a known low-dimensional equivariant
   subspace, one parameter varied per axis around a model-abiding reference, and the paired
   decoder effect sizes of the Methods (`effect_sizes_cached`, `effect_size_table`).
2. Calibration simulations based on pixels (ED Fig 9c-h, SI Fig 6f): the pooled CC_norm of the
   equivariance model over an (N, d) grid for the recorded populations (`ccnorm_surface`), the
   equivariant structure of a population measured at full rank (`band_structure`), and
   simulated populations built from that structure at a chosen equivariant signal fraction f
   and dimension R (`run_matched_populations`).
3. Fit of simulations to data (ED Fig 10): the measured generator parameters of every population
   (`datafit_params`), the equivariant fraction read off the (N, d) sweep (`best_iid`,
   `fill_swept`), and the structure-matched populations passed through the standard pipeline at
   N = 250, d = 20 (`run_target_cached`, `datafit_table`).

A population is generated as
    response_i(stim, rot) = g_i * (Q @ r_gen)_i(stim, rot) + noise,
with r_gen the generator (known invariant offsets, frequency-1 "tuned" planes, optional
higher-frequency planes, and optionally a non-equivariant iid remainder carrying a fraction
1 - f of the signal variance: many extra generator dimensions, independent across images
and rotations but consistent across trials, so that it dilutes the equivariant variance
without averaging out over trials and without carrying cross-rotation covariance), Q an
orthonormal embedding with equal row norms (so that the model-abiding reference is exactly
orthogonally equivariant after the pipeline's per-neuron std normalization, up to the
subsampling of neurons), g_i per-neuron gains (SNR heterogeneity), and Gaussian per-trial
noise as in utils.model_loader.train_test_split_w_noise (std = sqrt(inv_snr) * per-neuron
signal std; with gains, the noise is gain-independent so that SNR_i = g_i^2 / inv_snr).
Irregular angular steps (the angular coefficient of variation xi) are simulated by the
control 'C2p': shared planar rotations by positive angles that sum to 360 degrees.

The stimulus set is a single uniform block of n_images = 92 images, each with n_trials
trials (no train/test image split); the leave-two-out folds hold out image pairs for scoring.

The pipeline mirror then applies exactly the analysis steps of the real data: reliability
selection (corr_half > corr_thre, split-half over trials), disjoint neuron subsamples
(find_kn_unique_samples), leave-two-out pairs, per-neuron response-std normalization,
calculate_proj_rot_shift with d = cfg['d'], find_eigendecomp, band counts per fit
(helper.band_summary), the equivariance / fully invariant / per-rotation predictions scored
with the pooled within-image centered CC_norm (helper.pooled_scores) and, because the
ground truth is known, against the noiseless signal (true correlation and R^2; SI Fig 7),
and the two decoders of Fig 1d-e / Fig 3c in the full space and in the estimated subspaces
(run_decoders).

Every function that computes something heavy takes the project root and writes a cache; the
notebook loops over the cells / seeds / populations and skips what is cached. One such call is
also the unit of work for an array job on a cluster (see the README).

Usage (local, from the repository root):
    python src/utils/simulation.py list                   # grid cells
    python src/utils/simulation.py run <cell> <seed>      # one run -> cache/CACHE_DIR/<cell>_<seed>.csv
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.helper import find_eigendecomp, pooled_scores, band_summary, calculate_effect_size
from utils.neuron_selectivity import corr_half, direct_snr_stats, pooled_centred_r_half
from utils.equivariance_functions import calculate_proj_rot_shift
from utils.equivariance_validation import (_fit_per_step, predict_models, MODEL_NAMES, trial_caches_all,
                                           pooled_ccnorm_vs_dim_all, load_trials, subsamples)


# ----------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------
CACHE_DIR = 'simulation_v3'   # outputs: cache/<CACHE_DIR>/<cell>_<seed>.csv (+ _fits.csv)
DATA_INV_SNR = 3.9      # the data's noise level (noise sd = sqrt(3.9) x signal sd per trial)
SVM_C = 0.001           # SVM regularisation of the data analysis (decode_all c_range)
SVM_C_RANGE = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)   # sweep of run_decoders (svm_err_C*, svm_err_best)
DEFAULT = dict(
    # population / stimuli (92 images, 15 rotations, 8 trials; no train/test image split)
    n_neurons_tot=100000, n_images=92, n_rot=15, n_trials=8,   # N = 100k population, subsampled to n_neurons = 250
    # generator structure (dimensions)
    d_inv=6, d_tun=2, hf_planes=(), hf_amp=(),  # reference: 6 inv + 2 tun (1 plane) = 8 signal dims = d; no higher-frequency planes
    inv_amp=None,           # optional per-dimension amplitudes (sd) of the d_inv invariant dims, overriding the
                            # single epsilon: a within-band spectrum (used by the structure-matched populations)
    tun_amp=None,           # optional per-plane amplitudes (sd per dim) of the d_tun // 2 frequency-1 planes,
                            # overriding the single sqrt(var_tun)
    var_tun=1.0,            # variance of each tuned dimension relative to an invariant dimension (tuning share)
    epsilon=1.0,            # amplitude of the per-image invariant offsets (sd per invariant dim; 0 = images
                            # coincide in the invariant subspace, identity carried only by the rotating tuned code)
    r=1.0,                  # mean radius of the frequency-1 tuned planes per image (the unit scale)
    delta=1.0,              # sd of that radius across images; only the ratio delta/r (radius CV) affects
                            # results after the block's unit-variance normalization
    iid_frac=0.0,           # fraction 1 - f of signal variance in a non-equivariant iid remainder (structureless,
    n_iid_dims=200,         # consistent across trials); n_iid_dims extra generator dims embedded through Q
    iid_band_profile=None,  # angular profile of the remainder: None = white (independent across rotations);
                            # else the per-frequency variance shares (n_rot // 2 + 1 numbers, frequencies
                            # 0..n_rot // 2) the white block is filtered to along the rotation axis, so that
                            # the remainder has a prescribed angular spectrum (structure-matched populations)
    control='none',         # 'none', or 'C2p': irregular angular steps -- shared planar rotations by positive
                            # angles summing to 360 deg (the angular coefficient of variation xi of ED Fig 8b)
    c2_alpha=100.0,         # 'C2p' only: Dirichlet concentration of the step angles, phi = 360 deg * Dirichlet(alpha, ..., alpha)
                            # over the n_rot steps; xi = sqrt((n_rot - 1) / (n_rot * alpha + 1)): 100 -> 0.10, 1 -> 0.94;
                            # alpha -> inf = regular 24-deg steps
    # embedding / noise
    equal_rows=True, snr_hetero='none',   # per-neuron gain spread: 'none', 'mild' (lognormal sd 0.3),
                            # 'data' (lognormal sd 0.96, the measured spread), or a numeric lognormal sd
    inv_snr=1.0, corr_thre=0.1,   # reference inv_snr = 1; the noise sweep varies it (incl. the data's 3.9)
    # pipeline
    n_neurons=250, n_max=2, d=8, n_pairs=23,     # d = 8 (fits the 8-dim reference exactly); n_pairs: leave-two-out pairs used (None = all 46)
    decoders=True,          # image-identity SVM and relative-orientation decoders (run_decoders); the SVM
                            # regularisation sweep can be overridden per run with 'svm_c_range' / 'svm_c'
)


def grid():
    """
    Grid cells of ED Fig 8: name -> overrides of DEFAULT.

    The reference (= DEFAULT) is a model-abiding population (d_inv = 6, d_tun = 2 = 8 signal
    dims, fit d = 8, Gaussian invariant offsets, N = 100k neurons subsampled to 250,
    inv_snr = 1, 92 images, 8 trials, iid_frac = 0, homogeneous SNR, epsilon = delta = 1).
    One parameter is varied per axis around the reference (ED Fig 8b: fit d, N, inv_snr,
    angular irregularity xi, equivariant fraction f; ED Fig 8c: the invariant/tuned split,
    epsilon, delta, gain heterogeneity, tuning share, number of images, number of trials).
    """
    cells = {'reference': dict(n_pairs=None)}   # = DEFAULT
    #invariant/tuned SPLIT sweep at a fixed total dimension d_inv + d_tun = 8, with the fit d
    #matched to the truth (an exact fit). d_tun = 0 -> purely invariant, no across-rotation
    #split-half reliability: keep all neurons (corr_thre = -1).
    for di, dt in [(0, 8), (2, 6), (4, 4), (6, 2), (8, 0)]:
        extra = dict(corr_thre=-1.0) if dt == 0 else {}
        cells[f'dims_{di}_{dt}'] = dict(d_inv=di, d_tun=dt, d=di + dt, **extra)
    #fit-dimension sweep at the fixed true (d_inv, d_tun) = (6, 2): d = 2, 4, ..., 20
    for dd in range(2, 21, 2):
        cells[f'fitdim_{dd}'] = dict(d_inv=6, d_tun=2, d=dd)
    #N sweep: the neuron SUBSAMPLE size at the fixed 100k ambient population (`reference` is its
    #n_neurons = 250 point); the fit dimension is capped at min(8, N)
    for n in [4, 8, 16, 50, 1000]:
        cells[f'N_{n}'] = dict(n_neurons=n, n_neurons_tot=max(100000, n), d=min(8, n))
    for n in [12, 23, 46]:
        cells[f'images_{n}'] = dict(n_images=n)
    for t in [2, 4]:   # trial sweep 2, 4, 8 (reference)
        cells[f'trials_{t}'] = dict(n_trials=t)
    #noise sweep (`reference` is the inv_snr = 1 point; 3.9 = the data's noise)
    for s in [0, 0.5, 2, DATA_INV_SNR]:
        cells[f'snr_{s}'] = dict(inv_snr=s)
    #angular irregularity xi (`reference` is xi = 0): Dirichlet alpha 100 -> xi = 0.1, 1 -> xi = 0.9
    for a in [100, 1]:
        cells[f'control_C2p_a{a}'] = dict(control='C2p', c2_alpha=float(a), n_pairs=None)
    #tuning share: the tuned planes at a fraction of an invariant dimension's variance
    for v in [0.05, 0.1, 0.2, 0.5]:
        cells[f'datamatch_vtun_{v}'] = dict(var_tun=v, n_pairs=None)
    #equivariant signal fraction f = 1 - iid_frac
    for f in [0.75, 0.5, 0.25, 0.1]:
        cells[f'eqvfrac_{f}'] = dict(iid_frac=round(1 - f, 3), n_pairs=None)
    #signal geometry: the invariant-offset amplitude epsilon and the radius CV delta / r
    for e in [0.0, 0.4, 0.8, 1.5]:   # reference epsilon = 1.0
        cells[f'epsilon_{e}'] = dict(epsilon=e, n_pairs=None)
    for dl in [0.0, 0.3, 0.63]:      # reference delta = 1.0 (CV 1)
        cells[f'delta_{dl}'] = dict(delta=dl, n_pairs=None)
    #per-neuron gain heterogeneity (reference = 'none')
    for h in ['mild', 'data']:
        cells[f'snr_hetero_{h}'] = dict(snr_hetero=h, n_pairs=None)
    return cells


# ----------------------------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------------------------
def _unit_var(block):
    """Scale a (k, n_stim, n_rot) block so that the mean variance across conditions of its
    dimensions is 1 (one factor per block keeps planes isotropic)."""
    v = block.reshape(block.shape[0], -1).var(1).mean()
    return block / np.sqrt(v) if v > 0 else block


def _radius(rng, n, r, delta):
    """n per-image radii for a frequency-1 tuned plane, lognormal with mean r and sd delta (so
    coefficient of variation = delta / r; delta = 0 -> all radii = r). Positive by construction.
    After the tuned block's _unit_var normalization only the CV matters (the mean radius r is the
    unit scale)."""
    cv = delta / r if r > 0 else 0.0
    if cv <= 0:
        return np.full(n, r)
    sig = np.sqrt(np.log(1 + cv ** 2))
    return rng.lognormal(np.log(r) - sig ** 2 / 2, sig, n)


def make_generator(cfg, rng):
    """
    Generator responses r_gen (n_gen, n_stim, n_rot) and the ground truth.

    Returns
    -------
    r_gen : np.ndarray
    truth : dict with 'freq' (per generator dimension: 0 invariant, 1 tuned, 2/3/... higher
        frequency, -1 non-equivariant), 'd_inv', 'd_tun', 'angle' (rotation per step)
    """
    n_stim = cfg['n_images']
    n_rot = cfg['n_rot']
    theta = 2 * np.pi / n_rot
    k = np.arange(n_rot)
    blocks, freq = [], []

    def offsets(n):
        #per-image invariant offsets (the invariant-subspace displacement), returned as the
        #FINAL scaled (n, n_stim, n_rot) block (constant across rotations); None if n == 0.
        #Gaussian offsets: unit variance per invariant dim, then the amplitude epsilon (or the
        #per-dim amplitudes inv_amp -- a within-band spectrum)
        if n == 0:
            return None
        o = rng.normal(size=(n, n_stim))
        blk = _unit_var(np.repeat(o[:, :, None], n_rot, axis=2))
        if cfg.get('inv_amp') is not None:
            amp = np.asarray(cfg['inv_amp'], dtype=float)
            if amp.shape != (n,):
                raise ValueError(f'inv_amp must have d_inv = {n} entries, got {amp.shape}')
            return blk * amp[:, None, None]
        return blk * cfg['epsilon']

    control = cfg['control']
    if control not in ('none', 'C2p'):
        raise ValueError(f"control must be 'none' or 'C2p', got {control!r}")

    #shared structure: frequency-1 planes, higher-frequency planes, invariant offsets
    d_tun = cfg['d_tun']
    tun_amp = cfg.get('tun_amp')
    if tun_amp is not None and d_tun > 0 and len(tun_amp) != d_tun // 2:
        raise ValueError(f'tun_amp must have d_tun // 2 = {d_tun // 2} entries, got {len(tun_amp)}')
    for j in range(d_tun // 2):
        rad = _radius(rng, n_stim, cfg['r'], cfg['delta'])   # per-image radius, mean r, CV = delta/r
        phi = rng.uniform(0, 2 * np.pi, n_stim)
        blk = np.array([rad[:, None] * np.cos(k * theta + phi[:, None]), rad[:, None] * np.sin(k * theta + phi[:, None])])
        #per-plane amplitude (sd per dim): sqrt(var_tun) for every plane, or the tun_amp ladder
        amp = np.sqrt(cfg['var_tun']) if tun_amp is None else float(tun_amp[j])
        blocks.append(_unit_var(blk) * amp); freq += [1, 1]
    for f, amp in zip(cfg['hf_planes'], cfg['hf_amp']):
        #same structure as a frequency-1 plane (per-image lognormal radius, mean r sd delta,
        #and a random phase), only rotating f times per cycle
        rad = _radius(rng, n_stim, cfg['r'], cfg['delta'])
        phi = rng.uniform(0, 2 * np.pi, n_stim)
        blk = np.array([rad[:, None] * np.cos(f * k * theta + phi[:, None]), rad[:, None] * np.sin(f * k * theta + phi[:, None])])
        blocks.append(_unit_var(blk) * amp); freq += [f, f]
    o = offsets(cfg['d_inv'])
    if o is not None:
        blocks.append(o); freq += [0] * cfg['d_inv']
    if not blocks:
        #(d_inv, d_tun) = (0, 0) with no higher-frequency planes: no equivariant structure at all.
        #Emit a minimal structureless (noise-floor) block so the pipeline recovers ~0 dimensions.
        blocks.append(rng.normal(size=(2, n_stim, n_rot))); freq += [-1, -1]
    r = np.concatenate(blocks, 0)
    freq = np.array(freq)

    #C2p: irregular angular steps. On every non-invariant plane the per-step maps are planar
    #rotations by irregular, positive, image-shared angles phi_1..phi_n_rot that sum to 360 deg
    #(phi = 2 pi * Dirichlet(alpha, ..., alpha), alpha = cfg['c2_alpha']; each plane draws its own
    #sequence), so x_{i,k} = R(Phi_k) x_{i,0} with Phi_k = phi_1 + ... + phi_k: the object
    #manifold is a circle traversed once and closed at 360 deg, sampled at irregular angular
    #positions. Periodic and shared, but not compositional. Invariant offsets untouched.
    if control == 'C2p':
        tun = np.where(freq != 0)[0]
        if tun.size % 2:
            raise ValueError('C2p needs the non-invariant generator dims in planes (pairs)')
        for plane in tun.reshape(-1, 2):
            phi = 2 * np.pi * rng.dirichlet(cfg['c2_alpha'] * np.ones(n_rot))
            Phi = np.cumsum(phi)                           # Phi[n_rot - 1] = 2 pi (closed loop)
            x0 = r[plane, :, 0].copy()
            for j in range(1, n_rot):
                c, s = np.cos(Phi[j - 1]), np.sin(Phi[j - 1])
                r[plane, :, j] = np.array([[c, -s], [s, c]]) @ x0
        freq[tun] = -1

    #non-equivariant iid remainder (fraction iid_frac = 1 - f of the signal variance): many
    #extra generator dimensions, independent across images and rotations but CONSISTENT across
    #trials (a signal, not per-trial noise), embedded through Q like the structured dimensions.
    #Zero cross-rotation covariance -> stays out of the transition subspace in the full
    #population, so the estimated d_inv / d_tun and eigenphases are unaffected; it only inflates
    #each neuron's variance, diluting the equivariant signal and attenuating CC_norm. (Restricted
    #to a subsample of N neurons, each remainder direction puts ~d/N of its variance into the
    #true subspaces, which is why the subspace decoders degrade with f.)
    if cfg['iid_frac'] > 0:
        n_iid = int(cfg['n_iid_dims'])
        blk = _unit_var(rng.normal(size=(n_iid, n_stim, n_rot)))   # rotation-varying, ~unit var/dim
        prof = cfg.get('iid_band_profile')
        if prof is not None:
            #angular profile of the remainder: filter the white block along the rotation axis so
            #that its power over the rotation frequencies 0..n_rot//2 follows the given shares
            #instead of the white 1/n_rot : 2/n_rot : ... (each frequency >= 1 is a conjugate pair
            #of bins). The block stays image-specific (no shared transformation), only its tuning
            #curves become as smooth as prescribed. Deterministic given the block.
            prof = np.asarray(prof, dtype=float)
            n_bins = n_rot // 2 + 1
            if prof.shape != (n_bins,) or (prof < 0).any() or prof.sum() <= 0:
                raise ValueError(f'iid_band_profile must be {n_bins} non-negative shares (frequencies 0..{n_bins - 1})')
            prof = prof / prof.sum()
            white = np.full(n_bins, 2.0 / n_rot); white[0] = 1.0 / n_rot
            if n_rot % 2 == 0:
                white[-1] = 1.0 / n_rot           # the Nyquist bin is real and unpaired
            F = np.fft.rfft(blk, axis=2)
            F *= np.sqrt(prof / white)[None, None, :]
            blk = _unit_var(np.fft.irfft(F, n=n_rot, axis=2))
        v_struct = r.reshape(r.shape[0], -1).var(1).sum()
        blk_var = blk.reshape(n_iid, -1).var(1).sum()
        if cfg['iid_frac'] >= 1.0:
            #f = 0 (fully non-equivariant): DISCARD the shared structure and keep only the
            #image-specific remainder, at the same total variance the structure would have carried
            blk = blk * np.sqrt(v_struct / blk_var)
            r = blk
            freq = -np.ones(n_iid, dtype=int)
        else:
            blk = blk * np.sqrt(cfg['iid_frac'] / (1 - cfg['iid_frac']) * v_struct / blk_var)
            r = np.concatenate([r, blk], 0)
            freq = np.concatenate([freq, -np.ones(n_iid, dtype=int)])

    return r, dict(freq=freq, d_inv=int((freq == 0).sum()), d_tun=int((freq == 1).sum()), angle=theta)


def _svd_robust(A):
    """Thin SVD with a fallback: the divide-and-conquer driver (numpy's gesdd) can raise
    "SVD did not converge" on (near-)exactly orthonormal input, where LAPACK's gesvd succeeds."""
    try:
        return np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as _svd
        return _svd(A, full_matrices=False, lapack_driver='gesvd')


def make_embedding(n_neurons_tot, n_gen, rng, equal_rows=True, n_iter=200, tol=1e-12):
    """Orthonormal (n_neurons_tot, n_gen) embedding; with equal_rows, rows have equal norms
    (alternating projections), so that per-neuron std normalization of an isotropic
    generator is a global rescaling. The projections stop once every row norm is within
    tol (relative) of the target."""
    Q, _ = np.linalg.qr(rng.normal(size=(n_neurons_tot, n_gen)))
    if equal_rows:
        target = np.sqrt(n_gen / n_neurons_tot)
        for _ in range(n_iter):
            norms = np.linalg.norm(Q, axis=1, keepdims=True)
            if np.abs(norms - target).max() <= tol * target:
                break
            Q = Q / norms * target
            U, _, Vt = _svd_robust(Q)
            Q = U @ Vt
    return Q


def make_population(cfg, seed):
    """Noisy single-trial responses of a synthetic population (one uniform image set, no
    train/test split).

    Returns
    -------
    resp : (n_neurons_tot, n_images, n_rot, n_trials)
    truth : dict (see make_generator) plus 'Q' (embedding), 'gain', 'signal' (noiseless)
    """
    rng = np.random.default_rng(seed)
    r_gen, truth = make_generator(cfg, rng)
    n_gen = r_gen.shape[0]
    Q = make_embedding(cfg['n_neurons_tot'], n_gen, rng, equal_rows=cfg['equal_rows'])
    signal = np.einsum('ij,jkl->ikl', Q, r_gen)                      #(N, n_images, n_rot)
    #per-neuron gain spread: named preset ('none'/'mild'/'data') or a numeric lognormal sd
    _hetero = cfg['snr_hetero']
    sd = {'none': 0.0, 'mild': 0.3, 'data': 0.96}[_hetero] if isinstance(_hetero, str) else float(_hetero)
    gain = np.exp(rng.normal(0, sd, cfg['n_neurons_tot'])) if sd > 0 else np.ones(cfg['n_neurons_tot'])
    #noise: sqrt(inv_snr) x per-neuron std of the (gain-free) signal, as in
    #train_test_split_w_noise; the gain multiplies the signal only (SNR_i = g_i^2 / inv_snr)
    noise_sd = np.sqrt(cfg['inv_snr']) * signal.reshape(signal.shape[0], -1).std(1)
    rng_noise = np.random.default_rng(seed + 10_000)
    #resp = gain * signal + noise_sd * N(0,1), built in place (the single-trial array is ~9 GB at N = 100k)
    resp = rng_noise.normal(size=signal.shape + (cfg['n_trials'],))     #(N, n_images, n_rot, n_trials)
    resp *= noise_sd[:, None, None, None]
    resp += gain[:, None, None, None] * signal[..., None]
    truth.update(Q=Q, gain=gain, signal=signal)
    return resp, truth


# ----------------------------------------------------------------------------------------
# pipeline mirror
# ----------------------------------------------------------------------------------------
def reliable_mask(resp, corr_thre):
    """The pipeline's reliability selection on single trials (n, n_stim, n_rot, n_trials): the
    per-stimulus split-half correlation (mean over stimuli and rotations) above corr_thre and a
    non-degenerate trial-averaged response. Returns (mask, cc_half)."""
    cc_half = corr_half(resp, n_perm=1 if resp.shape[-1] == 2 else 10).mean(1).mean(1)
    std_resp = resp.mean(-1).reshape(resp.shape[0], -1).std(1)
    return (cc_half > corr_thre) & (std_resp > 0), cc_half


def _true_scores(sig, pred, center):
    """Per-neuron R^2 and correlation of predictions (n, n_stim, n_rot) against the noiseless
    signal of the same shape; centered: both by their own per-image mean over rotations.
    Neurons without signal variance to explain give NaN."""
    s, p = sig, pred
    if center:
        s = s - s.mean(2, keepdims=True); p = p - p.mean(2, keepdims=True)
    s2 = s.reshape(s.shape[0], -1); p2 = p.reshape(p.shape[0], -1)
    with np.errstate(invalid='ignore', divide='ignore'):
        vs = s2.var(1)
        r2 = 1 - np.nanmean((s2 - p2)**2, 1) / vs
        sc_ = s2 - s2.mean(1, keepdims=True); pc_ = p2 - np.nanmean(p2, 1, keepdims=True)
        cc_true = np.nansum(sc_ * pc_, 1) / np.sqrt(np.nansum(sc_**2, 1) * np.nansum(pc_**2, 1))
    degenerate = vs <= 1e-12 * np.nanmean(p2**2, 1).max()
    r2[degenerate] = np.nan
    cc_true[degenerate] = np.nan
    return r2, cc_true


def run_decoders(trials_all, k_samp, cfg, verbose=False):
    """
    The two decoders of Fig 1d-e on the reliable neurons of a synthetic population, in the
    full neuron space, with the neuron subsamples of the pipeline and the settings of the
    data analysis (decoder_functions.decode_all: ref_index = 0, first 2 trials):
    - image identity: linear SVM trained on the single trials of all but one rotation of every
      image, tested at the held-out rotation (trial average), all rotations held out in turn
      (texture2x_svm_decoder_parallel); reported as the test error 1 - accuracy at the data's
      C = SVM_C (`svm_err`) and for every C of SVM_C_RANGE (`svm_err_C<c>`, `svm_err_train_C<c>`),
      with the best test error over the range (`svm_err_best`, `svm_best_C`; the data's
      C = 0.001 underfits noiseless populations, so ED Fig 8 reports the best-C error). The
      sweep and the reported C can be overridden per run with cfg['svm_c_range'] / cfg['svm_c']
      (the structure-matched populations of ED Fig 10 use (0.001, 1) and C = 1);
    - relative orientation: 2-D PCA of the per-image-centred training responses, held-out
      image aligned at rotation 0 and read out as an angle, leave-one-image-out
      (orientation2x_2d_decoder); reported as the mean absolute angle error in degrees.

    Both decoders are also run in the estimated equivariant subspace at the pipeline's fit
    dimension d = cfg['d'] (image identity in the invariant subspace, relative orientation in
    the tuned subspace; decode_all with equivariant=True), reported with the `_eqv` suffix.
    Both image-identity SVMs are skipped when the true d_inv = 0: identity is then carried
    only by the rotating tuned code, so leave-one-rotation-out is at chance.

    Parameters
    ----------
    trials_all : (n_neurons, n_stim, n_rot, n_trials) single trials of the reliable neurons
    k_samp : (n_samples, n_neurons) neuron subsamples

    Returns
    -------
    dict : svm_err, svm_err_train, svm_conf (C = SVM_C), svm_err_C<c>, svm_err_train_C<c>,
        svm_err_best, svm_best_C, ori_err, ori_r2 (means over items and subsamples); and the
        equivariant-subspace variants svm_err_eqv, svm_err_eqv_train, svm_err_eqv_C<c>,
        svm_err_eqv_train_C<c>, svm_err_eqv_best, svm_best_C_eqv, ori_err_eqv, ori_r2_eqv
    """
    from utils.decoder_functions import orientation2x_2d_decoder, texture2x_svm_decoder_parallel
    n_neurons = k_samp.shape[1]
    n_rot = trials_all.shape[2]
    d = cfg['d']
    svm_c = float(cfg.get('svm_c', SVM_C))
    ori = orientation2x_2d_decoder(trials_all.mean(-1), n_neurons, rot_step=360 / n_rot, ref_index=0,
                                   k_samp=k_samp, verbose=verbose)
    c_range = sorted(set(cfg.get('svm_c_range', SVM_C_RANGE)) | {svm_c})
    out = dict(ori_err=float(ori['res'].mean()), ori_r2=float(ori['R2'].mean()))
    if cfg['d_inv'] > 0:
        svm = texture2x_svm_decoder_parallel(trials_all, n_neurons, c_range, k_samp=k_samp, verbose=verbose)
        by_c = svm.groupby('C')[['acc test', 'acc train', 'conf test']].mean()
        for c, r in by_c.iterrows():
            out[f'svm_err_C{c:g}'] = float(1 - r['acc test'])
            out[f'svm_err_train_C{c:g}'] = float(1 - r['acc train'])
        r = by_c.loc[svm_c]
        out.update(svm_err=float(1 - r['acc test']), svm_err_train=float(1 - r['acc train']), svm_conf=float(r['conf test']))
        out['svm_err_best'] = float(1 - by_c['acc test'].max())
        out['svm_best_C'] = float(by_c['acc test'].idxmax())
    elif verbose:
        print('  image-identity SVM (full space + equivariant) skipped (d_inv = 0)', flush=True)
    #same decoders in the estimated equivariant subspace. Guarded: the estimated subspace can be
    #empty for some fit, which would raise from LinearSVC; skip the equivariant columns then
    #rather than losing the whole cell.
    try:
        ori_e = orientation2x_2d_decoder(trials_all.mean(-1), n_neurons, rot_step=360 / n_rot, ref_index=0,
                                         d=d, k_samp=k_samp, verbose=verbose)
        out['ori_err_eqv'] = float(ori_e['res'].mean()); out['ori_r2_eqv'] = float(ori_e['R2'].mean())
        if cfg['d_inv'] > 0:
            svm_e = texture2x_svm_decoder_parallel(trials_all, n_neurons, c_range, d=d, k_samp=k_samp, verbose=verbose)
            by_c_e = svm_e.groupby('C')[['acc test', 'acc train']].mean()
            for c, r in by_c_e.iterrows():
                out[f'svm_err_eqv_C{c:g}'] = float(1 - r['acc test'])
                out[f'svm_err_eqv_train_C{c:g}'] = float(1 - r['acc train'])
            r = by_c_e.loc[svm_c]
            out.update(svm_err_eqv=float(1 - r['acc test']), svm_err_eqv_train=float(1 - r['acc train']))
            out['svm_err_eqv_best'] = float(1 - by_c_e['acc test'].max())
            out['svm_best_C_eqv'] = float(by_c_e['acc test'].idxmax())
        elif verbose:
            print('  equivariant image-identity SVM skipped (d_inv = 0)', flush=True)
    except Exception as e:
        print(f'  equivariant-subspace decoders skipped ({type(e).__name__}: {e})', flush=True)
    return out


def run_pipeline(resp, truth, cfg, seed=0, verbose=False):
    """
    Apply the real-data pipeline to a synthetic population and score against the truth.

    Parameters
    ----------
    resp : (n_neurons_tot, n_images, n_rot, n_trials) single-trial responses (one uniform
        image set; no train/test split)

    Returns
    -------
    summary : dict of scalar results (one CSV row): reliability selection, the decoders
        (run_decoders), the per-fit band counts / variances / correlations averaged over fits
        (d_inv, d_tun, ...), and per model m in MODEL_NAMES the medians over neurons of the
        pooled centered scores (cc_norm_<m>_c, feve_<m>_c, cc_<m>_c, gain_<m>_c,
        frac_ceiling_<m>_c) and of the true R^2 / correlation against the noiseless signal
        (true_r2_<m>_c, true_cc_<m>_c), averaged over neuron subsamples
    fits : pd.DataFrame with per-(stim, sample, dim) rows ('evals', 'var test', 'pc test') --
        the per-fit spectra, as the var_dim caches of the data
    """
    n_rot = cfg['n_rot']
    d = cfg['d']
    n_neurons_tot = resp.shape[0]
    #reliability selection (split-half over trials, averaged over images and rotations)
    keep, cc_half = reliable_mask(resp, cfg['corr_thre'])
    idx_resp = np.where(keep)[0]
    n_rel = idx_resp.shape[0]
    summary = dict(n_reliable=n_rel, frac_reliable=n_rel / n_neurons_tot,
                   median_cc_half=float(np.median(cc_half[idx_resp])) if n_rel else np.nan)
    if n_rel < 10:
        summary.update(status='too_few_reliable')
        return summary, pd.DataFrame()

    trials_all = resp[idx_resp]                                       #(n_rel, n_images, n_rot, n_trials)
    x_all = trials_all.mean(-1)
    n_all = cfg['n_images']
    n_neurons, k_rand, k_samp = subsamples(n_rel, cfg['n_neurons'], cfg['n_max'])
    #decoders (Fig 1d-e / Fig 3c) on the same neuron subsamples
    if cfg.get('decoders', False):
        if verbose:
            print('  decoders', flush=True)
        summary.update(run_decoders(trials_all, k_samp, cfg))
    #leave-two-out folds over the uniform image set: n_all/2 disjoint pairs cover every image once
    pair_idx = [(i, n_all // 2 + i) for i in range(n_all // 2)]
    if cfg['n_pairs'] is not None:
        pair_idx = pair_idx[:cfg['n_pairs']]
    rot_idx = np.arange(1, n_rot)
    pred_all = {k: {m: np.full((n_neurons, n_all, n_rot), np.nan) for m in MODEL_NAMES} for k in range(k_rand)}
    fit_rows = []

    for stim, (a, b) in enumerate(pair_idx):
        mask = (np.arange(n_all) != a) & (np.arange(n_all) != b)
        for k in range(k_rand):
            idx = k_samp[k]
            x_train = x_all[idx][:, mask]
            x_test = x_all[idx][:, [a, b]]
            std_norm = x_train.std(axis=(1, 2), keepdims=True)
            nz = std_norm.squeeze() > 0
            x_train_k = x_train[nz] / std_norm[nz]
            x_test_k = x_test[nz] / std_norm[nz]
            proj, proj_shift, rot, rot_shift = calculate_proj_rot_shift(x_train_k, d)
            evals, eigenvec = find_eigendecomp(rot)
            evals = evals / (2 * np.pi / n_rot)
            B = proj @ eigenvec
            preds = predict_models(x_test_k, proj, proj_shift, rot, rot_shift,
                                   per_step=_fit_per_step(x_train_k, proj, proj_shift))
            #per-dimension held-out variance share and correlation of the equivariance model
            xt = x_test_k.reshape(x_test_k.shape[0], -1)
            var_raw = (B.T @ xt).std(1)**2 / (xt.std(1)**2).sum()
            xp = np.einsum('ji,jkl->ikl', B, x_test_k - proj_shift[:, None, None]).reshape(d, -1)
            pp = np.einsum('ji,jkl->ikl', B, preds['equivariance'] - proj_shift[:, None, None]).reshape(d, -1)
            xp = xp - xp.mean(1, keepdims=True); pp = pp - pp.mean(1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                pc = (xp * pp).sum(1) / np.sqrt((xp**2).sum(1) * (pp**2).sum(1))
            fit_rows.append(pd.DataFrame({'stim': stim, 'sample': k, 'dim': np.arange(d), 'evals': evals,
                                          'var train': np.nan, 'var test': var_raw, 'pc train': np.nan,
                                          'pc test': pc}))
            #store in original (unnormalized) units so that pooling across folds does not mix
            #the folds' scalings
            for m in MODEL_NAMES:
                pred_all[k][m][nz, a] = preds[m][:, 0] * std_norm[nz][:, :, 0]
                pred_all[k][m][nz, b] = preds[m][:, 1] * std_norm[nz][:, :, 0]
        if verbose and stim % 10 == 0:
            print(f'  pair {stim + 1}/{len(pair_idx)}', flush=True)

    fits = pd.concat(fit_rows, ignore_index=True)
    band = band_summary(fits)
    summary.update(status='ok', n_samples=k_rand, n_pairs=len(pair_idx),
                   true_d_inv=truth['d_inv'], true_d_tun=truth['d_tun'], true_n_gen=len(truth['freq']),
                   d_inv=band['d_inv'].mean(), d_inv_sd=band['d_inv'].std(),
                   d_tun=band['d_tun'].mean(), d_tun_sd=band['d_tun'].std(),
                   var_test_inv=band['var test inv'].mean(), var_test_tun=band['var test tun'].mean(),
                   pc_test_inv=band['pc test inv'].mean(), pc_test_tun=band['pc test tun'].mean())

    #pooled scores over the held-out images, and the TRUE correlation / explained variance of
    #the predictions against the noiseless signal (the calibration of CC_norm, SI Fig 7)
    held = sorted(set(sum(([a, b] for a, b in pair_idx), [])))
    signal_rel = truth['gain'][idx_resp][:, None, None] * truth['signal'][idx_resp]     #(n_rel, n_all, n_rot)
    for k in range(k_rand):
        idx = k_samp[k]
        avg = x_all[idx][:, held][:, :, rot_idx]
        trials = trials_all[idx][:, held][:, :, rot_idx]
        sig = signal_rel[idx][:, held][:, :, rot_idx]
        for m in MODEL_NAMES:
            pred = pred_all[k][m][:, held][:, :, rot_idx]
            sc = pooled_scores(pred, avg, trials, center=True)
            tag = f'{m}_c'
            for q in ['cc_norm', 'feve', 'cc', 'gain']:
                summary.setdefault(f'{q}_{tag}', []).append(np.nanmedian(sc[q]))
            summary.setdefault(f'frac_ceiling_{tag}', []).append(np.mean(np.isfinite(sc['cc_max'])))
            r2, cc_true = _true_scores(sig, pred, True)
            summary.setdefault(f'true_r2_{tag}', []).append(np.nanmedian(r2))
            summary.setdefault(f'true_cc_{tag}', []).append(np.nanmedian(cc_true))
    for key in list(summary.keys()):
        if isinstance(summary[key], list):
            summary[key] = float(np.nanmean(summary[key]))
    return summary, fits


def run_cell(name, seed, overrides=None, verbose=False):
    """One grid cell at one seed: generate the population, run the pipeline mirror. Returns
    (summary dict, fits frame); the summary records the cell, seed and configuration."""
    cfg = dict(DEFAULT)
    cfg.update(grid()[name])
    if overrides:
        cfg.update(overrides)
    resp, truth = make_population(cfg, seed)
    summary, fits = run_pipeline(resp, truth, cfg, seed=seed, verbose=verbose)
    summary.update(cell=name, seed=seed, config=json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}))
    return summary, fits


def run_cell_cached(project_root, name, seed, n_max=4, n_pairs=None, verbose=False):
    """One (cell, seed) run of the low-dimensional benchmark, cached as
    cache/<CACHE_DIR>/<cell>_<seed>.csv (+ <cell>_<seed>_fits.csv). The paper's runs use all
    46 leave-two-out pairs (n_pairs None) and n_max = 4 neuron subsamples per fit, 10 seeds
    per cell (100 for the two angular-irregularity cells). Returns the summary row."""
    out_dir = Path(project_root) / 'cache' / CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_f = out_dir / f'{name}_{seed}.csv'
    if out_f.exists():
        return pd.read_csv(out_f).iloc[0].to_dict()
    print(f'simulation {name} seed {seed} (n_max {n_max}, n_pairs {n_pairs})', flush=True)
    summary, fits = run_cell(name, seed, dict(n_pairs=n_pairs, n_max=n_max), verbose=verbose)
    pd.DataFrame([summary]).to_csv(out_f, index=False)
    fits.to_csv(out_dir / f'{name}_{seed}_fits.csv', index=False)
    return summary


# ----------------------------------------------------------------------------------------
# decoder effect sizes of the low-dimensional simulations (Methods, "Decoder comparison and
# effect size quantification": the ED Fig 8b values for N and sigma^2)
# ----------------------------------------------------------------------------------------
EFFECT_CACHE_DIR = 'simulation_effect'           # cache/<EFFECT_CACHE_DIR>/<cell>_<seed>_bestC.csv
EFFECT_CELLS = ['reference', 'N_4', 'N_8', 'N_16', 'N_50', 'N_1000', 'snr_0', 'snr_0.5', 'snr_2', 'snr_3.9']
C_FULL, C_SUB = 0.001, 1.0                       # the manuscript's fixed C of the data (full space, subspace)


def _effect_size_row(full_df, sub_df, score_full, score_sub):
    """Merge full-space and subspace per-condition decoder outputs on (class, ori) and run the same
    calculate_effect_size as the real data (decoder_functions.decoder_effect_sizes). Returns a dict
    of the effect-size stats (NaNs if too few paired conditions, e.g. an empty estimated subspace)."""
    a = full_df[['class', 'ori']].copy(); a['score_1'] = score_full(full_df)
    b = sub_df[['class', 'ori']].copy(); b['score_2'] = score_sub(sub_df)
    df = a.merge(b, on=['class', 'ori']).dropna(subset=['score_1', 'score_2'])
    if len(df) < 3:
        return dict(mean_delta=np.nan, d_paired=np.nan, auc=np.nan,
                    auc_ci_low=np.nan, auc_ci_high=np.nan, n_cond=len(df))
    r = calculate_effect_size(df).iloc[0].to_dict()
    r['n_cond'] = len(df)
    return {k: r[k] for k in ('mean_delta', 'd_paired', 'auc', 'auc_ci_low', 'auc_ci_high', 'n_cond')}


def _svm_effect_sizes(trials_all, n_neurons, d, k_samp):
    """Image-identity effect sizes from ONE sweep of the SVM over SVM_C_RANGE in the full space
    and in the estimated invariant subspace (one row per (C, class, ori), averaged over
    subsamples): at the C with the lowest mean held-out error per space (svm_*, the rule of the
    ED Fig 8 panels), at the data's fixed C (svm_fixed_*), the chosen C per space and the mean
    held-out errors."""
    from utils.decoder_functions import texture2x_svm_decoder_parallel
    err = lambda x: 1.0 - x['acc test'].values
    full = texture2x_svm_decoder_parallel(trials_all, n_neurons, SVM_C_RANGE, k_samp=k_samp)
    sub = texture2x_svm_decoder_parallel(trials_all, n_neurons, SVM_C_RANGE, k_samp=k_samp, d=d)
    by_c_full = full.groupby('C')['acc test'].mean()
    by_c_sub = sub.groupby('C')['acc test'].mean()
    c_full = float(by_c_full.idxmax()) if np.isfinite(by_c_full).any() else C_FULL   # lowest mean held-out error
    c_sub = float(by_c_sub.idxmax()) if np.isfinite(by_c_sub).any() else C_SUB      # (run_decoders' svm_err_best)
    at = lambda df, c: df[np.isclose(df['C'], c)]
    out = {f'svm_{k}': v for k, v in _effect_size_row(at(full, c_full), at(sub, c_sub), err, err).items()}
    out.update({f'svm_fixed_{k}': v for k, v in _effect_size_row(at(full, C_FULL), at(sub, C_SUB), err, err).items()})
    out.update(svm_C_full=c_full, svm_C_sub=c_sub, svm_best_C_full=c_full, svm_best_C_sub=c_sub,
               svm_err_full=float(1 - by_c_full.get(c_full, np.nan)), svm_err_sub=float(1 - by_c_sub.get(c_sub, np.nan)),
               svm_err_full_fixed=float(1 - by_c_full.get(C_FULL, np.nan)), svm_err_sub_fixed=float(1 - by_c_sub.get(C_SUB, np.nan)))
    for c in SVM_C_RANGE:                                   # the whole sweep, for the record
        out[f'svm_err_full_C{c:g}'] = float(1 - by_c_full.get(c, np.nan))
        out[f'svm_err_sub_C{c:g}'] = float(1 - by_c_sub.get(c, np.nan))
    return out


def effect_sizes_cell(name, seed, n_neurons_tot=None, verbose=False):
    """Paired decoder effect sizes (full space vs estimated subspace) of one grid cell at one
    seed: the population of run_cell (deterministic in the configuration and seed), its
    reliability selection and neuron subsamples, and both decoders retaining the per-condition
    held-out errors that the paired statistics need (mean(delta)/std(delta) and the ROC AUC with
    the decoding error as the decision variable; helper.calculate_effect_size, as on the data).
    n_neurons_tot shrinks the ambient population for a local test."""
    from utils.decoder_functions import orientation2x_2d_decoder
    cfg = dict(DEFAULT); cfg.update(grid()[name])
    if n_neurons_tot:
        cfg['n_neurons_tot'] = min(int(n_neurons_tot), cfg['n_neurons_tot'])
    resp, _ = make_population(cfg, seed)
    keep, _ = reliable_mask(resp, cfg['corr_thre'])
    out = dict(cell=name, seed=seed, f=round(1 - cfg.get('iid_frac', 0.0), 3),
               N=cfg['n_neurons'], inv_snr=cfg['inv_snr'], d=cfg['d'],
               d_inv_true=cfg['d_inv'], d_tun_true=cfg['d_tun'])
    n_rel = int(keep.sum())
    if n_rel < 10:
        out['status'] = 'too_few_reliable'
        return out
    trials_all = resp[keep]
    n_neurons, _, k_samp = subsamples(n_rel, cfg['n_neurons'], cfg['n_max'])
    out.update(status='ok', n_reliable=n_rel, n_neurons=n_neurons)
    d, n_rot = cfg['d'], trials_all.shape[2]
    x_avg = trials_all.mean(-1)
    ori_full = orientation2x_2d_decoder(x_avg, n_neurons, rot_step=360 / n_rot, ref_index=0, k_samp=k_samp)
    ori_sub = orientation2x_2d_decoder(x_avg, n_neurons, rot_step=360 / n_rot, ref_index=0, d=d, k_samp=k_samp)
    out.update({f'ori_{k}': v for k, v in
                _effect_size_row(ori_full, ori_sub, lambda x: x['res'].values, lambda x: x['res'].values).items()})
    out.update(_svm_effect_sizes(trials_all, n_neurons, d, k_samp))
    if verbose:
        print(f'  {name}_{seed}: ori AUC={out.get("ori_auc", float("nan")):.3f}, '
              f'svm AUC={out.get("svm_auc", float("nan")):.3f} (C {out.get("svm_C_full")}/{out.get("svm_C_sub")})', flush=True)
    return out


def effect_sizes_cached(project_root, name, seed, n_neurons_tot=None):
    """effect_sizes_cell cached as cache/<EFFECT_CACHE_DIR>/<cell>_<seed>_bestC.csv."""
    out_dir = Path(project_root) / 'cache' / EFFECT_CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f'{name}_{seed}_bestC.csv'
    if f.exists():
        return pd.read_csv(f).iloc[0].to_dict()
    print(f'effect sizes {name} seed {seed}', flush=True)
    s = effect_sizes_cell(name, seed, n_neurons_tot=n_neurons_tot, verbose=True)
    pd.DataFrame([s]).to_csv(f, index=False)
    return s


def effect_size_table(project_root, cells=EFFECT_CELLS):
    """Per cell the mean and sem over seeds of the paired effect sizes (AUC, mean(delta)/std,
    mean(delta)) of both decoders at the best C (svm_*, ori_*) and at the data's fixed C
    (svm_fixed_*), the C in use and the mean decoding errors, with the cell's f, N and inv_snr
    -> results_extended/simulation/simulation_effect_sizes.csv. Reads the cached rows."""
    root = Path(project_root)
    rows = []
    for f in sorted((root / 'cache' / EFFECT_CACHE_DIR).glob('*_bestC.csv')):
        r = pd.read_csv(f).iloc[0].to_dict()
        if r.get('status') == 'ok':
            rows.append(r)
    if not rows:
        raise FileNotFoundError(f'no effect-size rows in cache/{EFFECT_CACHE_DIR}')
    df = pd.DataFrame(rows)

    def agg(g, col):
        v = g[col][np.isfinite(g[col])]
        if len(v) == 0:
            return np.nan, 0.0
        return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)

    recs = []
    for cell in cells:
        g = df[df['cell'] == cell]
        if g.empty:
            continue
        rec = dict(cell=cell, n_seeds=len(g), f=float(g['f'].iloc[0]), N=int(g['N'].iloc[0]), inv_snr=float(g['inv_snr'].iloc[0]))
        for dec in ('ori', 'svm', 'svm_fixed'):
            for col in ('auc', 'd_paired', 'mean_delta'):
                rec[f'{dec}_{col}_mean'], rec[f'{dec}_{col}_sem'] = agg(g, f'{dec}_{col}')
        for col in ('svm_C_full', 'svm_C_sub'):           # the C in use (mode over seeds) ...
            rec[col] = float(g[col].mode().iloc[0])
        for col in ('svm_err_full', 'svm_err_sub', 'svm_err_full_fixed', 'svm_err_sub_fixed'):   # ... and the errors
            rec[f'{col}_mean'] = agg(g, col)[0]
        recs.append(rec)
    out = pd.DataFrame(recs)
    out_dir = root / 'results_extended' / 'simulation'
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / 'simulation_effect_sizes.csv', index=False)
    return out


# ----------------------------------------------------------------------------------------
# calibration simulations based on pixels (ED Fig 9c-h, SI Fig 6f) and the fit of simulations
# to data (ED Fig 10): populations and targets
# ----------------------------------------------------------------------------------------
CALIBRATION_DIR = 'calibration'                  # results_extended/<CALIBRATION_DIR>/: band_structure_<name>.json, figures
DATAFIT_DIR = 'data_fit'                         # results_extended/<DATAFIT_DIR>/: data_fit_params.csv, data_fit_sim.csv, figures
DATAFIT_CACHE_DIR = 'simulation_datafit_sized'   # cache/<DATAFIT_CACHE_DIR>/<target>_<seed>.csv (ED Fig 10b)
SWEPT_IID_CSV = 'params/datafit_swept_iid.csv'   # the equivariant fraction f = 1 - iid per target (fill_swept)
FMAX = 7                                         # highest rotation frequency (Nyquist of 15 rotations)
FREQS = list(range(FMAX + 1))
MICE = {'TX108': 'TX108_05_31_2023', 'TX109': 'TX109_05_30_2023', 'TX111': 'TX111_05_18_2023'}   # mouse 1, 2, 3
AREAS = ['V1', 'LM', 'RL', 'AL']
CONVS = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
#the ten populations of ED Fig 9-10: name -> (cache folder, layer). The mouse areas are the
#reliably responsive neurons pooled across the three mice (build_pooled_caches).
TARGETS = {'pixels': ('pixels', 'pixels'), **{a: ('mouse_pooled', a) for a in AREAS}, **{c: ('alexnet', c) for c in CONVS}}
TARGET_ORDER = ['pixels'] + AREAS + CONVS
#populations whose noise was injected proportionally to each unit's own std (model_loader.
#train_test_split_w_noise): every unit has the same SNR, so their simulations use a uniform gain
UNIFORM_SNR = {'pixels', *CONVS}
#simulated population sizes (Methods, "Fit of simulations to data"): V1 and pixels 1e5; the other
#mouse areas scale with their pooled reliable-neuron counts; an AlexNet layer is the whole
#population, 1.1 x its reliably responsive units (the reliability selection then leaves at least
#as many simulated as real units)
V1_TOTAL = 100_000
N_MEASURED = dict(V1=10_698, LM=3_100, RL=1_696, AL=855)      # pooled reliable neurons, 3 mice
CONV_N_REL = dict(conv1=5040, conv2=2374, conv3=927, conv4=539, conv5=303)   # reliable units at the data's noise
CONV_POP_MARGIN = 1.1
#the (N, d) grid of the surfaces (d >= N is dropped by ccnorm_surface)
N_LIST = (250, 1000, 4000, 8000)
D_LIST = (6, 20, 50, 100, 200, 400, 800)
FIT_DMIN, FIT_DMAX = 20, 200          # d-window of the surface comparison (best_iid), see Methods
#the standard analysis of the data (fit dimension, neurons per subsample, trials)
FIT = dict(d=20, n_neurons=250, n_trials=2)
#pre-selection gain spread = HETERO_TRUNC x the measured post-selection spread: the reliability
#selection truncates the low-SNR tail and narrows the spread to ~0.8 of the injected value
HETERO_TRUNC = 1.3
MIN_LEADING_VAR = 1e-3                 # within-band PCs below this fraction of the leading one are discarded


def population_size(name):
    """Size of the simulated population a target's neurons are drawn from (see above)."""
    if name in CONV_N_REL:
        return int(round(CONV_POP_MARGIN * CONV_N_REL[name]))
    if name in N_MEASURED and name != 'V1':
        return int(round(V1_TOTAL * N_MEASURED[name] / N_MEASURED['V1']))
    return V1_TOTAL


def _params_single(project_root, n_neurons, n_max, noise_model='gaussian'):
    """params frame for the validation functions with the standard analysis parameters and the
    given subsample size / count / noise model."""
    params_df = pd.read_csv(Path(project_root) / 'params' / 'fixed_params.csv', header=0, index_col=0)
    params_df.loc['n_neurons', 'value'] = str(int(n_neurons))
    params_df.loc['n_max', 'value'] = str(int(n_max))
    params_df.loc['noise_model', 'value'] = noise_model
    return params_df


# ---- measured populations: trial caches and (N, d) surfaces --------------------------------
def build_pooled_caches(project_root, areas=AREAS):
    """Trial caches of each visual area pooled across the three mice (cache/mouse_pooled/
    {area}_resp_trials_{train,test}.p): the per-mouse reliable-neuron caches (trial_caches_all)
    concatenated along the neuron axis (same stimuli; trials truncated to the common minimum)."""
    cache = Path(project_root) / 'cache'
    outdir = cache / 'mouse_pooled'; outdir.mkdir(parents=True, exist_ok=True)
    for area in areas:
        ftr = outdir / f'{area}_resp_trials_train.p'; fte = outdir / f'{area}_resp_trials_test.p'
        if ftr.exists() and fte.exists():
            print(f'pooled {area}: exists', flush=True); continue
        trs = [pickle.load(open(cache / m / f'{area}_resp_trials_train.p', 'rb')) for m in MICE.values()]
        tes = [pickle.load(open(cache / m / f'{area}_resp_trials_test.p', 'rb')) for m in MICE.values()]
        ntr = min(t.shape[3] for t in trs); nte = min(t.shape[3] for t in tes)
        tr = np.concatenate([t[:, :, :, :ntr] for t in trs], axis=0)
        te = np.concatenate([t[:, :, :, :nte] for t in tes], axis=0)
        pickle.dump(tr, open(ftr, 'wb'), protocol=4); pickle.dump(te, open(fte, 'wb'), protocol=4)
        print(f'pooled {area}: train {tr.shape} test {te.shape}', flush=True)


def build_noiseless_caches(project_root, layers=('pixels',) + tuple(CONVS)):
    """Noiseless trial caches (inv_snr = 0, every unit kept; suffix '_noiseless') of the pixels
    and the AlexNet layers, written by trial_caches_all with params/fixed_params_eqv_snr0.csv:
    the noise-free responses whose equivariant structure the calibration simulations are built
    from (band_structure of the pixels) and the dashed ceilings of ED Fig 9c."""
    root = Path(project_root)
    model_df = pd.read_csv(root / 'params' / 'model_params.csv', header=0)
    model_df = model_df[model_df['layer'].isin(layers) & model_df['network'].isin(['pixels', 'alexnet'])].reset_index(drop=True)
    params_df = pd.read_csv(root / 'params' / 'fixed_params_eqv_snr0.csv', header=0, index_col=0)
    trial_caches_all(model_df, params_df, root)


def build_noiseless_rel_caches(project_root, layers=CONVS, network='alexnet', chunk=2000):
    """Noiseless responses of exactly the units in the paper's NOISY AlexNet caches (suffix
    '_noiseless_rel'), so that the solid and dashed lines of ED Fig 9e-f differ only by noise.
    Every noisy unit (2-trial mean over the training images) is matched to the all-units
    noiseless cache by response correlation; the matches must be unique and in order (both caches
    keep the layer's unit order). Needs the noisy caches and the '_noiseless' caches."""
    folder = Path(project_root) / 'cache' / network
    for lay in layers:
        out_tr = folder / f'{lay}_resp_trials_train_noiseless_rel.p'
        out_te = folder / f'{lay}_resp_trials_test_noiseless_rel.p'
        if out_tr.exists() and out_te.exists():
            print(f'{network}/{lay}: noiseless_rel caches exist, skipping', flush=True); continue
        noisy = pickle.load(open(folder / f'{lay}_resp_trials_train.p', 'rb'))
        nl_tr = pickle.load(open(folder / f'{lay}_resp_trials_train_noiseless.p', 'rb'))
        nl_te = pickle.load(open(folder / f'{lay}_resp_trials_test_noiseless.p', 'rb'))
        a = noisy.mean(-1).reshape(noisy.shape[0], -1).astype(np.float64)
        b = nl_tr.mean(-1).reshape(nl_tr.shape[0], -1).astype(np.float64)
        a -= a.mean(1, keepdims=True); a /= np.linalg.norm(a, axis=1, keepdims=True)
        b -= b.mean(1, keepdims=True); b /= np.linalg.norm(b, axis=1, keepdims=True)
        best = np.empty(a.shape[0], int); top = np.empty(a.shape[0]); margin = np.empty(a.shape[0])
        for s in range(0, a.shape[0], chunk):           # chunked: the full correlation matrix can be huge
            C = a[s:s + chunk] @ b.T
            j = np.argmax(C, 1); t = C[np.arange(len(j)), j]
            C[np.arange(len(j)), j] = -np.inf
            best[s:s + chunk] = j; top[s:s + chunk] = t; margin[s:s + chunk] = t - C.max(1)
        ok_unique = len(np.unique(best)) == len(best)
        ok_order = bool(np.all(np.diff(best) > 0))
        print(f'{network}/{lay}: matched {len(best)} noisy units -> {nl_tr.shape[0]} noiseless: best corr '
              f'median {np.median(top):.3f} (min {top.min():.3f}), margin min {margin.min():.3f}, '
              f'unique {ok_unique}, order preserved {ok_order}', flush=True)
        if not (ok_unique and ok_order and top.min() > 0.3):
            raise RuntimeError(f'{lay}: unit matching failed (unique {ok_unique}, order {ok_order}, '
                               f'min corr {top.min():.3f}) -- not writing noiseless_rel caches')
        np.savetxt(folder / f'{lay}_noiseless_rel_idx.csv', best, fmt='%d', delimiter=',')
        pickle.dump(nl_tr[best].astype(np.float32), open(out_tr, 'wb'), protocol=4)
        pickle.dump(nl_te[best].astype(np.float32), open(out_te, 'wb'), protocol=4)
        print(f'  saved {out_tr.name} {nl_tr[best].shape}, {out_te.name} {nl_te[best].shape}', flush=True)


def ccnorm_surface(project_root, network, layer, n_list, d_list=D_LIST, n_max=2, noise_model='gaussian', verbose=True):
    """The pooled centered CC_norm of the equivariance model over the (N, d) grid of one
    population (equivariance_validation.pooled_ccnorm_vs_dim_all at every N of n_list, d of
    d_list with d < N): cache/{network}/{layer}_ccnorm_pooled_vsdim_{N}avg{suffix}.csv and the
    per-fit eigen-dimensions {layer}_bands_vsdim_{N}avg{suffix}.csv. noise_model 'gaussian'
    (the data's injected noise), 'noiseless' or 'noiseless_rel' selects the trial caches by
    suffix. Cached (N, d) points are skipped."""
    model_df = pd.DataFrame({'network': [network], 'layer': [layer]})
    for N in n_list:
        d_ok = [int(d) for d in d_list if int(d) < int(N)]
        if not d_ok:
            continue
        if verbose:
            print(f'=== {network}/{layer}  N={N}  d={d_ok}  noise={noise_model} ===', flush=True)
        pooled_ccnorm_vs_dim_all(model_df, _params_single(project_root, N, n_max, noise_model), Path(project_root),
                                 d_range=d_ok, verbose=verbose)


def reliable_count(project_root, network, layer, suffix=''):
    """Number of units in a population's trial caches (the reliably responsive units for the
    data's noise model): the largest N of its (N, d) surface."""
    f = Path(project_root) / 'cache' / network / f'{layer}_resp_trials_train{suffix}.p'
    return int(pickle.load(open(f, 'rb')).shape[0])


def cached_n_levels(project_root, network, layer, suffix=''):
    """N values with a cached (N, d) surface of a population and noise suffix ('', '_noiseless',
    '_noiseless_rel')."""
    import re
    pat = re.compile(rf'^{re.escape(layer)}_ccnorm_pooled_vsdim_(\d+)avg{re.escape(suffix)}\.csv$')
    folder = Path(project_root) / 'cache' / network
    return sorted(int(m.group(1)) for f in folder.glob(f'{layer}_ccnorm_pooled_vsdim_*avg{suffix}.csv')
                  if (m := pat.match(f.name)))


def surface_cc(project_root, network, layer, suffix='', stat='median'):
    """{(N, d): CC_norm} of a population's cached surfaces: the stat ('median', as in the paper's
    pooled figures) over neurons per subsample, then the mean over subsamples."""
    cc = {}
    folder = Path(project_root) / 'cache' / network
    for f in sorted(folder.glob(f'{layer}_ccnorm_pooled_vsdim_*avg{suffix}.csv')):
        if suffix == '' and 'avg_' in f.stem:          # the bare glob also matches the noiseless files
            continue
        N = int(f.stem.split('_vsdim_')[1].split('avg')[0])
        df = pd.read_csv(f); df = df[df['centered'] == True]  # noqa: E712
        samp = getattr(df.groupby(['sample', 'dim'])['cc_norm'], stat)().reset_index()
        for d, v in samp.groupby('dim')['cc_norm'].mean().items():
            cc[(N, int(d))] = float(v)
    return cc


# ---- the equivariant structure of a population at full rank ---------------------------------
def proj_rot_fullrank(x, d):
    """The equivariance estimator (equivariance_functions.calculate_proj_rot_shift: top-d left
    singular vectors of the lag-one cross-rotation matrix M = B A^T, then orthogonal Procrustes
    in that subspace) without forming the N x N matrix: with the thin QR factorizations
    A = Q_A R_A, B = Q_B R_B, M = Q_B (R_B R_A^T) Q_A^T, so the left singular vectors of M are
    Q_B U with U those of the small (n_cond x n_cond) matrix R_B R_A^T. Returns proj, proj_shift,
    rot, rot_shift exactly as calculate_proj_rot_shift."""
    n = x.shape[0]
    shift = x.reshape(n, -1).mean(-1)
    xc = x - shift[:, None, None]
    A = xc.reshape(n, -1)
    B = np.roll(xc, -1, axis=2).reshape(n, -1)
    QA, RA = np.linalg.qr(A); QB, RB = np.linalg.qr(B)
    U, _, _ = np.linalg.svd(RB @ RA.T)
    proj = QB @ U[:, :d]
    A_d = proj.T @ A; B_d = proj.T @ B
    U_d, _, VT_d = np.linalg.svd(B_d @ A_d.T)
    rot = U_d @ VT_d
    if np.linalg.det(rot) < 0:
        U_d[:, -1] = -U_d[:, -1]; rot = U_d @ VT_d
    rot_shift = (rot @ A_d - B_d).mean(1)
    return proj, shift, rot, rot_shift


def band_structure_procrustes(trials, fit='auto', rel_floor=MIN_LEADING_VAR):
    """Per-frequency equivariant structure of a population (Methods, "Calibration simulations
    based on pixels" / "Fit of simulations to data"): the equivariance model fitted at the full
    stimulus rank to all 92 images, the eigenphases rounded to integer multiples of the rotation
    step (frequency f = 0 invariant, 1 tuned, 2..FMAX higher order), and within each frequency's
    degenerate eigenspace the principal components of the projected responses; PCs with less than
    rel_floor of the leading variance are discarded.

    trials : (n_units, 92, n_rot, n_trials) single trials, all images.
    fit : 'avg' -- the subspace, R and the within-band covariance from the TRIAL AVERAGE (exact
          for noise-free caches: pixels, AlexNet noiseless); 'split' -- the subspace and R from
          trial 1 only and the within-band covariance as the symmetrized cross-trial product of
          the two trials' projections (unbiased under independent trial noise; the recorded
          data and the noisy AlexNet caches); 'auto' -- 'avg' when the trials are identical.
    Returns (bands, meta): bands[f] = dict(count, n_resolved, var (per dimension, descending; a
    plane's two dimensions are adjacent), total, share, bulk_edge)."""
    trials = np.asarray(trials, dtype=np.float64)
    n, n_img, n_rot, n_tr = trials.shape
    identical = n_tr == 1 or np.allclose(trials[..., 0], trials[..., 1])
    if fit == 'auto':
        fit = 'avg' if identical else 'split'
    xbar = trials.mean(-1)
    mu = xbar.reshape(n, -1).mean(1); sd = xbar.reshape(n, -1).std(1)
    keep = sd > 0
    norm = lambda a: (a[keep] - mu[keep, None, None]) / sd[keep, None, None]
    xbar = norm(xbar); n = int(keep.sum())
    if fit == 'avg':
        x_fit, X1, X2 = xbar, xbar.reshape(n, -1), xbar.reshape(n, -1)
    else:
        x1 = norm(trials[..., 0]); x2 = norm(trials[..., 1]) if n_tr > 1 else x1
        x_fit, X1, X2 = x1, x1.reshape(n, -1), x2.reshape(n, -1)
    #full stimulus rank (mean removed), even for the eigen-pairing; capped by the unit count
    d = min(2 * ((n_img * n_rot - 1) // 2), 2 * ((n - 1) // 2))
    proj, _, rot, _ = proj_rot_fullrank(x_fit, d)
    ev, V = find_eigendecomp(rot)
    ph = np.abs(ev) / (2 * np.pi / n_rot)
    fq = np.clip(np.round(ph).astype(int), 0, FMAX)
    tot = float((X1 * X2).mean(1).sum())             # total (avg: trial-average; split: cross-trial signal) variance, basis-free
    out = {}
    for f in FREQS:
        sel = fq == f
        if not sel.any():
            out[f] = dict(count=0, n_resolved=0, var=[], total=0.0, bulk_edge=0.0); continue
        W = proj @ V[:, sel]                         # orthonormal basis of the band subspace (n, n_f)
        Y1 = W.T @ X1; Y2 = W.T @ X2
        C = (Y1 @ Y2.T + Y2 @ Y1.T) / (2 * X1.shape[1])
        w = np.sort(np.linalg.eigvalsh(C))[::-1] / tot
        edge = float(abs(min(w.min(), 0.0)))
        n_res = int((w > 2 * edge).sum()) if edge > 0 else int((w > rel_floor * w[0]).sum())
        w = w[w > rel_floor * w[0]] if (len(w) and w[0] > 0) else w[:0]
        out[f] = dict(count=int(len(w)), n_resolved=n_res, var=w.tolist(), total=float(w.sum()), bulk_edge=edge)
    tot_kept = sum(o['total'] for o in out.values())
    for f in out:
        out[f]['share'] = out[f]['total'] / tot_kept if tot_kept > 0 else 0.0
    meta = dict(method='procrustes+pca', fit=fit, n_units=n, n_images=int(n_img), d=int(d), n_trials=int(n_tr),
                trials_identical=bool(identical),
                eigenphase_rms_dev=float(np.sqrt(np.mean((ph - np.round(ph)) ** 2))),
                signal_fraction=float(tot / n))      # split: cross-trial signal fraction of the per-unit variance; avg: 1
    return out, meta


def structure_json(project_root, name):
    """results_extended/<CALIBRATION_DIR>/band_structure_<name>.json"""
    return Path(project_root) / 'results_extended' / CALIBRATION_DIR / f'band_structure_{name}.json'


def band_structure(project_root, name, noisy=None, fit='auto'):
    """Measure and store the per-frequency equivariant structure of a target population from
    its trial caches (all 92 images: the training cache plus the test cache's first trials).
    noisy: the caches with the data's noise (default for every target but the pixels, whose
    structure is measured on the noise-free responses, '_noiseless' caches of
    build_noiseless_caches); the pixels with noisy=True give the Methods' check of the
    split-trial estimator. Writes band_structure_<name>.json (bands '0'..'7' + 'meta') and
    returns (bands, meta). Cached: an existing file is read back."""
    out_f = structure_json(project_root, name)
    if out_f.exists():
        tr = json.load(open(out_f))
        return {int(k): v for k, v in tr.items() if k.isdigit()}, tr.get('meta', {})
    if noisy is None:
        noisy = name != 'pixels'
    net, layer = TARGETS[name]
    suffix = '' if noisy else '_noiseless'
    loaded = load_trials(Path(project_root) / 'cache' / net, layer, suffix)
    if loaded is None:
        raise FileNotFoundError(f'cache/{net}/{layer}_resp_trials_*{suffix}.p missing (trial_caches_all / build_* first)')
    trials = loaded[0]
    print(f'band structure of {name}: {trials.shape} units x images x rotations x trials ({"noisy" if noisy else "noiseless"} caches)', flush=True)
    bands, meta = band_structure_procrustes(trials, fit=fit)
    meta.update(name=name, noisy=bool(noisy))
    out_f.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict({str(k): v for k, v in bands.items()}, meta=meta), open(out_f, 'w'), indent=1)
    print('  ' + '  '.join(f'f={f}: {bands[f]["count"]} dims ({100 * bands[f]["share"]:.1f}%)' for f in FREQS)
          + f'; eigenphase rms dev {meta["eigenphase_rms_dev"]:.3f}', flush=True)
    return bands, meta


# ---- simulated populations built from a measured structure --------------------------------
def truncate_bands(bands, rank):
    """Nested truncation of the bands to their top-`rank` DIMENSIONS by per-dimension variance
    across all bands (a plane counts 2 dims): every band keeps a prefix of its (descending)
    ranks, so rank 20 is a subset of rank 500 is a subset of the full structure. Greedy in
    descending variance, stopping at the first item that does not fit."""
    items = []
    for f in FREQS:
        b = bands[str(f)]
        if not b['count']:
            continue
        nd = 1 if f == 0 else 2
        for j, v in enumerate(b['base'][:int(b['count'])]):
            items.append((float(v), f, j, nd))
    items.sort(key=lambda t: -t[0])
    keep = {f: 0 for f in FREQS}; tot = 0
    for v, f, j, nd in items:
        if tot + nd > rank:
            break
        keep[f] = j + 1; tot += nd
    out = {}
    for f in FREQS:
        b = dict(bands[str(f)]); n = keep[f]
        b['count'] = n; b['base'] = list(b['base'][:n])
        out[str(f)] = b
    return out


def matched_population_spec(project_root, name, iid_list=(0.0,), rank=None):
    """Specification of the simulated populations built from the measured structure of a target
    (band_structure_<name>.json): per frequency band the number of generator dimensions /
    planes and their amplitudes (the measured per-dimension variances, per plane for f >= 1),
    the measured per-band variance shares as the angular profile of the non-equivariant
    remainder, the population size (population_size), and the equivariant signal fractions
    f = 1 - iid of the populations (one population per iid value; cache tag rtruth_c<i>). rank:
    keep only the top-`rank` dimensions (truncate_bands; tag rtruth_r<rank>_c<i>)."""
    tj = structure_json(project_root, name)
    if not tj.exists():
        raise FileNotFoundError(f'{tj} missing: run band_structure(project_root, {name!r}) first')
    tr = json.load(open(tj))
    bands = {}
    for f in FREQS:
        v = np.asarray(tr[str(f)]['var'], float)
        base = v if f == 0 else v[::2]              # per DIM: a plane's two dims are adjacent ranks, and the
                                                    # generator's plane amplitude is the sd per dim
        bands[str(f)] = dict(count=int(len(base)), base=base.tolist())
    if rank is not None:
        bands = truncate_bands(bands, int(rank))
    return dict(name=name, iid_list=[float(v) for v in iid_list], bands=bands, n_pop=population_size(name),
                rank=None if rank is None else int(rank),
                band_profile=[float(tr[str(f)]['share']) for f in FREQS], structure_meta=tr.get('meta', {}))


def population_tag(spec, i):
    """cache layer name of population i of a spec: rtruth_c<i>, or rtruth_r<rank>_c<i>."""
    return f'rtruth_c{i}' if spec['rank'] is None else f'rtruth_r{spec["rank"]}_c{i}'


def cfg_from_spec(project_root, spec, iid, n_pop=None):
    """Generator configuration of a structure-matched population: bands -> (d_inv, inv_amp,
    d_tun, tun_amp, hf_planes, hf_amp) with amplitudes = sqrt of the measured variances, the
    target's measured radius CV delta and gain spread (uniform for the injected-noise
    populations), the remainder fraction iid with the measured angular profile and
    min(n_eq, n_pop - n_eq) remainder dimensions (as many as equivariant ones whenever the
    population is large), the standard analysis geometry (FIT), 2 trials, no decoders."""
    b = spec['bands']
    pars = area_params(project_root)[spec['name']]
    cfg = dict(DEFAULT); cfg.update(FIT)
    cfg.update(dict(n_neurons_tot=int(n_pop or spec['n_pop']), n_trials=2, decoders=False))
    cfg['delta'] = pars['delta']
    cfg['snr_hetero'] = 0.0 if spec['name'] in UNIFORM_SNR else pars['snr_hetero']
    ladder = lambda bf: list(np.sqrt(np.asarray(bf['base'][:int(bf['count'])], float)))
    b0, b1 = b['0'], b['1']
    cfg['d_inv'] = int(b0['count']); cfg['inv_amp'] = ladder(b0) if b0['count'] else None
    cfg['d_tun'] = 2 * int(b1['count']); cfg['tun_amp'] = ladder(b1) if b1['count'] else None
    hf_planes, hf_amp = [], []
    for f in range(2, FMAX + 1):
        bf = b[str(f)]
        hf_planes += [f] * int(bf['count']); hf_amp += ladder(bf)
    cfg['hf_planes'] = tuple(hf_planes); cfg['hf_amp'] = tuple(hf_amp)
    cfg['iid_frac'] = float(iid)
    cfg['iid_band_profile'] = spec.get('band_profile')
    n_eq = cfg['d_inv'] + cfg['d_tun'] + 2 * len(hf_planes)
    cfg['n_iid_dims'] = min(n_eq, cfg['n_neurons_tot'] - n_eq)
    return cfg


def calibrate_noise_direct(cfg, target_inv_snr, target_hetero, n_pop=3000, seed=123, n_iter=0, tol=0.03):
    """Noise of a simulated population from the target's measured per-neuron SNR statistics
    (neuron_selectivity.direct_snr_stats on its reliable neurons: inv_snr = mean noise/signal
    variance ratio, snr_hetero = its spread): inv_snr is injected as measured; the pre-selection
    gain spread is HETERO_TRUNC x the measured post-selection spread (None / 0 for uniform-SNR
    populations). One population of n_pop neurons is built, selected with the pipeline rule and
    measured with the same estimator; the post-selection statistics are returned as checks
    (inv_snr_achieved, hetero_achieved, the reliable fraction, the pooled centred split-half
    correlation = the centred CC_norm ceiling, the median per-stimulus corr_half). n_iter > 0
    iterates the gain spread only (multiplicative) for a tighter spread match."""
    fixed_h = target_hetero is None or not np.isfinite(target_hetero) or target_hetero <= 0
    inv = float(target_inv_snr)
    het = 0.0 if fixed_h else HETERO_TRUNC * float(target_hetero)
    best = None
    for it in range(n_iter + 1):
        c = dict(cfg); c.update(inv_snr=inv, snr_hetero=het, n_neurons_tot=n_pop)
        resp, _ = make_population(c, seed)
        sel, cc = reliable_mask(resp, c['corr_thre'])
        if sel.sum() < 10:
            raise RuntimeError(f'calibrate_noise_direct: only {sel.sum()} reliable neurons at inv_snr={inv:.2f} '
                               f'snr_hetero={het:.2f}')
        st = direct_snr_stats(resp[sel])
        rp = pooled_centred_r_half(resp[sel])
        err = 0.0 if fixed_h else abs(np.log(max(st['snr_hetero'], 1e-3) / target_hetero))
        rec = dict(inv_snr=inv, snr_hetero=het, frac_reliable=float(sel.mean()),
                   inv_snr_achieved=st['inv_snr'], hetero_achieved=st['snr_hetero'],
                   r_half_pooled=float(np.nanmedian(rp)), cc_perstim_med=float(np.median(cc[sel])),
                   err=float(err), n_iter=it)
        if best is None or err < best['err']:
            best = rec
        if fixed_h or it == n_iter or err < tol:
            break
        het *= target_hetero / max(st['snr_hetero'], 1e-3)
    return best


def build_matched_population(project_root, name, spec, i, seed=0):
    """Build population i of a spec (equivariant fraction 1 - spec['iid_list'][i]) at the
    target's measured noise (calibrate_noise_direct), select its reliable neurons, and store
    it as trial caches cache/calib_<name>/<tag>_resp_trials_{train,test}.p (90 / 2 images, as
    the data caches) with <tag>_info.json (the parameters and the post-selection checks).
    Returns (cache folder, tag, n_reliable). A cached population is reused only if it was built
    for the same iid; the index i is the position in the iid list, so extend a sweep by
    appending values."""
    root = Path(project_root)
    iid = float(spec['iid_list'][i])
    net, layer = f'calib_{name}', population_tag(spec, i)
    outdir = root / 'cache' / net; outdir.mkdir(parents=True, exist_ok=True)
    ftr = outdir / f'{layer}_resp_trials_train.p'; fte = outdir / f'{layer}_resp_trials_test.p'
    info_f = outdir / f'{layer}_info.json'
    cfg = cfg_from_spec(root, spec, iid)
    n_eq = cfg['d_inv'] + cfg['d_tun'] + 2 * len(cfg['hf_planes'])
    if ftr.exists() and fte.exists():
        if info_f.exists():
            info = json.load(open(info_f))
            if abs(float(info.get('iid', iid)) - iid) > 1e-6:
                raise RuntimeError(f'{net}/{layer} is cached for iid={info.get("iid")}, not {iid}: append new values '
                                   f'to the iid list instead of reordering it, or remove {outdir}/{layer}_*')
            if int(info.get('n_equivariant', n_eq)) != n_eq or int(info.get('n_pop', cfg['n_neurons_tot'])) != cfg['n_neurons_tot']:
                raise RuntimeError(f'{net}/{layer} is cached for another structure or population size: remove {outdir}/{layer}_*')
        n = pickle.load(open(ftr, 'rb')).shape[0]
        print(f'{net}/{layer}: exists ({n} reliable, iid={iid})', flush=True)
        return net, layer, n
    pars = area_params(root)[name]
    cal = calibrate_noise_direct(cfg, pars['inv_snr'], None if name in UNIFORM_SNR else pars['snr_hetero'])
    cfg['inv_snr'] = cal['inv_snr']; cfg['snr_hetero'] = cal['snr_hetero']
    print(f'{net}/{layer}: building N={cfg["n_neurons_tot"]} iid={iid:.3f} (equivariant dims {n_eq}) '
          f'inv_snr={cal["inv_snr"]:.2f} snr_hetero={cal["snr_hetero"]:.2f}', flush=True)
    resp, _ = make_population(cfg, seed)
    keep, _ = reliable_mask(resp, cfg['corr_thre'])
    resp = resp[keep]
    pickle.dump(resp[:, :90].copy(), open(ftr, 'wb'), protocol=4)
    pickle.dump(resp[:, 90:92].copy(), open(fte, 'wb'), protocol=4)
    json.dump(dict(iid=iid, rank=spec['rank'], inv_snr=cal['inv_snr'], snr_hetero=cal['snr_hetero'],
                   inv_snr_target=pars['inv_snr'], inv_snr_achieved=cal['inv_snr_achieved'],
                   hetero_target=(0.0 if name in UNIFORM_SNR else pars['snr_hetero']),
                   hetero_achieved=cal['hetero_achieved'], r_half_pooled=cal['r_half_pooled'],
                   cc_half=cal['cc_perstim_med'], frac_reliable_calib=cal['frac_reliable'],
                   n_reliable=int(keep.sum()), n_pop=int(len(keep)), d_inv=cfg['d_inv'], d_tun=cfg['d_tun'],
                   n_hf=len(cfg['hf_planes']), n_equivariant=n_eq),
              open(info_f, 'w'), indent=1)
    print(f'  reliable {keep.sum()}/{len(keep)}; post-selection inv_snr {cal["inv_snr_achieved"]:.2f} '
          f'(target {pars["inv_snr"]:.2f}), spread {cal["hetero_achieved"]:.2f}, ceiling {cal["r_half_pooled"]:.3f}', flush=True)
    return net, layer, int(keep.sum())


def target_n_levels(project_root, name):
    """The N at which a target was measured (its cached surfaces), so that its simulated
    populations are compared at the same N; N_LIST when nothing is cached yet."""
    net, layer = TARGETS[name]
    return cached_n_levels(project_root, net, layer) or list(N_LIST)


def run_matched_populations(project_root, name, iid_list=(0.0,), rank=None, index=None, d_list=D_LIST):
    """Build the populations of matched_population_spec(name, iid_list, rank) and run their
    (N, d) surfaces at the target's N levels (up to each population's reliable count) and the d
    of d_list, with 2 neuron subsamples per (N, d) -- the sweep over the equivariant fraction f
    (ED Fig 10a, read off by best_iid / fill_swept) or over (f, R) for the pixels (ED Fig 9d).
    index runs a single population of the list (the unit of an array job)."""
    spec = matched_population_spec(project_root, name, iid_list, rank)
    for i in range(len(spec['iid_list'])):
        if index is not None and i != index:
            continue
        net, layer, n_rel = build_matched_population(project_root, name, spec, i)
        n_use = [N for N in target_n_levels(project_root, name) if N <= n_rel]
        ccnorm_surface(project_root, net, layer, n_use, d_list, n_max=2)


def _rmse(sim, target, dmin=FIT_DMIN, dmax=FIT_DMAX):
    keys = [k for k in target if k in sim and np.isfinite(target[k]) and np.isfinite(sim[k]) and dmin <= k[1] <= dmax]
    if not keys:
        return np.nan, 0
    e = np.array([sim[k] - target[k] for k in keys])
    return float(np.sqrt((e ** 2).mean())), len(keys)


def best_iid(project_root, name):
    """The equivariant fraction of a target: the iid of the structure-matched population whose
    CC_norm surface has the smallest root-mean-square deviation from the target's over the
    (N, d) points with FIT_DMIN <= d <= FIT_DMAX (Methods). Reads cache/calib_<name>/
    rtruth_c<i>_info.json for the iid of every cached population. Returns (iid, rmse, rows)
    with rows = [(i, iid, rmse, n_points)]; iid None when nothing is scored."""
    root = Path(project_root)
    net, layer = TARGETS[name]
    target = surface_cc(root, net, layer)
    folder = root / 'cache' / f'calib_{name}'
    rows = []
    for info_f in sorted(folder.glob('rtruth_c*_info.json'), key=lambda p: int(p.stem.split('_c')[1].split('_')[0])):
        i = int(info_f.stem.split('_c')[1].split('_')[0])
        iid = float(json.load(open(info_f))['iid'])
        sim = surface_cc(root, f'calib_{name}', f'rtruth_c{i}')
        if not sim:
            rows.append((i, iid, np.nan, 0)); continue
        rmse, n = _rmse(sim, target)
        rows.append((i, iid, rmse, n))
    ok = [t for t in rows if np.isfinite(t[2]) and t[3] > 0]
    if not ok:
        return None, np.nan, rows
    _, iid, rmse, _ = min(ok, key=lambda t: t[2])
    return float(iid), float(rmse), rows


def fill_swept(project_root, names=TARGET_ORDER, source=None):
    """Read the minimum of every target's sweep (best_iid) into params/datafit_swept_iid.csv
    (area, iid, source), the file the data-fit simulations read. Targets without a scored
    population keep their existing row; a target with neither raises. Prints the per-population
    RMSE table and returns {area: iid}."""
    import datetime as _dt
    root = Path(project_root)
    csv = root / SWEPT_IID_CSV
    old = {}
    if csv.exists():
        for _, r in pd.read_csv(csv).iterrows():
            old[r['area']] = (float(r['iid']), str(r.get('source', '')))
    source = source or f'sweep_{_dt.date.today().isoformat()}'
    out = []
    for area in TARGET_ORDER:
        iid, rmse, rows = best_iid(root, area) if area in names else (None, np.nan, [])
        tab = '  '.join(f'c{i}: iid {v:.2f} -> rmse {("n/a" if not np.isfinite(e) else f"{e:.4f}")}' for i, v, e, n in rows)
        if iid is not None:
            print(f'{area}: iid = {iid:.2f} (f = {1 - iid:.2f}; rmse {rmse:.4f} over d in [{FIT_DMIN}, {FIT_DMAX}])  [{tab}]')
            out.append((area, iid, source))
        elif area in old:
            if area in names:
                print(f'{area}: no scored population, keeping iid = {old[area][0]:.2f} ({old[area][1]})')
            out.append((area, old[area][0], old[area][1]))
        else:
            raise RuntimeError(f'fill_swept: {area} has no scored population and no row in {csv}')
    text = 'area,iid,source\n' + ''.join(f'{a},{v:g},{s}\n' for a, v, s in out)
    csv.write_text(text)
    return {a: v for a, v, _ in out}


# ----------------------------------------------------------------------------------------
# fit of simulations to data (ED Fig 10): measured parameters and the simulated populations at
# the standard analysis capacity
# ----------------------------------------------------------------------------------------
def measure_subsample(resp, d=FIT['d']):
    """Geometry of the recovered tuned planes of one neuron subsample at signal level: the
    equivariance model fitted at d on the trial average (per-neuron std normalization), and in
    every frequency-1 plane the per-image squared radius as the cross-trial dot product of the
    two trials' projections averaged over rotations (trial noise removed), its mean r and sd
    delta across images -> the radius coefficient of variation delta / r (mean over planes).
    Also the invariant-offset amplitude relative to r (eps_over_r, a record).
    resp: (n, n_stim, n_rot, 2) single trials."""
    x = resp.mean(-1)
    std = x.std(axis=(1, 2), keepdims=True)
    nz = std.squeeze() > 0
    x = x[nz] / std[nz]
    resp_n = resp[nz] / std[nz][..., None]
    n_rot = x.shape[2]
    proj, proj_shift, rot, rot_shift = calculate_proj_rot_shift(x, d)
    evals, evec = find_eigendecomp(rot)
    phase = np.abs(evals) / (2 * np.pi / n_rot)
    B = proj @ evec
    c = proj_shift[:, None, None]
    coords = np.stack([np.einsum('j,jklm->klm', B[:, dcol], resp_n - c[..., None]) for dcol in range(d)])
    inv_off_var, tun_r, tun_delta = [], [], []
    for j in range(d // 2):
        ph = phase[2 * j]
        a = coords[2 * j]
        b = coords[2 * j + 1]
        a0, a1, b0, b1 = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
        if ph < 0.5:                                    # invariant: offset = mean over rotations
            for u0, u1 in [(a0.mean(1), a1.mean(1)), (b0.mean(1), b1.mean(1))]:
                inv_off_var.append(np.cov(u0, u1)[0, 1])    # cross-trial covariance across images
        elif ph < 1.5:                                  # tuned: radius^2 signal = cross-trial power
            r2 = (a0 * a1 + b0 * b1).mean(1)
            radii = np.sqrt(np.clip(r2, 0, None))
            tun_r.append(radii.mean())
            tun_delta.append(radii.std())
    inv_off_var = np.array(inv_off_var)
    eps = np.sqrt(max(inv_off_var.mean(), 0)) if len(inv_off_var) else np.nan
    r = np.mean(tun_r) if tun_r else np.nan
    cv = np.mean([dl / rr for dl, rr in zip(tun_delta, tun_r) if rr > 0]) if tun_r else np.nan
    return dict(n_inv=len(inv_off_var), n_tun=len(tun_r), eps_over_r=eps / r if r else np.nan, delta_over_r=cv)


def _measure_response_set(f, family, mouse, group, corr_thre, n_neurons, n_max):
    resp = np.asarray(pickle.load(open(f, 'rb')))
    keep, _ = reliable_mask(resp, corr_thre)
    idx = np.where(keep)[0]
    _, k_rand, k_samp = subsamples(len(idx), n_neurons, n_max)
    rows = []
    for si in range(k_rand):
        sub = resp[idx[k_samp[si]]]
        m = measure_subsample(sub)
        st = direct_snr_stats(sub)
        rows.append(dict(family=family, mouse=mouse, group=group, subsample=si, n_rel=len(idx),
                         eps_over_r=m['eps_over_r'], delta_over_r=m['delta_over_r'],
                         snr_hetero=st['snr_hetero'], inv_snr=st['inv_snr']))
    print(f'  {group:8s} {mouse:6s} n_rel={len(idx):6d} n_subsamples={k_rand}', flush=True)
    return rows


def datafit_params(project_root, recompute=False):
    """The generator parameters measurable from the data, per population and disjoint subsample
    of n_neurons reliable neurons (fixed_params.csv: n_neurons, n_max, corr_thre): the radius
    coefficient of variation delta / r (measure_subsample), the mean per-neuron noise/signal
    variance ratio inv_snr and its spread snr_hetero (neuron_selectivity.direct_snr_stats) --
    for the pixels, AlexNet conv1-5 and every visual area of every mouse, from their trial
    caches (trial_caches_all). Writes results_extended/data_fit/data_fit_params.csv (read back
    unless recompute) and returns the frame."""
    root = Path(project_root)
    out_dir = root / 'results_extended' / DATAFIT_DIR
    csv = out_dir / 'data_fit_params.csv'
    if csv.exists() and not recompute:
        return pd.read_csv(csv)
    fp = pd.read_csv(root / 'params' / 'fixed_params.csv', header=0, index_col=0)
    n_neurons, n_max, corr_thre = int(fp.loc['n_neurons', 'value']), int(fp.loc['n_max', 'value']), float(fp.loc['corr_thre', 'value'])
    cache = root / 'cache'
    print(f'measuring the data-fit parameters (n_neurons={n_neurons}, n_max={n_max}, corr_thre={corr_thre})', flush=True)
    rows = []
    rows += _measure_response_set(cache / 'pixels' / 'pixels_resp_trials_train.p', 'pixels', '', 'pixels', corr_thre, n_neurons, n_max)
    for conv in CONVS:
        rows += _measure_response_set(cache / 'alexnet' / f'{conv}_resp_trials_train.p', 'conv', '', conv, corr_thre, n_neurons, n_max)
    for mshort, mfull in MICE.items():
        for area in AREAS:
            f = cache / mfull / f'{area}_resp_trials_train.p'
            if not f.exists():
                print(f'  {area:8s} {mshort:6s} MISSING {f}', flush=True)
                continue
            rows += _measure_response_set(f, 'mouse', mshort, area, corr_thre, n_neurons, n_max)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    return df


def area_params(project_root):
    """{target: dict(delta, snr_hetero, inv_snr)}: the means of the measured parameters over the
    subsamples of a population (the visual areas pooled over the three mice)."""
    df = datafit_params(project_root)
    if 'mouse' in df.columns:
        df['mouse'] = df['mouse'].fillna('')
    out = {}
    for name in TARGET_ORDER:
        sub = df[(df['family'] == 'mouse') & (df['group'] == name)] if name in AREAS else df[df['group'] == name]
        if sub.empty:
            raise ValueError(f'no measured parameters for {name} in data_fit_params.csv')
        m = sub[['delta_over_r', 'snr_hetero', 'inv_snr']].mean()
        out[name] = dict(delta=float(m['delta_over_r']), snr_hetero=float(m['snr_hetero']), inv_snr=float(m['inv_snr']))
    return out


def swept_iid(project_root, name):
    """The non-equivariant fraction iid = 1 - f of a target from params/datafit_swept_iid.csv."""
    csv = Path(project_root) / SWEPT_IID_CSV
    if not csv.exists():
        raise FileNotFoundError(f'{csv} missing: run the sweeps (run_matched_populations) and fill_swept first')
    d = pd.read_csv(csv).set_index('area')['iid'].to_dict()
    if name not in d or not np.isfinite(d[name]):
        raise ValueError(f'{name} has no swept fraction in {csv}')
    return float(d[name])


def run_target(project_root, name, seed, n_pop_max=10000, n_max=4, n_pairs=None, verbose=False):
    """One structure-matched population of a target at its swept equivariant fraction, through
    the standard pipeline (N = 250, d = 20, 2 trials) with the decoders of Fig 1d-e (SVM at
    C = 0.001 and 1): the simulated counterpart of the data points of ED Fig 10b. The
    population size is population_size(name) capped at n_pop_max (10,000, enough for n_max
    disjoint subsamples of 250 neurons). Returns the summary row."""
    root = Path(project_root)
    iid = swept_iid(root, name)
    spec = matched_population_spec(root, name, iid_list=[iid])
    cfg = cfg_from_spec(root, spec, iid, n_pop=min(int(n_pop_max), population_size(name)))
    cfg.update(dict(n_max=n_max, n_pairs=n_pairs, decoders=True, svm_c_range=(0.001, 1.0), svm_c=1.0))
    pars = area_params(root)[name]
    hetero_target = None if name in UNIFORM_SNR else pars['snr_hetero']
    cal = calibrate_noise_direct(cfg, pars['inv_snr'], hetero_target)
    cfg['inv_snr'] = cal['inv_snr']; cfg['snr_hetero'] = cal['snr_hetero']
    resp, truth = make_population(cfg, seed)
    summary, _ = run_pipeline(resp, truth, cfg, seed=seed, verbose=verbose)
    summary.update(area=name, area_pooled=name, seed=seed, n_neurons=int(cfg['n_neurons']),
                   d_inv_true=int(cfg['d_inv']), d_tun_true=int(cfg['d_tun']),
                   n_hf_planes=len(cfg['hf_planes']), iid_frac=iid,
                   inv_snr_target=pars['inv_snr'], inv_snr_achieved=cal['inv_snr_achieved'],
                   hetero_target=(0.0 if hetero_target is None else hetero_target),
                   hetero_achieved=cal['hetero_achieved'], r_half_pooled_achieved=cal['r_half_pooled'],
                   cc_perstim_achieved=cal['cc_perstim_med'], calib_err=cal['err'], calib_iter=cal['n_iter'],
                   inv_snr_pop=cal['inv_snr'], snr_hetero_pop=cal['snr_hetero'],
                   frac_reliable_calib=cal['frac_reliable'],
                   signal_fraction=float(spec['structure_meta'].get('signal_fraction', np.nan)),
                   config=json.dumps(dict(name=name, iid=iid, n_pop=int(cfg['n_neurons_tot']), d_inv=int(cfg['d_inv']),
                                          d_tun=int(cfg['d_tun']), n_hf=len(cfg['hf_planes']))))
    return summary


def run_target_cached(project_root, name, seed, **kwargs):
    """run_target cached as cache/<DATAFIT_CACHE_DIR>/<target>_<seed>.csv."""
    out_dir = Path(project_root) / 'cache' / DATAFIT_CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f'{name}_{seed}.csv'
    if f.exists():
        return pd.read_csv(f).iloc[0].to_dict()
    print(f'data-fit simulation {name} seed {seed}', flush=True)
    s = run_target(project_root, name, seed, **kwargs)
    pd.DataFrame([s]).to_csv(f, index=False)
    print(f'  iid={s["iid_frac"]:.2f} d_inv={s.get("d_inv", np.nan):.2f} d_tun={s.get("d_tun", np.nan):.2f} '
          f'cc_norm={s.get("cc_norm_equivariance_c", np.nan):.3f} svm_eqv_C1={s.get("svm_err_eqv_C1")} '
          f'ori_eqv={s.get("ori_err_eqv")}', flush=True)
    return s


def datafit_table(project_root):
    """All cached rows of the data-fit simulations (cache/<DATAFIT_CACHE_DIR>) of the ten
    targets -> results_extended/data_fit/data_fit_sim.csv; returns the frame."""
    root = Path(project_root)
    rows = [pd.read_csv(f).iloc[0].to_dict() for f in sorted((root / 'cache' / DATAFIT_CACHE_DIR).glob('*_*.csv'))]
    if not rows:
        raise FileNotFoundError(f'no runs in cache/{DATAFIT_CACHE_DIR}')
    df = pd.DataFrame(rows)
    df = df[df['area'].isin(TARGET_ORDER)]
    out_dir = root / 'results_extended' / DATAFIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'data_fit_sim.csv', index=False)
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['list', 'run'])
    ap.add_argument('cell', nargs='?')
    ap.add_argument('seed', nargs='?', type=int, default=0)
    ap.add_argument('--n_max', type=int, default=4)
    ap.add_argument('--n_pairs', type=int, default=None)
    args = ap.parse_args()
    if args.what == 'list':
        for k, v in grid().items():
            print(k, v)
        sys.exit(0)
    root = Path(__file__).resolve().parents[2]
    summary = run_cell_cached(root, args.cell, args.seed, n_max=args.n_max, n_pairs=args.n_pairs, verbose=True)
    keys = ['status', 'n_reliable', 'true_d_inv', 'true_d_tun', 'd_inv', 'd_inv_sd', 'd_tun', 'd_tun_sd',
            'cc_norm_equivariance_c', 'true_cc_equivariance_c', 'cc_norm_per_step_c', 'svm_err_best', 'svm_err_eqv_best',
            'ori_err', 'ori_err_eqv']
    print({k: (round(summary[k], 3) if isinstance(summary.get(k), float) else summary.get(k)) for k in keys})
