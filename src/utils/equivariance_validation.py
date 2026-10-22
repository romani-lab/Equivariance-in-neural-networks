"""
Validation of the equivariance model on single-trial responses: pooled CC_norm of the
equivariance model and of its control models, shuffle nulls, and the CC_norm-vs-fitted-
dimension sweep.

- trial_caches_all: single-trial response caches (the trial averages and split-half
  correlations written by save_average_resp_n_cc are not enough for pooled noise ceilings).
- pooled_ccnorm_all: per fit (neuron subsample x leave-two-out pair) of the SAME equivariance
  model as equivariance_summary_all, the held-out chain predictions are pooled over all
  held-out images and scored per neuron, within-image centered, with a pooled split-half
  ceiling (CC_norm; helper.pooled_scores). The control models -- fully invariant (R = I) and
  per-rotation (one orthogonal map per rotation step) -- are scored identically, and shuffles
  of the training labels give the null distributions of CC_norm and of the per-dimension
  eigenphases and correlations (the dashed lines of Fig 3b).
- pooled_ccnorm_subset_all: the same for the scenes or the textures only (within-class
  leave-one-image-out folds; SI Fig 9).
- pooled_ccnorm_vs_dim_all: pooled CC_norm of the equivariance model as a function of the
  fitted dimension d, with the per-dimension quantities of every fit (ED Fig 9a, c-h).

Conventions shared with equivariance_functions: per-neuron division by the training std,
d-dimensional projection, chain predictions from the reference orientation, neuron
subsamples from find_kn_unique_samples (identical to the quantification's), leave-two-out
pairs (one scene + one texture image, the last pair the two 80x images).
"""
import pickle as pkl

import numpy as np
import pandas as pd

from utils.helper import find_kn_unique_samples, find_eigendecomp, pooled_scores
from utils.equivariance_functions import calculate_proj_rot_shift, calculate_predictions, per_dim_quantities
from utils.model_loader import model_train_test_responses, is_model_row
from utils.data_loader import mouse_all_train_test_responses, find_area_neurons

#models scored by pooled_ccnorm_all / pooled_ccnorm_subset_all: the equivariance model (R^k from
#the reference orientation), the fully invariant model (R = I) and the per-rotation model (one
#orthogonal map per rotation step)
MODEL_NAMES = ('equivariance', 'invariant', 'per_step')


# ----------------------------------------------------------------------------------------
# single-trial caches
# ----------------------------------------------------------------------------------------
def trial_caches_all(model_df, params_df, project_root, dtype=np.float32):
    """
    Write single-trial response caches for all rows of model_df, mirroring the loading of
    example_equiv_projs (same reliable-neuron selection and row order as the
    {layer}_avg_resp_*.p caches):
        {layer}_resp_trials_train{suffix}.p : (n_neurons, n_stim_train, n_rot, n_trials_train)
        {layer}_resp_trials_test{suffix}.p  : (n_neurons, n_stim_test, n_rot, n_trials_test)
    Rows whose caches exist are skipped. Arrays are stored as dtype (float32 by default).
    """
    stim_name = params_df.loc['stimulus'].value
    corr_thre = float(params_df.loc['corr_thre'].value)
    inv_snr = float(params_df.loc['inv_snr'].value)
    noise_model = params_df.loc['noise_model'].value
    noise_suffix = '' if noise_model == 'gaussian' else f'_{noise_model}'
    data_root = project_root / 'data'
    stim_root = data_root / 'stimuli'
    retinotopy_root = data_root / 'retinotopy'

    resp_train_trials0 = resp_test_trials0 = idx_resp = retinotopy_fn = None
    loaded_model = None
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        monitor = model_df['monitor'].values[i0]
        cache_folder = project_root / 'cache' / model_name
        cache_folder.mkdir(parents=True, exist_ok=True)
        train_f = cache_folder / f'{layer_name}_resp_trials_train{noise_suffix}.p'
        test_f = cache_folder / f'{layer_name}_resp_trials_test{noise_suffix}.p'
        if train_f.exists() and test_f.exists():
            print(f'{model_name} {layer_name}: trial caches exist, skipping', flush=True)
            continue
        print(f'{model_name} {layer_name}: writing trial caches', flush=True)

        if is_model_row(model_name):
            resp_train_trials, resp_test_trials, _, idx = model_train_test_responses(
                model_name, layer_name, stim_root, stim_name, monitor, corr_thre, cache_folder,
                inv_snr=inv_snr, noise_model=noise_model)
            resp_train_trials = resp_train_trials[idx]
            resp_test_trials = resp_test_trials[idx]
        else:
            if loaded_model != model_name:
                date = model_name[-10:]
                resp_train_trials0, resp_test_trials0, _, idx_resp = mouse_all_train_test_responses(
                    data_root / date, model_name, stim_root, stim_name, monitor, corr_thre, cache_folder)
                resp_train_trials0 = resp_train_trials0[idx_resp]
                resp_test_trials0 = resp_test_trials0[idx_resp]
                retinotopy_fn = retinotopy_root / f'{model_name}_{stim_name}.npz'
                loaded_model = model_name
            idx_area = find_area_neurons(retinotopy_fn, layer_name, idx_resp)
            resp_train_trials = resp_train_trials0[idx_area]
            resp_test_trials = resp_test_trials0[idx_area]

        #consistency with the average caches used by the quantification
        avg_f = cache_folder / f'{layer_name}_avg_resp_train{noise_suffix}.p'
        if avg_f.exists():
            x_avg = pkl.load(open(avg_f, 'rb'))
            err = np.abs(x_avg - resp_train_trials.mean(-1)).max()
            if not np.allclose(x_avg, resp_train_trials.mean(-1), atol=1e-5, rtol=1e-4):
                print(f'  WARNING: trial averages differ from cached averages (max |diff| {err:.3g})', flush=True)
        pkl.dump(resp_train_trials.astype(dtype), open(train_f, 'wb'), protocol=4)
        pkl.dump(resp_test_trials.astype(dtype), open(test_f, 'wb'), protocol=4)
        print(f'  saved {train_f.name} {resp_train_trials.shape}, {test_f.name} {resp_test_trials.shape}', flush=True)


# ----------------------------------------------------------------------------------------
# shuffles of the training data (nulls)
# ----------------------------------------------------------------------------------------
def shuffle_train(x, kind, rng):
    """
    Shuffle condition labels of trial-averaged training responses x (n_neurons, n_stim, n_rot).

    kind='rot'  : rotations permuted independently within each image -- destroys the
                  rotation-transition (tuned) structure, keeps image identity
    kind='img'  : image labels permuted independently at each rotation -- destroys the
                  image correspondence across rotations (invariant structure), keeps the
                  per-rotation response distribution
    kind='both' : all (image, rotation) conditions permuted jointly -- destroys both (the
                  shuffle control of the paper)
    """
    xs = x.copy()
    n_stim, n_rot = x.shape[1], x.shape[2]
    if kind == 'rot':
        for i in range(n_stim):
            xs[:, i] = xs[:, i][:, rng.permutation(n_rot)]
    elif kind == 'img':
        for r in range(n_rot):
            xs[:, :, r] = xs[:, rng.permutation(n_stim), r]
    elif kind == 'both':
        perm = rng.permutation(n_stim * n_rot)
        xs = xs.reshape(x.shape[0], -1)[:, perm].reshape(x.shape)
    else:
        raise ValueError(f'unknown shuffle kind {kind!r}')
    return xs


# ----------------------------------------------------------------------------------------
# control models in the fitted subspace
# ----------------------------------------------------------------------------------------
def _fit_per_step(x_train, proj, proj_shift):
    """
    Per-rotation-step orthogonal maps in the fitted d-dimensional subspace: for every
    rotation index k = 1..n_rot-1 an orthogonal Procrustes map R_k (and shift) from the
    projected reference-orientation responses to the projected responses at rotation k,
    fitted on the training images. The equivariance model (R^k, one shift) is the
    constrained special case; this is the per-rotation control model of the paper.

    Returns
    -------
    rots : list of (d, d) arrays, index k (index 0 is the identity)
    shifts : list of (d,) arrays
    """
    d = proj.shape[1]
    n_rot = x_train.shape[2]
    A = proj.T @ (x_train[:, :, 0] - proj_shift[:, np.newaxis])
    rots, shifts = [np.eye(d)], [np.zeros(d)]
    for k in range(1, n_rot):
        B = proj.T @ (x_train[:, :, k] - proj_shift[:, np.newaxis])
        U, _, VT = np.linalg.svd(B @ A.T)
        R = U @ VT
        if np.linalg.det(R) < 0:
            U[:, -1] = -U[:, -1]
            R = U @ VT
        rots.append(R)
        shifts.append((R @ A - B).mean(1))
    return rots, shifts


def predict_models(x, proj, proj_shift, rot, rot_shift, per_step=None):
    """
    Predictions of the held-out manifolds under the equivariance model (chain from the
    reference orientation, as calculate_predictions), the fully invariant model (R = I: the
    projected reference response at every rotation) and, if per_step is given, the
    per-rotation model of _fit_per_step.

    Returns
    -------
    preds : dict of name -> (n_neurons, n_stim, n_rot)
    """
    _, n_stim, n_rot = x.shape
    x_proj = np.einsum('ji,jkl->ikl', proj, x - proj_shift[:, np.newaxis, np.newaxis])
    back = lambda p: np.einsum('ij,jkl->ikl', proj, p) + proj_shift[:, np.newaxis, np.newaxis]

    preds = {'equivariance': calculate_predictions(x, proj, proj_shift, rot, rot_shift)}
    preds['invariant'] = back(np.repeat(x_proj[:, :, [0]], n_rot, axis=2))
    if per_step is not None:
        rots, shifts = per_step
        ps = np.zeros_like(x_proj)
        for k in range(n_rot):
            ps[:, :, k] = rots[k] @ x_proj[:, :, 0] - shifts[k][:, np.newaxis]
        preds['per_step'] = back(ps)
    return preds


# ----------------------------------------------------------------------------------------
# pooled CC_norm, control models, nulls
# ----------------------------------------------------------------------------------------
def _score_rows(preds, avg, trials, rot_idx, sample, model_tag, neuron):
    """Per-neuron pooled (within-image centered) scores for every model in preds on the
    rotations rot_idx; one frame per model with 'neuron', 'sample', 'model', 'centered', 'tag'."""
    rows = []
    for name, pred in preds.items():
        sc = pooled_scores(pred[:, :, rot_idx], avg[:, :, rot_idx], trials[:, :, rot_idx], center=True)
        df = pd.DataFrame(sc)
        df['neuron'] = neuron
        df['sample'] = sample
        df['model'] = name
        df['centered'] = True
        df['tag'] = model_tag
        rows.append(df)
    return rows


def load_trials(cache_folder, layer_name, noise_suffix):
    """Single-trial caches of a row as 2-trial averages and trials for all 92 images (test
    images: first 2 of their 80 trials, as the quantification's test2x caches).

    Returns trials_all (n, 92, 15, 2), x_all_avg (n, 92, 15), n_train, n_rot, or None when the
    caches are missing."""
    train_f = cache_folder / f'{layer_name}_resp_trials_train{noise_suffix}.p'
    test_f = cache_folder / f'{layer_name}_resp_trials_test{noise_suffix}.p'
    if not (train_f.exists() and test_f.exists()):
        return None
    tr_trials = pkl.load(open(train_f, 'rb')).astype(np.float64)      #(n, 90, 15, 2)
    te_trials = pkl.load(open(test_f, 'rb')).astype(np.float64)       #(n, 2, 15, 80)
    n_train, n_rot, n_tr = tr_trials.shape[1:]
    trials_all = np.concatenate((tr_trials, te_trials[:, :, :, :n_tr]), axis=1)     #(n, 92, 15, 2)
    return trials_all, trials_all.mean(-1), n_train, n_rot


def subsamples(n_all_neurons, n_neurons, n_max):
    """Disjoint neuron subsamples, as in the quantification."""
    n_neurons_l = min(n_neurons, n_all_neurons)
    k_rand = int(np.min([np.max([n_all_neurons // n_neurons_l, 1]), n_max]))
    return n_neurons_l, k_rand, find_kn_unique_samples(n_neurons_l, n_all_neurons, k_rand)


def pooled_ccnorm_all(model_df, params_df, project_root, n_shuffle=10, shuffle_kinds=('rot', 'img', 'both'),
                      seed=0, verbose=True):
    """
    Pooled CC_norm of the equivariance model and its control models, and shuffle nulls, for
    all rows of model_df. Requires the trial caches (trial_caches_all) and the average
    caches (example_equiv_projs).

    Per row and leave-two-out pair, for every neuron subsample (as in the quantification):
    the models are fitted on the trial-averaged training images, the held-out pair's
    manifolds are predicted, and the held-out predictions / trial averages / single trials
    are collected per neuron. After all pairs, every neuron has predictions for all 92
    images x 15 rotations, which are scored in one go (helper.pooled_scores, within-image
    centered) over the 14 non-reference rotations (the reference orientation seeds the
    predictions).

    Outputs in cache/{model}/ (N = n_neurons, d = n_procs, suffix = noise model):
    - {layer}_ccnorm_pooled_{N}avg_{d}dim{suffix}.csv : per neuron, model ('equivariance',
      'invariant', 'per_step') and tag ('data', or the shuffle kind with its index for the
      equivariance model refitted on shuffled training data): 'cc', 'r_half', 'cc_max',
      'cc_norm', 'feve', 'mse', 'var_signal', 'var_noise', 'gain'.
    - {layer}_var_dim_null_{N}avg_{d}dim{suffix}.csv : per (kind, shuffle, stim, sample,
      dim) 'evals', 'var test', 'pc test' of fits on shuffled training data -- the nulls of
      the dimension counts, band variances and band correlations of the quantification
      (kind 'both': image and rotation labels permuted jointly, the shuffle control of
      Fig 3b; 'rot' / 'img': only the rotation / only the image labels).
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    n_max = int(params_df.loc['n_max'].value)
    d = int(params_df.loc['n_procs'].value)
    noise_model = params_df.loc['noise_model'].value
    noise_suffix = '' if noise_model == 'gaussian' else f'_{noise_model}'

    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / 'cache' / model_name
        tag = f'{n_neurons}avg_{d}dim{noise_suffix}'
        out_pooled = cache_folder / f'{layer_name}_ccnorm_pooled_{tag}.csv'
        out_null = cache_folder / f'{layer_name}_var_dim_null_{tag}.csv'
        if out_pooled.exists() and out_null.exists():
            print(f'{model_name} {layer_name}: pooled caches exist, skipping', flush=True)
            continue
        loaded = load_trials(cache_folder, layer_name, noise_suffix)
        if loaded is None:
            print(f'{model_name} {layer_name}: no trial caches, run trial_caches_all first', flush=True)
            continue
        print(f'{model_name} {layer_name}: pooled CC_norm / controls / nulls', flush=True)
        trials_all, x_all_avg, n_train, n_rot = loaded
        n_all = x_all_avg.shape[1]
        n_neurons_l, k_rand, k_samp = subsamples(x_all_avg.shape[0], n_neurons, n_max)
        pair_idx = [(i, n_train // 2 + i) for i in range(n_train // 2)] + [(n_train, n_train + 1)]
        rot_idx = np.arange(1, n_rot)
        rng = np.random.default_rng(seed)

        #containers: held-out predictions per model, per subsample (neurons x 92 x 15)
        pred_all = {k: {m: np.full((n_neurons_l, n_all, n_rot), np.nan) for m in MODEL_NAMES} for k in range(k_rand)}
        pred_null = {k: {(kind, s): np.full((n_neurons_l, n_all, n_rot), np.nan)
                         for kind in shuffle_kinds for s in range(n_shuffle)} for k in range(k_rand)}
        null_rows = []

        for stim, (a, b) in enumerate(pair_idx):
            mask = (np.arange(n_all) != a) & (np.arange(n_all) != b)
            for k in range(k_rand):
                idx = k_samp[k]
                x_train = x_all_avg[idx][:, mask]
                x_test = x_all_avg[idx][:, [a, b]]
                std_norm = x_train.std(axis=(1, 2), keepdims=True)
                nz = std_norm.squeeze() > 0
                x_train_k = x_train[nz] / std_norm[nz]
                x_test_k = x_test[nz] / std_norm[nz]

                proj, proj_shift, rot, rot_shift = calculate_proj_rot_shift(x_train_k, d)
                per_step = _fit_per_step(x_train_k, proj, proj_shift)
                preds = predict_models(x_test_k, proj, proj_shift, rot, rot_shift, per_step=per_step)
                #store in original (unnormalized) units so that pooling across folds does
                #not mix the folds' scalings
                for m in MODEL_NAMES:
                    pred_all[k][m][nz, a] = preds[m][:, 0] * std_norm[nz][:, :, 0]
                    pred_all[k][m][nz, b] = preds[m][:, 1] * std_norm[nz][:, :, 0]

                #shuffle nulls: refit on shuffled training data, predict the held-out pair
                for kind in shuffle_kinds:
                    for s in range(n_shuffle):
                        x_sh = shuffle_train(x_train_k, kind, rng)
                        proj_s, shift_s, rot_s, rshift_s = calculate_proj_rot_shift(x_sh, d)
                        pred_s = calculate_predictions(x_test_k, proj_s, shift_s, rot_s, rshift_s)
                        pred_null[k][(kind, s)][nz, a] = pred_s[:, 0] * std_norm[nz][:, :, 0]
                        pred_null[k][(kind, s)][nz, b] = pred_s[:, 1] * std_norm[nz][:, :, 0]
                        evals_s, eigenvec_s = find_eigendecomp(rot_s)
                        evals_s = evals_s / (2 * np.pi / n_rot)
                        B_s = proj_s @ eigenvec_s
                        s_te = (B_s.T @ x_test_k.reshape(x_test_k.shape[0], -1)).std(1)
                        var_s = s_te**2 / (x_test_k.reshape(x_test_k.shape[0], -1).std(1)**2).sum()
                        #per-dimension correlation between held-out data and prediction
                        xp = np.einsum('ji,jkl->ikl', B_s, x_test_k - shift_s[:, np.newaxis, np.newaxis]).reshape(d, -1)
                        pp = np.einsum('ji,jkl->ikl', B_s, pred_s - shift_s[:, np.newaxis, np.newaxis]).reshape(d, -1)
                        xp = xp - xp.mean(1, keepdims=True); pp = pp - pp.mean(1, keepdims=True)
                        with np.errstate(invalid='ignore', divide='ignore'):
                            pc = (xp * pp).sum(1) / np.sqrt((xp**2).sum(1) * (pp**2).sum(1))
                        null_rows.append(pd.DataFrame({'kind': kind, 'shuffle': s, 'stim': stim, 'sample': k,
                                                       'dim': np.arange(d), 'evals': evals_s,
                                                       'var test': var_s, 'pc test': pc}))
            if verbose and (stim % 10 == 0):
                print(f'  pair {stim + 1}/{len(pair_idx)}', flush=True)

        #pooled scoring per neuron (all 92 held-out images); predictions were stored in original
        #units, so the per-neuron correlation is scale-free across folds
        pooled = []
        for k in range(k_rand):
            idx = k_samp[k]
            avg = x_all_avg[idx]
            trials = trials_all[idx]
            pooled += _score_rows(pred_all[k], avg, trials, rot_idx, k, 'data', idx)
            for (kind, s), pred in pred_null[k].items():
                pooled += _score_rows({'equivariance': pred}, avg, trials, rot_idx, k, f'{kind}{s}', idx)
        pd.concat(pooled, ignore_index=True).to_csv(out_pooled, index=False)
        if null_rows:
            pd.concat(null_rows, ignore_index=True).to_csv(out_null, index=False)
        print(f'  saved {out_pooled.name}, {out_null.name}', flush=True)


def pooled_ccnorm_subset_all(model_df, params_df, project_root, subset, verbose=True):
    """
    Class-subset (scenes / textures) pooled CC_norm of the equivariance model and its
    controls (SI Fig 9). Mirrors pooled_ccnorm_all but with the WITHIN-CLASS leave-one-out
    folds of equivariance_summary_all(subset=...): the class has 46 images (45 training +
    the 80x image of that class); for each held-out class image the models are fitted on the
    other 45 class images and the held-out image's manifold is predicted. After all 46 folds
    every class image has a held-out prediction, and the pooled per-neuron scores
    (helper.pooled_scores, within-image centered) are computed in one go over the 46
    held-out class images. Neuron subsamples are drawn identically to pooled_ccnorm_all.

    subset : 'scenes' or 'textures'
    Output: cache/{model}/{layer}_ccnorm_pooled_{N}avg_{d}dim{suffix}_{subset}.csv
    """
    if subset not in ('scenes', 'textures'):
        raise ValueError(f"subset must be 'scenes' or 'textures', got {subset!r}")
    n_neurons = int(params_df.loc['n_neurons'].value)
    n_max = int(params_df.loc['n_max'].value)
    d = int(params_df.loc['n_procs'].value)
    noise_model = params_df.loc['noise_model'].value
    noise_suffix = '' if noise_model == 'gaussian' else f'_{noise_model}'

    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / 'cache' / model_name
        tag = f'{n_neurons}avg_{d}dim{noise_suffix}'
        out_pooled = cache_folder / f'{layer_name}_ccnorm_pooled_{tag}_{subset}.csv'
        if out_pooled.exists():
            print(f'{model_name} {layer_name}: {subset} pooled cache exists, skipping', flush=True)
            continue
        loaded = load_trials(cache_folder, layer_name, noise_suffix)
        if loaded is None:
            print(f'{model_name} {layer_name}: no trial caches, run trial_caches_all first', flush=True)
            continue
        print(f'{model_name} {layer_name}: {subset} pooled CC_norm / controls', flush=True)
        trials_all, x_all_avg, n_train, n_rot = loaded
        n_all = x_all_avg.shape[1]
        n_neurons_l, k_rand, k_samp = subsamples(x_all_avg.shape[0], n_neurons, n_max)

        #within-class image indices and leave-one-out folds (mirror equivariance_summary_all):
        #scenes = 45 training scenes + the 80x scene; textures = 45 training textures + 80x texture
        keep = (np.concatenate((np.arange(0, n_train // 2), [n_train])) if subset == 'scenes'
                else np.concatenate((np.arange(n_train // 2, n_train), [n_train + 1])))
        rot_idx = np.arange(1, n_rot)
        pred_all = {k: {m: np.full((n_neurons_l, n_all, n_rot), np.nan) for m in MODEL_NAMES}
                    for k in range(k_rand)}

        for stim, held in enumerate(keep):
            mask = np.isin(np.arange(n_all), keep) & (np.arange(n_all) != held)
            for k in range(k_rand):
                idx = k_samp[k]
                x_train = x_all_avg[idx][:, mask]
                x_test = x_all_avg[idx][:, [held]]
                std_norm = x_train.std(axis=(1, 2), keepdims=True)
                nz = std_norm.squeeze() > 0
                x_train_k = x_train[nz] / std_norm[nz]
                x_test_k = x_test[nz] / std_norm[nz]
                proj, proj_shift, rot, rot_shift = calculate_proj_rot_shift(x_train_k, d)
                per_step = _fit_per_step(x_train_k, proj, proj_shift)
                preds = predict_models(x_test_k, proj, proj_shift, rot, rot_shift, per_step=per_step)
                #store unnormalized so pooling across folds does not mix the folds' scalings
                for m in MODEL_NAMES:
                    pred_all[k][m][nz, held] = preds[m][:, 0] * std_norm[nz][:, :, 0]
            if verbose and (stim % 15 == 0):
                print(f'  image {stim + 1}/{len(keep)}', flush=True)

        #pooled scoring per neuron over the 46 held-out class images
        pooled = []
        for k in range(k_rand):
            idx = k_samp[k]
            preds_keep = {m: pred_all[k][m][:, keep] for m in MODEL_NAMES}
            pooled += _score_rows(preds_keep, x_all_avg[idx][:, keep], trials_all[idx][:, keep], rot_idx, k, 'data', idx)
        pd.concat(pooled, ignore_index=True).to_csv(out_pooled, index=False)
        print(f'  saved {out_pooled.name}', flush=True)


# ----------------------------------------------------------------------------------------
# pooled CC_norm vs fitted dimension
# ----------------------------------------------------------------------------------------
def pooled_ccnorm_vs_dim_all(model_df, params_df, project_root, d_range=None, verbose=True):
    """
    Pooled (within-image centered) CC_norm of the equivariance model as a function of the
    fitted subspace dimension d. Same folds, neuron subsamples and scoring as
    pooled_ccnorm_all, but only the equivariance model, refitted at every d of d_range
    (default 2, 4, ..., 60); the n_procs point reproduces the 'equivariance' rows of
    {layer}_ccnorm_pooled_*.csv.

    Outputs in cache/{model}/:
    - {layer}_ccnorm_pooled_vsdim_{N}avg{suffix}.csv : one row per (dim, sample, neuron) with
      'cc', 'cc_max', 'cc_norm', 'feve' (ED Fig 9a, c, e, g);
    - {layer}_bands_vsdim_{N}avg{suffix}.csv : the per-dimension quantities of every fit --
      one row per (dim = fitted d, sample, stim = held-out pair, idx = eigen index) with
      'evals' (eigenphase in units of the rotation step), 'var train' / 'var test' (share of
      the total training / held-out variance along that eigen-dimension) and 'pc test'
      (held-out correlation along it), as in the var_dim caches of equivariance_summary_all
      but over the whole (N, d) grid (the per-band structure the simulations are calibrated to).
    When both files exist, only the d's of d_range they do not yet contain are computed and
    APPENDED (so a grid can be widened without recomputing; nothing to do -> skipped).
    Requires the trial caches (trial_caches_all).
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    n_max = int(params_df.loc['n_max'].value)
    noise_model = params_df.loc['noise_model'].value
    noise_suffix = '' if noise_model == 'gaussian' else f'_{noise_model}'
    if d_range is None:
        d_range = list(range(2, 61, 2))
    d_range = [int(v) for v in d_range]
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / 'cache' / model_name
        out_f = cache_folder / f'{layer_name}_ccnorm_pooled_vsdim_{n_neurons}avg{noise_suffix}.csv'
        bands_f = cache_folder / f'{layer_name}_bands_vsdim_{n_neurons}avg{noise_suffix}.csv'
        old_cc = old_bands = None
        d_todo = list(d_range)
        if out_f.exists() and bands_f.exists():
            old_cc = pd.read_csv(out_f); old_bands = pd.read_csv(bands_f)
            have = set(old_cc['dim'].unique()) & set(old_bands['dim'].unique())
            d_todo = [d0 for d0 in d_range if d0 not in have]
            if not d_todo:
                print(f'{model_name} {layer_name}: vs-dim caches exist with all d, skipping', flush=True)
                continue
            print(f'{model_name} {layer_name}: appending d = {d_todo} to the existing vs-dim caches', flush=True)
        loaded = load_trials(cache_folder, layer_name, noise_suffix)
        if loaded is None:
            print(f'{model_name} {layer_name}: no trial caches, run trial_caches_all first', flush=True)
            continue
        print(f'{model_name} {layer_name}: pooled CC_norm vs d for d in {d_todo}', flush=True)
        trials_all, x_all_avg, n_train, n_rot = loaded
        n_all = x_all_avg.shape[1]
        n_neurons_l, k_rand, k_samp = subsamples(x_all_avg.shape[0], n_neurons, n_max)
        pair_idx = [(i, n_train // 2 + i) for i in range(n_train // 2)] + [(n_train, n_train + 1)]
        rot_idx = np.arange(1, n_rot)
        pred_all = {(k, d0): np.full((n_neurons_l, n_all, n_rot), np.nan) for k in range(k_rand) for d0 in d_todo}
        band_rows = []
        for stim, (a, b) in enumerate(pair_idx):
            mask = (np.arange(n_all) != a) & (np.arange(n_all) != b)
            for k in range(k_rand):
                idx = k_samp[k]
                x_train = x_all_avg[idx][:, mask]
                x_test = x_all_avg[idx][:, [a, b]]
                std_norm = x_train.std(axis=(1, 2), keepdims=True)
                nz = std_norm.squeeze() > 0
                x_train_k = x_train[nz] / std_norm[nz]
                x_test_k = x_test[nz] / std_norm[nz]
                for d0 in d_todo:
                    proj, proj_shift, rot, rot_shift = calculate_proj_rot_shift(x_train_k, d0)
                    pred = calculate_predictions(x_test_k, proj, proj_shift, rot, rot_shift)
                    pred_all[(k, d0)][nz, a] = pred[:, 0] * std_norm[nz][:, :, 0]
                    pred_all[(k, d0)][nz, b] = pred[:, 1] * std_norm[nz][:, :, 0]
                    ev, v_tr, v_te, _, pc_te = per_dim_quantities(x_train_k, x_test_k, proj, proj_shift,
                                                                  rot, rot_shift, pc_train=False)
                    band_rows.append(pd.DataFrame({'sample': k, 'stim': stim, 'dim': d0, 'idx': np.arange(d0),
                                                   'evals': ev, 'var train': v_tr, 'var test': v_te,
                                                   'pc test': pc_te}))
            if verbose and stim % 10 == 0:
                print(f'  pair {stim + 1}/{len(pair_idx)}', flush=True)
        rows = []
        for k in range(k_rand):
            idx = k_samp[k]
            avg = x_all_avg[idx][:, :, rot_idx]
            trials = trials_all[idx][:, :, rot_idx]
            for d0 in d_todo:
                sc = pooled_scores(pred_all[(k, d0)][:, :, rot_idx], avg, trials, center=True)
                df = pd.DataFrame({q: sc[q] for q in ['cc', 'cc_max', 'cc_norm', 'feve']})
                df['neuron'] = idx
                df['sample'] = k
                df['dim'] = d0
                df['centered'] = True
                rows.append(df)
        cc_df = pd.concat(rows, ignore_index=True); bd_df = pd.concat(band_rows, ignore_index=True)
        if old_cc is not None:
            cc_df = pd.concat([old_cc, cc_df], ignore_index=True).sort_values(['dim', 'sample']).reset_index(drop=True)
            bd_df = pd.concat([old_bands, bd_df], ignore_index=True).sort_values(['dim', 'sample', 'stim']).reset_index(drop=True)
        cc_df.to_csv(out_f, index=False)
        bd_df.to_csv(bands_f, index=False)
        print(f'  saved {out_f.name}, {bands_f.name}', flush=True)
