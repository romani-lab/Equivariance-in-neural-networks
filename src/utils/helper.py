import numpy as np
import pandas as pd
from math import factorial
from scipy.linalg import logm
from scipy.stats import wilcoxon
from scipy.stats import zscore 
from sklearn.metrics import roc_auc_score


def calculate_rel_center_dist(x):
    """
    Calculate relative center distance.

    Parameters
    ----------
    x : np.ndarray
        neural responses of shape (n_neurons, 2, n_rot)

    Returns
    -------
    rel_center : float
        relative center distance
    """
    norm = np.linalg.norm(x - x.mean(axis=2, keepdims=True), axis=0).mean()
    rel_center = np.linalg.norm(x[:,0].mean(1) - x[:,1].mean(1)) / norm

    return rel_center


def calculate_norm_dists(x):
    """
    Calculate normalized distance to center.

    Parameters
    ----------
    x : np.ndarray
        neural responses of shape (n_neurons, n_stim, n_rot)

    Returns
    -------
    norm_dist_mean : float
        mean normalized distance to center
    norm_dist_std : float
        std of normalized distance to center
    """
    n_neurons = x.shape[0]
    center = x.mean(axis=2, keepdims=True)

    # divide by sqrt(n_neurons) to make distance independent of dimensionality
    norm_dist_mean = np.linalg.norm(x - center, axis=0).mean() / np.sqrt(n_neurons)
    norm_dist_std = np.sqrt((np.linalg.norm(x - center, axis=0).std(axis=1)**2).mean()) / np.sqrt(n_neurons)

    return norm_dist_mean, norm_dist_std


def calculate_effect_size(df):
    """
    Calculate effect size statistics between two paired samples in df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the paired samples with columns "score_1" and "score_2"

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame containing the effect size statistics:
        - mean_delta: mean difference between score_1 and score_2
        - d_paired: Cohen's d for paired samples
        - delta_ci_low: lower bound of 95% confidence interval for mean difference
        - delta_ci_high: upper bound of 95% confidence interval for mean difference
        - auc: Area Under the Curve (AUC) for the two scores
        - auc_ci_low: lower bound of 95% confidence interval for AUC
        - auc_ci_high: upper bound of 95% confidence interval for AUC
    """
    delta = df["score_1"] - df["score_2"]

    mean_delta = delta.mean()
    sd_delta = delta.std(ddof=1)
    d_paired = mean_delta / sd_delta

    rng = np.random.default_rng(0)
    n_boot = 10000
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(delta, size=len(delta), replace=True)
        boot_means[i] = sample.mean()
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    scores = np.concatenate([-df["score_1"].values, -df["score_2"].values])
    labels = np.concatenate([
        np.zeros(len(df)),
        np.ones(len(df))
    ])
    auc = roc_auc_score(labels, scores)

    n = len(df)
    boot_auc = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        scores_b = np.concatenate([
            -df["score_1"].values[idx],
            -df["score_2"].values[idx]
        ])
        labels_b = np.concatenate([
            np.zeros(n),
            np.ones(n)
        ])
        boot_auc[i] = roc_auc_score(labels_b, scores_b)
    auc_ci_low, auc_ci_high = np.percentile(boot_auc, [2.5, 97.5])

    results_df = pd.DataFrame({
        "mean_delta": [mean_delta],
        "d_paired": [d_paired],
        "delta_ci_low": [ci_low],
        "delta_ci_high": [ci_high],
        "auc": [auc],
        "auc_ci_low": [auc_ci_low],
        "auc_ci_high": [auc_ci_high]
    })

    return results_df


def calculate_cc_norm(x, x_pred, cc_max, tol=1e-6):
    """
    Calculate normalized correlation coefficient across last dimension, as defined in     
    https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2016.00010/full
    If cc_max is 1, this reduces to the standard Pearson correlation coefficient

    Parameters
    ----------
    x : np.ndarray
        trial-averaged neural responses of shape (n_neurons, n_stim, n_rot)
    x_pred : np.ndarray
        predicted neural responses of shape (n_neurons, n_stim, n_rot)
    cc_max : np.ndarray
        maximum correlation coefficient of shape (n_neurons, n_stim)
    tol : float, default=1e-6
        tolerance for max. standard deviation across rotations, below which the neuron gets a cc_norm of NaN

    Returns
    -------
    cc_norm : np.ndarray
        normalized correlation coefficient of shape (n_neurons, n_stim)
    """
    n_neurons, _, n_rot = x.shape

    x_norm = zscore(x, axis=2, ddof=1)
    x_norm = x_norm/cc_max[:,:,np.newaxis]
    x_norm = x_norm.reshape((-1,n_rot))

    x_pred_norm = zscore(x_pred, axis=2, ddof=1)
    x_pred_norm = x_pred_norm.reshape((-1,n_rot))

    cc_norm = np.einsum('ij,ij->i', x_norm, x_pred_norm)/(n_rot-1)
    cc_norm = cc_norm.reshape((n_neurons,-1))

    idx_noresp = np.where(x.std(-1).max(-1) < tol)[0]
    cc_norm[idx_noresp] = np.nan
    
    return cc_norm


def compute_significance(results_df, alternative='greater'):
    """
    Compute one-sided pairwise Wilcoxon signed-rank test statistics and 
    p-values for the given results_df.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing the results with columns 'area' and 'score'.
    alternative : str, default='greater'
        Defines the one-sidedness of the alternative hypothesis.

    Returns
    -------
    stat_df : pd.DataFrame
        DataFrame containing the Wilcoxon signed-rank test statistics and p-values.
    """
    #tested pairs: the pre-specified families of the Online Methods -- neighbouring stages along
    #the expected hierarchy, mouse V1>LM, V1>RL, LM>RL, LM>AL, RL>AL and the adjacent AlexNet
    #layers conv1>conv2, conv2>conv3, conv3>conv4, conv4>conv5
    MOUSE_PAIRS = [('V1', 'LM'), ('V1', 'RL'), ('LM', 'RL'), ('LM', 'AL'), ('RL', 'AL')]
    ALEXNET_PAIRS = [('conv1', 'conv2'), ('conv2', 'conv3'), ('conv3', 'conv4'), ('conv4', 'conv5')]

    areas = set(results_df['area'].unique())
    rows = []
    for pairs, needed in ((MOUSE_PAIRS, {'V1', 'LM', 'RL', 'AL'}),
                          (ALEXNET_PAIRS, {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'})):
        if not needed.issubset(areas):
            continue
        for area0, area1 in pairs:
            stat, p = wilcoxon(results_df[results_df['area'] == area0]['score'],
                               results_df[results_df['area'] == area1]['score'],
                               alternative=alternative)
            rows.append({'area0': area0, 'area1': area1, 'stat': stat, 'p': p})

    return pd.DataFrame(rows, columns=['area0', 'area1', 'stat', 'p'])


def find_eigendecomp(rot, tol=1e-10):
    """
    Find the real eigenphases and eigenvectors of the rotation matrix rot.
    The eigenphases are ordered by magnitude.
    The corresponding real eigenvectors come in pairs.
    Has only been tested for even-dimensional rotation matrices.

    Parameters
    ----------
    rot : np.ndarray
        special orthogonal matrix of shape (d, d)
    tol : float, default=1e-10
        tolerance for eigenvalues to be considered 0, in which case the corresponding 
        complex eigenvectors need not be complex conjugates of each other

    Returns
    -------
    evals : np.ndarray
        ordered eigenphases of shape (d,)
    evecs : np.ndarray
        ordered real eigenvectors of shape (d, d), each column is an eigenvector
    """
    dp = rot.shape[0]

    #find the traceless, antisymmetric generator matrix GM corresponding to the orthogonal matrix rot
    GM = -1j*logm(rot)
    #GM should be antisymmetric
    Gs = (GM-GM.T)/2

    #eigenvalues of Gs are the eigenphases of the corresponding rotation matrix
    ws, v = np.linalg.eig( Gs )
    #order by absolute value
    I = np.argsort(np.abs(ws))
    evals = np.real(ws[I])
    v = v[:,I]
    #order pairs such that negative first then positive
    for i in range( len(I)//2 ):
        if (np.abs(evals[2*i]) - np.abs(evals[2*i+1]) < 1e-6) & (evals[2*i] > 0):
            evals[2*i], evals[2*i+1] = evals[2*i+1], evals[2*i]
            v[:,2*i], v[:,2*i+1] = v[:,2*i+1].copy(), v[:,2*i].copy()

    #convert complex eigenvectors, which come in complex conjugate pairs, to real eigenvectors
    #if the eigenvalue is close to zero, we take the real part of the eigenvector
    #if the eigenvalue is not close to zero, we take the real and imaginary
    #parts of the eigenvector and multiply by sqrt(2) to ensure normalization
    evecs = np.zeros(v.shape)
    for i in range(int(dp/2)):
        if (np.abs(evals)[2*i]<tol)|(np.abs(evals)[2*i+1]<tol):
            evecs[:,2*i] = np.abs(v[:,2*i])/np.linalg.norm(np.abs(v[:,2*i]))
            evecs[:,2*i+1] = np.abs(v[:,2*i+1])/np.linalg.norm(np.abs(v[:,2*i+1]))
        else:        
            evecs[:,2*i] = np.sqrt(2)*np.real(v[:,2*i])
            evecs[:,2*i+1] = np.sqrt(2)*np.imag(v[:,2*i])

    return evals, evecs


def find_n_unique_perm(n_tot, n_perm):
    """
    Find n_perm unique permutations of n_tot integers.

    Parameters
    ----------
    n_tot : int
        number of integers to permute
    n_perm : int
        number of permutations

    Returns
    -------
    id_all : np.ndarray
        array of shape (n_perm, n_tot)
    """
    if n_perm > factorial(n_tot):
        raise ValueError("n_perm exceeds number of possible permutations")
    
    rng = np.random.default_rng(seed=4)
    seen = set()
    while len(seen) < n_perm:
        seen.add(tuple(rng.permutation(n_tot)))
    id_all = np.array(list(seen))

    return id_all


def find_kn_unique_samples(k, n, N):
    """
    From the integers 0 to n-1,
    select N non-overlapping samples of k integers each.

    Parameters
    ----------
    k : int
        number of integers per sample
    n : int
        total number of integers available (0 to n-1)
    N : int
        number of samples

    Returns
    -------
    id_all : np.ndarray
        array of shape (N, k) containing the sampled indices
    """
    if N * k > n:
        raise ValueError(f"Need N*k <= n, but got N*k={N*k} > n={n}.")
        
    rng = np.random.default_rng(seed=4)
    id_all = rng.permutation(n)[:N*k].reshape(N, k)

    return id_all


def load_var_dim(var_f):
    """
    Read a {layer}_var_dim_{N}avg_{d}dim*.csv cache written by equivariance_summary_all.

    Current caches hold one row per (stim, sample, dim). Older caches (one rank-averaged
    spectrum per held-out pair, without 'sample'/'dim' columns) are given sample = 0 and
    dim = row rank so that downstream code treats them as a single fit per pair.

    Parameters
    ----------
    var_f : str or pathlib.Path
        path to the cache file

    Returns
    -------
    var_df : pd.DataFrame
        columns 'stim', 'sample', 'dim', 'evals', 'var train', 'var test', 'pc train', 'pc test'
    """
    var_df = pd.read_csv(var_f)
    if 'sample' not in var_df.columns:
        var_df['sample'] = 0
    if 'dim' not in var_df.columns:
        var_df['dim'] = var_df.groupby(['stim', 'sample']).cumcount()
    return var_df


def band_summary(var_df, keys=(), inv_max=0.5, tuned_max=1.5, pc_weighted=False):
    """
    Per-fit invariant/tuned band summary of a var_dim frame (one row per dimension).

    One fit = one neuron subsample on one held-out pair, i.e. one group of
    keys + ['stim', 'sample']. Band membership is by |eigenphase| in units of the rotation
    step: invariant |evals| < inv_max, tuned inv_max <= |evals| < tuned_max. Counts and
    variances are aggregated per fit and only then averaged across fits: averaging sorted
    spectra rank by rank across subsamples would smear an eigenplane across rank slots.

    Parameters
    ----------
    var_df : pd.DataFrame
        as returned by load_var_dim (optionally with extra key columns, e.g. 'model', 'area')
    keys : sequence of str, default=()
        extra grouping columns preceding 'stim' and 'sample'
    inv_max : float, default=0.5
        upper edge of the invariant band
    tuned_max : float, default=1.5
        upper edge of the tuned band
    pc_weighted : bool, default=False
        if True, the band Pearson correlations ('pc test inv/tun', 'pc train inv/tun') are the
        VARIANCE-WEIGHTED mean over the band's dimensions -- each dimension's held-out
        correlation weighted by the variance it explains (pc test by var test, pc train by
        var train) -- the paper's degrees of invariance and tuning (Fig. 4b) -- so dominant
        dimensions count more than weak/noise ones. If False (default), a plain mean over
        band dimensions.
        The band dimension counts and variance sums are unaffected.

    Returns
    -------
    fit_df : pd.DataFrame
        one row per fit with columns keys + ['stim', 'sample', 'd_inv', 'd_tun',
        'var train inv', 'var train tun', 'var test inv', 'var test tun' (band sums),
        'pc train inv', 'pc train tun', 'pc test inv', 'pc test tun' (band means, NaN when
        the band is empty -- or, when pc_weighted, when the band's weight sum is zero)]
    """
    keys = list(keys)
    df = var_df.copy()
    for q in ['var train', 'var test', 'pc train', 'pc test']:
        if q not in df.columns:
            df[q] = np.nan
    a = np.abs(df['evals'].values)
    df['inv'] = (a < inv_max).astype(float)
    df['tun'] = ((a >= inv_max) & (a < tuned_max)).astype(float)
    for q in ['var train', 'var test', 'pc train', 'pc test']:
        df[f'{q} inv'] = df[q] * df['inv']
        df[f'{q} tun'] = df[q] * df['tun']
    # variance-weighted band correlation: numerator = sum over band dims of pc * weight,
    # denominator = the band variance sum (already computed below). Weight pc test by
    # var test and pc train by var train.
    pc_weight = {'pc train': 'var train', 'pc test': 'var test'}
    if pc_weighted:
        for q in ['pc train', 'pc test']:
            w = pc_weight[q]
            df[f'{q}w inv'] = df[q] * df[w] * df['inv']
            df[f'{q}w tun'] = df[q] * df[w] * df['tun']
    g = df.groupby(keys + ['stim', 'sample'], sort=True)
    fit_df = g[['inv', 'tun']].sum().rename(columns={'inv': 'd_inv', 'tun': 'd_tun'})
    for q in ['var train', 'var test']:
        fit_df[f'{q} inv'] = g[f'{q} inv'].sum()
        fit_df[f'{q} tun'] = g[f'{q} tun'].sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        for q in ['pc train', 'pc test']:
            if pc_weighted:
                w = pc_weight[q]
                fit_df[f'{q} inv'] = g[f'{q}w inv'].sum() / fit_df[f'{w} inv'].replace(0, np.nan)
                fit_df[f'{q} tun'] = g[f'{q}w tun'].sum() / fit_df[f'{w} tun'].replace(0, np.nan)
            else:
                fit_df[f'{q} inv'] = g[f'{q} inv'].sum() / fit_df['d_inv'].replace(0, np.nan)
                fit_df[f'{q} tun'] = g[f'{q} tun'].sum() / fit_df['d_tun'].replace(0, np.nan)
    return fit_df.reset_index()


def freq_band_summary(var_df):
    """Per-rotation-frequency held-out band shares of a var_dim frame (one row per dimension).

    Every recovered dimension is assigned to its integer rotation frequency (|eigenphase|
    rounded, in units of the rotation step): 0 = invariant, 1 = frequency-1 tuned, 2/3/... =
    higher-frequency equivariant. Unlike band_summary (which lumps everything at |phase| >= 1.5
    into "out of band"), this keeps the higher harmonics resolved so an all-frequency
    equivariant fraction and per-frequency generator planes can be built from them
    (used by the data-fit simulations).

    Returns
    -------
    freq_df : pd.DataFrame
        one row per frequency, columns 'freq', 'd' (mean # recovered dims at that frequency,
        over fits = stim x sample), 's' (mean held-out var-test share at that frequency)
    d_all : float
        mean total # recovered dims per fit (= the fit dimension d)
    s_all : float
        mean total held-out var-test share per fit (all frequencies)
    """
    df = var_df.copy()
    df['freq'] = np.round(np.abs(df['evals'].values)).astype(int)
    per_fit = df.groupby(['stim', 'sample', 'freq']).agg(d=('dim', 'count'),
                                                         s=('var test', 'sum'))
    freq_df = per_fit.groupby('freq').mean().reset_index()
    tot = df.groupby(['stim', 'sample']).agg(d=('dim', 'count'), s=('var test', 'sum')).mean()
    return freq_df, float(tot['d']), float(tot['s'])


def band_summary_per_stim(var_df, keys=(), inv_max=0.5, tuned_max=1.5, pc_weighted=False):
    """
    band_summary averaged over neuron subsamples: one row per keys + ['stim'] with the mean
    over samples of every band quantity, plus the across-subsample standard deviation in
    columns suffixed ' sd' (NaN for a single subsample).

    Parameters
    ----------
    var_df, keys, inv_max, tuned_max, pc_weighted : see band_summary

    Returns
    -------
    stim_df : pd.DataFrame
    """
    keys = list(keys)
    fit_df = band_summary(var_df, keys=keys, inv_max=inv_max, tuned_max=tuned_max,
                          pc_weighted=pc_weighted)
    cols = [c for c in fit_df.columns if c not in keys + ['stim', 'sample']]
    g = fit_df.groupby(keys + ['stim'], sort=True)[cols]
    stim_df = g.mean()
    sd_df = g.std().rename(columns={c: f'{c} sd' for c in cols})
    return pd.concat([stim_df, sd_df], axis=1).reset_index()


def band_summary_table(var_df, keys=(), inv_max=0.5, tuned_max=1.5, pc_weighted=False):
    """
    Summary of the band quantities per keys group, with three dispersion measures:
    'mean' (over all fits), 'sd fit' (over single fits, i.e. one subsample on one held-out
    pair -- the variability of a single 250-neuron estimate), 'sd sample' (across-subsample
    sd, averaged over held-out pairs) and 'sem stim' (sem across held-out pairs of the
    per-pair means, the error bar of the figures).

    Parameters
    ----------
    var_df, keys, inv_max, tuned_max, pc_weighted : see band_summary

    Returns
    -------
    table_df : pd.DataFrame
        one row per keys group and quantity, columns keys + ['quantity', 'mean', 'sd fit',
        'sd sample', 'sem stim', 'n fits', 'n samples', 'n stim']
    """
    keys = list(keys)
    fit_df = band_summary(var_df, keys=keys, inv_max=inv_max, tuned_max=tuned_max,
                          pc_weighted=pc_weighted)
    cols = [c for c in fit_df.columns if c not in keys + ['stim', 'sample']]
    stim_df = band_summary_per_stim(var_df, keys=keys, inv_max=inv_max, tuned_max=tuned_max,
                                    pc_weighted=pc_weighted)
    rows = []
    groups = fit_df.groupby(keys, sort=True) if keys else [((), fit_df)]
    for name, gf in groups:
        name = name if isinstance(name, tuple) else (name,)
        gs = stim_df
        for k, v in zip(keys, name):
            gs = gs[gs[k] == v]
        for c in cols:
            rows.append(dict(zip(keys, name)) | {
                'quantity': c,
                'mean': gf[c].mean(),
                'sd fit': gf[c].std(),
                'sd sample': gs[f'{c} sd'].mean(),
                'sem stim': gs[c].std() / np.sqrt(gs['stim'].nunique()),
                'n fits': gf[c].notna().sum(),
                'n samples': gf['sample'].nunique(),
                'n stim': gf['stim'].nunique()})
    return pd.DataFrame(rows)


def pooled_scores(pred, avg, trials, center=False, n_perm=20, seed=0):
    """
    Per-neuron prediction scores pooled over all (stimulus, rotation) conditions.

    Parameters
    ----------
    pred : np.ndarray
        model predictions of shape (n_neurons, n_stim, n_rot)
    avg : np.ndarray
        trial-averaged responses of the same shape
    trials : np.ndarray
        single trials of shape (n_neurons, n_stim, n_rot, n_trials)
    center : bool, default=False
        if True, the mean over rotations of each stimulus (from the trial average) is
        subtracted from pred, avg and every trial, so that only the within-stimulus tuning
        curves are scored ("centered"); split halves used for the ceiling are centred by
        their own per-stimulus means. If False, the between-stimulus structure is scored too.
    n_perm : int, default=20
        random balanced trial splits averaged for the split-half ceiling (1 split if
        n_trials == 2)
    seed : int, default=0

    Returns
    -------
    scores : dict of np.ndarray, each of shape (n_neurons,)
        'cc' Pearson correlation between pred and avg over all conditions;
        'r_half' pooled split-half correlation of the trial averages;
        'cc_max' = sqrt(2 r_half / (1 + r_half)) (NaN when r_half <= 0), the noise ceiling
        of Schoppe et al. (2016); 'cc_norm' = cc / cc_max;
        'feve' fraction of explainable variance explained,
        1 - (MSE_trials - var_noise) / (var_trials - var_noise), with MSE_trials the mean
        squared error of pred against single trials, var_noise the mean over conditions of
        the unbiased across-trial variance, var_trials the variance of the single trials
        over all conditions and trials (NaN when var_trials <= var_noise);
        'mse', 'var_signal' (= var_trials - var_noise), 'var_noise'.
    NaN predictions (neurons without a fit) give NaN scores.
    """
    n = pred.shape[0]
    n_trials = trials.shape[-1]
    P = pred.reshape(n, -1).astype(float)
    A = avg.reshape(n, -1).astype(float)
    T = trials.reshape(n, A.shape[1], n_trials).astype(float)
    if center:
        mu = avg.mean(2, keepdims=True)                                  #(n, n_stim, 1)
        P = (pred - mu).reshape(n, -1)
        A = (avg - mu).reshape(n, -1)
        T = (trials - mu[..., np.newaxis]).reshape(n, A.shape[1], n_trials)

    def _corr(x, y):
        with np.errstate(invalid='ignore', divide='ignore'):
            xc = x - np.nanmean(x, 1, keepdims=True)
            yc = y - np.nanmean(y, 1, keepdims=True)
            return np.nansum(xc * yc, 1) / np.sqrt(np.nansum(xc**2, 1) * np.nansum(yc**2, 1))

    valid = ~np.isnan(P).any(1)
    cc = np.full(n, np.nan)
    cc[valid] = _corr(P[valid], A[valid])

    #pooled split-half ceiling on the trial averages (halves centred per stimulus if center)
    rng = np.random.default_rng(seed)
    half = n_trials // 2
    n_perm = 1 if n_trials == 2 else n_perm
    r_half = np.zeros(n)
    for _ in range(n_perm):
        p = rng.permutation(n_trials) if n_trials > 2 else np.arange(n_trials)
        ta = trials[..., p[:half]].mean(-1)
        tb = trials[..., p[half:2 * half]].mean(-1)
        if center:
            ta = ta - ta.mean(2, keepdims=True)
            tb = tb - tb.mean(2, keepdims=True)
        r_half += _corr(ta.reshape(n, -1), tb.reshape(n, -1)) / n_perm
    with np.errstate(invalid='ignore', divide='ignore'):
        cc_max = np.sqrt(2 * r_half / (1 + r_half))
    cc_max[~(r_half > 0)] = np.nan
    cc_norm = cc / cc_max

    #FEVE against single trials
    with np.errstate(invalid='ignore', divide='ignore'):
        mse = np.nanmean((T - P[:, :, np.newaxis])**2, axis=(1, 2))
        var_noise = T.var(axis=2, ddof=1).mean(1)
        var_trials = T.reshape(n, -1).var(1)
        var_signal = var_trials - var_noise
        feve = 1 - (mse - var_noise) / var_signal
    feve[~(var_signal > 0)] = np.nan
    feve[~valid] = np.nan

    #regression gain of the trial average on the prediction (1 = calibrated amplitude);
    #a gain far below 1 explains FEVE << cc_norm**2 (the prediction is confined to the
    #fitted subspace and under-shoots the response amplitude)
    with np.errstate(invalid='ignore', divide='ignore'):
        Pc = P - np.nanmean(P, 1, keepdims=True)
        Ac = A - np.nanmean(A, 1, keepdims=True)
        gain = np.nansum(Pc * Ac, 1) / np.nansum(Pc**2, 1)
    gain[~valid] = np.nan
    return {'cc': cc, 'r_half': r_half, 'cc_max': cc_max, 'cc_norm': cc_norm, 'feve': feve,
            'mse': mse, 'var_signal': var_signal, 'var_noise': var_noise, 'gain': gain}
