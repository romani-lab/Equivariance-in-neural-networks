import numpy as np


def fit_multiplicative_gain_svd(x, eps=1e-6, mask_low_f=True, f_thresh=1e-3):
    """
    Fit x_{N,s,r} = f_{N,s} * (1 + a_N * g_{s,r}).

    Parameters
    ----------
    x : np.ndarray
        response array of shape (n_neurons, n_stim, n_trials)
    eps : float, default=1e-6
        small constant added to denominators for numerical stability
    mask_low_f : bool, default=True
        whether to mask entries whose trial-mean response is below f_thresh
    f_thresh : float, default=1e-3
        minimum absolute trial-mean response used when mask_low_f is True

    Returns
    -------
    a : np.ndarray
        neuron gain loadings of shape (n_neurons,)
    g : np.ndarray
        stimulus-by-trial gain factors of shape (n_stim, n_trials)
    f : np.ndarray
        trial-mean responses of shape (n_neurons, n_stim)
    """
    N, n_stim, n_trials = x.shape
    f = x.mean(axis=2)

    if mask_low_f:
        valid_ns = np.abs(f) >= f_thresh
    else:
        valid_ns = np.ones_like(f, dtype=bool)

    f_exp = f[:, :, None]
    denom = np.where(np.abs(f_exp) < f_thresh,
                     np.sign(f_exp) * f_thresh, f_exp) if mask_low_f else (f_exp + eps)
    Y = x / (denom + eps) - 1.0
    Y = Y - Y.mean(axis=2, keepdims=True)

    T = n_stim * n_trials
    Ymat = Y.reshape(N, T)
    valid_mask = np.repeat(valid_ns[:, :, None], n_trials, axis=2).reshape(N, T)
    if mask_low_f:
        Ymat = np.where(valid_mask, Ymat, 0.0)

    U, S, Vt = np.linalg.svd(Ymat, full_matrices=False)
    a = U[:, 0] * np.sqrt(S[0])
    g_flat = Vt[0, :] * np.sqrt(S[0])

    anorm = np.linalg.norm(a) + 1e-12
    a /= anorm
    g_flat *= anorm

    g = g_flat.reshape(n_stim, n_trials)
    g = g - g.mean(axis=1, keepdims=True)

    if np.mean(a) < 0:
        a = -a
        g = -g

    return a, g, f


def ev_additive_residuals_by_rank(x, max_rank=15):
    """
    SVD on additive residuals (x - trial_mean).
    EV_k = cumsum(S^2) / sum(S^2).

    Parameters
    ----------
    x : np.ndarray
        response array of shape (n_neurons, n_stim, n_trials)
    max_rank : int, default=15
        number of singular values to compute

    Returns
    -------
    cumvar : np.ndarray
        cumulative explained variance for ranks 1 through max_rank
    """
    from scipy.sparse.linalg import svds

    N, n_stim, n_trials = x.shape
    f = x.mean(axis=2)
    resid = (x - f[:, :, None]).reshape(N, n_stim * n_trials)

    U, S_top, Vt = svds(resid, k=max_rank)
    S_top = S_top[::-1]  # descending

    total_var = np.sum(resid**2)
    cumvar = np.cumsum(S_top**2) / total_var
    return cumvar
