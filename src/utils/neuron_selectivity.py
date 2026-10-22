import numpy as np
from scipy.stats import zscore 
import pickle as pkl


def save_average_resp_n_cc(resp_train_trials, resp_test_trials, cache_folder, layer_name, noise_suffix=''):
    """
    Save average responses and half correlations for training and testing images.
    For testing images, save one version including all trials and another with only the first two trials.
    Files are saved in cache_folder and their filename starts with layer_name.

    Parameters
    ----------
    resp_train_trials : np.ndarray
        array of size (n_neurons, n_stim_train, n_rot, n_trials_train) containing responses
    resp_test_trials : np.ndarray
        array of size (n_neurons, n_stim_test, n_rot, n_trials_test) containing responses
    cache_folder : str
        path to the folder where results will be saved
    layer_name : str
        name of the layer for which responses are saved
    noise_suffix : str, default=''
        suffix appended to all cache filenames so non-gaussian noise models do not
        clobber the default gaussian caches (e.g. '_mult_gain')

    Returns
    -------
    None
    """
    avg_train_f = cache_folder / f'{layer_name}_avg_resp_train{noise_suffix}.p'
    pkl.dump(resp_train_trials.mean(-1), open( avg_train_f, "wb" ) )

    avg_test_f = cache_folder / f'{layer_name}_avg_resp_test{noise_suffix}.p'
    pkl.dump(resp_test_trials.mean(-1), open( avg_test_f, "wb" ) )

    avg_test2x_f = cache_folder / f'{layer_name}_avg_resp_test2x{noise_suffix}.p'
    pkl.dump(resp_test_trials[:,:,:,:2].mean(-1), open( avg_test2x_f, "wb" ) )

    cc_train_f = cache_folder / f'{layer_name}_cc_half_train{noise_suffix}.csv'
    cc_half_train2x = corr_half(resp_train_trials).mean(-1)
    np.savetxt(cc_train_f, cc_half_train2x, delimiter=',')

    cc_test_f = cache_folder / f'{layer_name}_cc_half_test{noise_suffix}.csv'
    cc_half_test = corr_half(resp_test_trials).mean(-1)
    np.savetxt(cc_test_f, cc_half_test, delimiter=',')

    cc_test2x_f = cache_folder / f'{layer_name}_cc_half_test2x{noise_suffix}.csv'
    cc_half_test2x = corr_half(resp_test_trials[:,:,:,:2]).mean(-1)
    np.savetxt(cc_test2x_f, cc_half_test2x, delimiter=',')


def reliable_neurons(resp_train_trials, cc_half_train_f, corr_thre, n_perm=100):
    """
    Find indices of reliably responsive neurons based on split-half correlation and nonzero variance.

    Parameters
    ----------
    resp_train_trials : np.ndarray
        array of size (n_neurons, n_stim_train, n_rot, n_trials_train) containing responses
    cc_half_train_f : str
        path to cache file containing half correlation
    corr_thre : float
        threshold for correlation
    n_perm : int, default=100
        number of random samples for computing corr_half

    Returns
    -------
    idx_resp : np.ndarray
        array of indices of responsive neurons
    """
    if cc_half_train_f.exists():
        cc_half_train = np.loadtxt(cc_half_train_f, delimiter=',')
    else:
        cc_half_train = corr_half(resp_train_trials, n_perm=n_perm).mean(1).mean(1)
        np.savetxt(cc_half_train_f, cc_half_train, delimiter=',')

    #find responsive neurons
    std_train = resp_train_trials.mean(-1).reshape((resp_train_trials.shape[0], -1)).std(1)
    idx_resp = np.where( (cc_half_train>corr_thre) & (std_train>0) )[0]

    return idx_resp


def corr_half(resp, n_perm=100, tol=1e-6):
    """
    Compute the correlation for two halves of the trials 
    for n_perm choices of halves.

    Parameters
    ----------
    resp : np.ndarray
        array of size (n_neurons, n_stim, n_rot, n_trials)
    n_perm : int, default=100
        number of random samples
    tol : float, default=1e-6
        tolerance for standard deviation

    Returns
    -------
    cc_half : np.ndarray
        array of size (n_neurons, n_stim, n_perm)
    """
    np.random.seed(4)

    n_neurons, n_stim, n_rot, n_trials = resp.shape
    n_half = int(n_trials/2)
    cc_half = np.zeros((n_neurons, n_stim, n_perm))

    #if std<tol, set cc_half to 1
    idx_const = np.where( resp.std(-1).max(1)<tol )[0]
    cc_half[idx_const] = 1

    #go through neurons with std>=tol
    idx_nonconst = np.array([i for i in range(n_neurons) if i not in idx_const])
    if idx_nonconst.shape[0]>0:        
        for ct in range(n_perm):
            idx_perm = np.random.permutation(n_trials)
            resp_perm_tmp = resp[idx_nonconst][:,:,:,idx_perm]
            for i in range(n_stim):
                resp_half = np.zeros((2, n_rot, idx_nonconst.shape[0]))
                resp_half[0] = resp_perm_tmp[:,i,:,:n_half].mean(-1).T
                resp_half[1] = resp_perm_tmp[:,i,:,n_half:].mean(-1).T
                resp_tmp = zscore(resp_half, 1)
                cc_half[idx_nonconst,i,ct] = (resp_tmp[0] * resp_tmp[1]).mean(0)

    return cc_half


def direct_snr_stats(resp, cc_max=0.999):
    """
    Per-neuron single-trial SNR statistics from the first two trials, on the WHOLE response
    vector (all stimuli x rotations). This is the exact inverse of the simulation's noise
    definition (utils.simulation.make_population: noise_sd = sqrt(inv_snr) x the neuron's signal
    std, gain multiplying the signal only, so SNR_i = g_i^2 / inv_snr), and the estimator of
    gain_model_analyses.ipynb (inv_snr = (1 - cc) / cc of the whole-vector split-half cc).

    Parameters
    ----------
    resp : np.ndarray
        array of size (n_neurons, n_stim, n_rot, n_trials) single trials; trials 0 and 1 are used
    cc_max : float, default=0.999
        neurons whose implied whole-vector split-half cc = 1 / (1 + inv_snr) exceeds this are
        (near-)noiseless units and are excluded from the per-neuron statistics

    Returns
    -------
    stats : dict
        'inv_snr'        mean over neurons of inv_snr_i = noise_var_i / signal_var_i (the value the
                         simulation injects; mean, not median, as in the notebook estimate)
        'inv_snr_median' median of inv_snr_i
        'snr_hetero'     0.5 * sd over neurons of log(inv_snr_i) = sd of log g_i, the simulation's
                         lognormal gain spread. Uniform-SNR populations (pixels / ANN layers with
                         injected noise) return the estimator's own floor (~0.07), not 0
        'inv_snr_rom'    ratio of means, mean(noise_var) / mean(signal_var) (population-level
                         noise fraction, tail-weighted)
        'mag_sd'         sd over neurons of log(signal std): the per-neuron MAGNITUDE spread. This
                         is what the earlier data-fit called gain_sd; it is not an SNR spread (it is
                         ~0.7-0.95 for the uniform-SNR pixels / ANN layers)
        'n'              neurons entering the per-neuron statistics
    """
    n = resp.shape[0]
    t0 = resp[..., 0].reshape(n, -1)
    t1 = resp[..., 1].reshape(n, -1)
    sig_var = ((t0 - t0.mean(1, keepdims=True)) * (t1 - t1.mean(1, keepdims=True))).mean(1)
    noise_var = ((t0 - t1) ** 2).mean(1) / 2.0
    ok = (sig_var > 0) & (noise_var > 0)
    out = dict(inv_snr=np.nan, inv_snr_median=np.nan, snr_hetero=np.nan, inv_snr_rom=np.nan, mag_sd=np.nan, n=0)
    if not np.any(ok):
        return out
    out['inv_snr_rom'] = float(noise_var[ok].mean() / sig_var[ok].mean())
    out['mag_sd'] = float(np.log(np.sqrt(sig_var[ok])).std())
    isnr = noise_var[ok] / sig_var[ok]
    isnr = isnr[1.0 / (1.0 + isnr) < cc_max]
    if len(isnr):
        out.update(inv_snr=float(isnr.mean()), inv_snr_median=float(np.median(isnr)),
                   snr_hetero=float(0.5 * np.log(isnr).std()), n=int(len(isnr)))
    return out


def pooled_centred_r_half(resp):
    """
    Per-neuron split-half correlation of the first two trials pooled over all conditions after
    within-stimulus centring (each trial's mean over rotations removed per stimulus): the noise
    ceiling statistic of the centred CC_norm (utils.helper.pooled_scores with center=True).

    Parameters
    ----------
    resp : np.ndarray
        array of size (n_neurons, n_stim, n_rot, n_trials); trials 0 and 1 are used

    Returns
    -------
    r_half : np.ndarray
        array of size (n_neurons,); NaN for neurons without variance
    """
    n = resp.shape[0]
    w0 = (resp[..., 0] - resp[..., 0].mean(2, keepdims=True)).reshape(n, -1)
    w1 = (resp[..., 1] - resp[..., 1].mean(2, keepdims=True)).reshape(n, -1)
    w0 = w0 - w0.mean(1, keepdims=True)
    w1 = w1 - w1.mean(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (w0 * w1).sum(1) / np.sqrt((w0 ** 2).sum(1) * (w1 ** 2).sum(1))