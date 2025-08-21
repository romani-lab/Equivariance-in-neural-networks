import numpy as np
from scipy.stats import zscore 
import pickle as pkl


def save_average_resp_n_cc(resp_train_trials, resp_test_trials, cache_folder, layer_name):
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
        
    Returns
    -------
    None
    """
    avg_train_f = cache_folder / f'{layer_name}_avg_resp_train.p'
    pkl.dump(resp_train_trials.mean(-1), open( avg_train_f, "wb" ) )

    avg_test_f = cache_folder / f'{layer_name}_avg_resp_test.p'
    pkl.dump(resp_test_trials.mean(-1), open( avg_test_f, "wb" ) )

    avg_test2x_f = cache_folder / f'{layer_name}_avg_resp_test2x.p'
    pkl.dump(resp_test_trials[:,:,:,:2].mean(-1), open( avg_test2x_f, "wb" ) )

    cc_train_f = cache_folder / f'{layer_name}_cc_half_train.csv'
    cc_half_train2x = corr_half(resp_train_trials).mean(-1)
    np.savetxt(cc_train_f, cc_half_train2x, delimiter=',')

    cc_test_f = cache_folder / f'{layer_name}_cc_half_test.csv'
    cc_half_test = corr_half(resp_test_trials).mean(-1)
    np.savetxt(cc_test_f, cc_half_test, delimiter=',')

    cc_test2x_f = cache_folder / f'{layer_name}_cc_half_test2x.csv'
    cc_half_test2x = corr_half(resp_test_trials[:,:,:,:2]).mean(-1)
    np.savetxt(cc_test2x_f, cc_half_test2x, delimiter=',')


def reliable_n_std_responses(resp_train_trials, resp_test_trials, xy_pos, cc_half_train_f, corr_thre, n_perm=100):
    """
    Find index for responsive neurons and normalize responses by std.

    Parameters
    ----------
    resp_train_trials : np.ndarray
        array of size (n_neurons, n_stim_train, n_rot, n_trials_train) containing responses
    resp_test_trials : np.ndarray
        array of size (n_neurons, n_stim_test, n_rot, n_trials_test) containing responses
    xy_pos : np.ndarray
        array of size (n_neurons, 2) containing xy position of neurons
    cc_half_train_f : str
        path to cache file containing half correlation
    corr_thre : float
        threshold for correlation
    n_perm : int, default=100
        number of random samples for computing corr_half

    Returns
    -------
    resp_train_trials : np.ndarray
        array of size (n_neurons, n_stim_train, n_rot, n_trials_train) containing normalized responses
    resp_test_trials : np.ndarray
        array of size (n_neurons, n_stim_test, n_rot, n_trials_test) containing normalized responses
    xy_pos : np.ndarray
        array of size (n_neurons, 2) containing xy position of neurons
    std_train : np.ndarray
        array of size n_neurons containing standard deviation of training responses
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

    #normalize responses by std
    resp_train_trials = resp_train_trials/std_train[:,np.newaxis,np.newaxis,np.newaxis]
    resp_test_trials = resp_test_trials/std_train[:,np.newaxis,np.newaxis,np.newaxis]

    return resp_train_trials, resp_test_trials, xy_pos, std_train, idx_resp


def corr_half(resp, n_perm=100, tol=1e-6):
    """
    Compute the correlation between two halves of the data 
    for n_perm choices of halves

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