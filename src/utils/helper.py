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
    stat_df = pd.DataFrame()

    #mouse areas
    if np.isin(['V1', 'LM', 'RL', 'AL'], results_df['area'].unique()).all():
        for test_area in ['LM','RL','AL']:
            stat, p = wilcoxon(results_df[results_df['area']=='V1']['score'], \
                            results_df[results_df['area']==test_area]['score'], \
                            alternative=alternative)
            tmp_df = pd.DataFrame({'area0': 'V1', 'area1': test_area, 'stat': stat, 'p': p}, index=[0])
            stat_df = pd.concat([stat_df, tmp_df], axis=0)

        stat, p = wilcoxon(results_df[results_df['area']=='LM']['score'], \
                        results_df[results_df['area']=='RL']['score'], \
                        alternative=alternative)
        tmp_df = pd.DataFrame({'area0': 'LM', 'area1': 'RL', 'stat': stat, 'p': p}, index=[0])
        stat_df = pd.concat([stat_df, tmp_df], axis=0)

        stat, p = wilcoxon(results_df[results_df['area']=='RL']['score'], \
                        results_df[results_df['area']=='AL']['score'], \
                        alternative=alternative)
        tmp_df = pd.DataFrame({'area0': 'RL', 'area1': 'AL', 'stat': stat, 'p': p}, index=[0])
        stat_df = pd.concat([stat_df, tmp_df], axis=0)

    #AlexNet areas
    if np.isin(['conv1', 'conv2', 'conv3', 'conv4', 'conv5'], results_df['area'].unique()).all():
        for test_area in ['conv2','conv3','conv4','conv5']:
            stat, p = wilcoxon(results_df[results_df['area']=='conv1']['score'], \
                            results_df[results_df['area']==test_area]['score'], \
                            alternative=alternative)
            tmp_df = pd.DataFrame({'area0': 'conv1', 'area1': test_area, 'stat': stat, 'p': p}, index=[0])
            stat_df = pd.concat([stat_df, tmp_df], axis=0)

        for i in range(3):
            stat, p = wilcoxon(results_df[results_df['area']==f'conv{i+2}']['score'], \
                            results_df[results_df['area']==f'conv{i+3}']['score'], \
                            alternative=alternative)
            tmp_df = pd.DataFrame({'area0': f'conv{i+2}', 'area1': f'conv{i+3}', 'stat': stat, 'p': p}, index=[0])
            stat_df = pd.concat([stat_df, tmp_df], axis=0)

    stat_df.reset_index(drop=True, inplace=True)

    return stat_df


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
