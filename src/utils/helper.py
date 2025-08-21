import numpy as np
import pickle as pkl
import pandas as pd
from scipy.linalg import logm
from scipy.stats import wilcoxon
from scipy.stats import zscore 


def calculate_cc_norm(x, x_pred, cc_max, tol=1e-6):
    """
    Calculate normalized correlation coefficient as defined in     
    https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2016.00010/full

    Parameters
    ----------
    x : np.ndarray
        trial-averaged neural responses of shape (n_neurons, n_stim, n_rot)
    x_pred : np.ndarray
        predicted neural responses of shape (n_neurons, n_stim, n_rot)
    cc_max : np.ndarray
        maximum correlation coefficient of shape (n_neurons, n_stim)
    tol : float, default=1e-6
        tolerance for standard deviation, below which the neuron gets a cc_norm of NaN

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


def compute_significance(results_df, results_f, alternative='greater'):
    """
    Compute one-sided pairwise Wilcoxon signed-rank test statistics and 
    p-values for the given results_df and save the results to a CSV file results_f.
    Compare putative lower to higher-order areas.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing the results with columns 'area' and 'score'.
    results_f : str
        path to save the results.
    alternative : str, default='greater'
        Defines the one-sidedness of the alternative hypothesis.

    Returns
    -------
    None
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
    stat_df.to_csv(results_f, index=False)


def find_eigendecomp(rot, n_rot, tol=1e-10):
    """
    Find the real eigenphases and eigenvectors of the rotation matrix rot.
    The eigenphases are give in units of 2*np.pi/n_rot and ordered by magnitude.
    The corresponding real eigenvectors come in pairs.

    Parameters
    ----------
    rot : np.ndarray
        orthogonal matrix of shape (d, d)
    n_rot : int
        number of rotations
    tol : float, default=1e-10
        tolerance for eigenvalues to be equal to 0, in which case the corresponding 
        complex eigenvectors need not be complex conjugates of each other

    Returns
    -------
    evals : np.ndarray
        ordered eigenphases of shape d, in units of 2*np.pi/n_rot
    evecs : np.ndarray
        ordered eigenvectors of shape (d, d), each column is an eigenvector
    """
    dp = rot.shape[0]
    unit_phase = 2*np.pi/n_rot

    #find the traceless, antisymmetric generator matrix GM corresponding to the orthogonal matrix rot
    #if det(rot) = 1 then GM = -1j*logm(rot)
    #if det(rot) = -1 then shift the diagonal elements by pi/dp such that tr(GM) = 0
    GM = -1j*logm(rot) + (np.linalg.det(rot)-1)/2 * np.pi/dp * np.identity(dp)
    #GM should be antisymmetric and in units of unit_phase
    Gs = (GM-GM.T)/2 / (unit_phase)

    #eigenvalues of Gs are the eigenphases of the corresponding rotation matrix
    ws, v = np.linalg.eig( Gs )
    #order by absolute value
    I = np.argsort(np.abs(ws))
    evals = np.real(ws[I])
    v = v[:,I]
    #order pairs such that negative first then positive
    for i in range( len(I)//2 ):
        if evals[2*i] > 0:
            evals[2*i], evals[2*i+1] = evals[2*i+1], evals[2*i]
            v[:,2*i], v[:,2*i+1] = v[:,2*i+1], v[:,2*i]

    #convert complex eigenvectors, which come in complex conjugate pairs, to real eigenvectors
    #if the eigenvalue is close to zero, we take the real part of the eigenvector
    #if the eigenvalue is not close to zero, we take the real and imaginary
    #part of the eigenvector and multiply by sqrt(2) to ensure normalization
    evecs = np.zeros(v.shape)
    for i in range(int(dp/2)):
        if (np.abs(evals)[2*i]<tol)|(np.abs(evals)[2*i+1]<tol):
            evecs[:,2*i] = np.real(v[:,2*i])/np.linalg.norm(np.real(v[:,2*i]))
            evecs[:,2*i+1] = np.real(v[:,2*i+1])/np.linalg.norm(np.real(v[:,2*i+1]))
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
    id_all = np.zeros((0,n_tot))
    rng = np.random.default_rng(seed=4)
    while id_all.shape[0]<n_perm:
        id_all = rng.permuted(np.tile(range(n_tot), n_perm).reshape(n_perm,n_tot), axis=1)
        id_all = np.unique(id_all, axis=0)

    return id_all


def find_kn_unique_samples(k, n, N):
    """
    Find N unique samples of k integers between 0 and n-1.

    Parameters
    ----------
    k : int
        number of integers to sample
    n : int
        number of integers to sample from
    N : int
        number of samples
    
    Returns
    -------
    id_all : np.ndarray
        array of shape (N, k)
    """
    rng = np.random.default_rng(seed=4)
    id_all = rng.permuted(np.tile(range(n), N).reshape(N,n), axis=1)[:,:k]

    return id_all