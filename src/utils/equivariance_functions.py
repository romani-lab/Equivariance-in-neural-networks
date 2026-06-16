import numpy as np
import pandas as pd
import pickle as pkl
import os
from scipy.stats import wilcoxon
from sklearn.utils.extmath import randomized_svd
from scipy.spatial.distance import pdist
from concurrent.futures import ProcessPoolExecutor

from utils.helper import find_kn_unique_samples, find_n_unique_perm, find_eigendecomp, \
    calculate_cc_norm, calculate_norm_dists, calculate_rel_center_dist
from utils.neuron_selectivity import save_average_resp_n_cc
from utils.model_loader import model_train_test_responses
from utils.data_loader import mouse_all_train_test_responses, find_area_neurons


def equivariance_summary_all(model_df, params_df, project_root):
    """
    Run equivariance analysis for all models and layers specified in model_df
    with parameters from params_df.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model specifications with columns 'network', 'layer', 'monitor'
    params_df : pd.DataFrame
        dataframe containing parameters including 'n_neurons', 'n_max', 'n_procs'
    project_root : pathlib.Path
        path to the project root directory

    Returns
    -------
    None
        saves results to CSV files in project_root / 'cache' / model_name:
        - variance analysis: {layer_name}_var_dim_{n_neurons}avg_{d}dim.csv
        - performance analysis: {layer_name}_equivariance_{n_neurons}avg_{d}dim.csv
    """
    
    n_neurons = int(params_df.loc['n_neurons'].value)
    n_max = int(params_df.loc['n_max'].value)
    d = int(params_df.loc['n_procs'].value)
    noise_model = params_df.loc['noise_model'].value
    #non-gaussian noise models read/write suffixed caches (written by example_equiv_projs)
    #so they do not clobber the default gaussian caches
    noise_suffix = '' if noise_model == 'gaussian' else f'_{noise_model}'

    print(f'Starting equivariance analysis...')
    print(f'Parameters: n_neurons={n_neurons}, n_max={n_max}, d={d}')

    #loop over models & layers
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name
        print(f'Processing {model_name} {layer_name}')

        # Check if all required outputs already exist in cache
        var_cache = cache_folder / f'{layer_name}_var_dim_{n_neurons}avg_{d}dim{noise_suffix}.csv'
        perf_cache = cache_folder / f'{layer_name}_equivariance_{n_neurons}avg_{d}dim{noise_suffix}.csv'
        if var_cache.exists() and perf_cache.exists():
            print(f'  {model_name} {layer_name}: all cached, skipping')
            continue

        avg_train_f = cache_folder / f'{layer_name}_avg_resp_train{noise_suffix}.p'
        avg_test2x_f = cache_folder / f'{layer_name}_avg_resp_test2x{noise_suffix}.p'
        cc_train_f = cache_folder / f'{layer_name}_cc_half_train{noise_suffix}.csv'
        cc_test2x_f = cache_folder / f'{layer_name}_cc_half_test2x{noise_suffix}.csv'
        if (avg_train_f.exists()==False) | (avg_test2x_f.exists()==False) | (cc_train_f.exists()==False) | (cc_test2x_f.exists()==False):
            print(f'No average data found; run example_equiv_projs first')
            continue
        x_train_avg = pkl.load( open( avg_train_f, "rb" ) )
        x_test_avg = pkl.load( open( avg_test2x_f, "rb" ) )
        cc_half_train = np.loadtxt(cc_train_f, delimiter=',')
        cc_half_test = np.loadtxt(cc_test2x_f, delimiter=',')

        #combine train with test and initialize dataframes
        n_train = x_train_avg.shape[1]
        n_test = x_test_avg.shape[1]
        n_all = n_train + n_test
        x_all_avg = np.concatenate((x_train_avg, x_test_avg), axis=1)
        cc_half_all = np.concatenate((cc_half_train, cc_half_test), axis=1)
        var_df = pd.DataFrame()
        performance_df = pd.DataFrame()

        #loop over pairs of images
        for i in range(n_train // 2):

            x_train_i = x_all_avg[:,(np.arange(n_all)!=i)&(np.arange(n_all)!=n_train//2+i)]
            x_test_i = x_all_avg[:,[i, n_train//2+i]]       
            cc_test_i = cc_half_all[:,[i, n_train//2+i]]

            tmp_df = equivariance_quantification_parallel(x_train_i, x_test_i, n_neurons, d, n_max=n_max)
            tmp_df['stim'] = i
            var_df = pd.concat((var_df, tmp_df), axis=0)

            tmp_df, _, _ = equivariance_performance_parallel(x_train_i, x_test_i,
                                                             cc_test_i, n_neurons, [d],
                                                             verbose=False, n_shuffle=1, n_max=n_max)
            tmp_df['stim'] = i
            performance_df = pd.concat((performance_df, tmp_df), axis=0)

        #final image pair
        x_train_i = x_all_avg[:,(np.arange(n_all)!=n_train)&(np.arange(n_all)!=n_train+1)]
        x_test_i = x_all_avg[:,[n_train, n_train+1]]       
        cc_test_i = cc_half_all[:,[n_train, n_train+1]]

        tmp_df = equivariance_quantification_parallel(x_train_i, x_test_i, n_neurons, d, n_max=n_max)
        tmp_df['stim'] = n_train // 2
        var_df = pd.concat((var_df, tmp_df), axis=0)

        tmp_df, _, _ = equivariance_performance_parallel(x_train_i, x_test_i,
                                                         cc_test_i, n_neurons, [d],
                                                         verbose=False, n_shuffle=1, n_max=n_max)
        tmp_df['stim'] = n_train // 2
        performance_df = pd.concat((performance_df, tmp_df), axis=0)

        var_df.to_csv(cache_folder / f'{layer_name}_var_dim_{n_neurons}avg_{d}dim{noise_suffix}.csv', index=False)
        performance_df.to_csv(cache_folder / f'{layer_name}_equivariance_{n_neurons}avg_{d}dim{noise_suffix}.csv', index=False)
        

def example_equiv_projs(model_df, params_df, project_root):
    """
    Generate example projections and eigenvalues for all models and layers specified in model_df
    with parameters from params_df.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model specifications with columns 'network', 'layer', 'monitor'
    params_df : pd.DataFrame
        dataframe containing parameters including 'stimulus', 'corr_thre', 'n_neurons', 
        'inv_snr', 'n_procs'
    project_root : pathlib.Path
        path to the project root directory

    Returns
    -------
    None
        saves multiple files in project_root / 'cache' / model_name:
        - absolute eigenvalues: {layer_name}_example_evals_abs_{n_neurons}rand_{d}dim.p
        - projection weights: {layer_name}_example_proj_weights_{n_neurons}rand_{d}dim.csv
        - data projections: {layer_name}_example_projs_data_{n_neurons}rand_{d}dim.p
        - model projections: {layer_name}_example_projs_model_{n_neurons}rand_{d}dim.p
    """
    #get parameters
    stim_name = params_df.loc['stimulus'].value
    corr_thre = float(params_df.loc['corr_thre'].value)
    n_neurons = int(params_df.loc['n_neurons'].value)
    inv_snr = float(params_df.loc['inv_snr'].value)
    d = int(params_df.loc['n_procs'].value)
    noise_model = params_df.loc['noise_model'].value
    #non-gaussian noise models get a filename suffix so they do not clobber the
    #default gaussian caches
    noise_suffix = '' if noise_model == 'gaussian' else f'_{noise_model}'

    #define folders
    data_root = project_root / 'data'
    stim_root = data_root / 'stimuli'
    retinotopy_root = data_root / 'retinotopy'

    #loop over models & layers
    for i0 in range(model_df.shape[0]):

        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        monitor = model_df['monitor'].values[i0]
        cache_folder = project_root / f'cache' / model_name
        cache_folder.mkdir(parents=True, exist_ok=True)
        print(f'Processing {model_name} {layer_name}')

        # Check if all required outputs already exist in cache
        evals_cache = cache_folder / f'{layer_name}_example_evals_abs_{n_neurons}rand_{d}dim{noise_suffix}.p'
        projs_data_cache = cache_folder / f'{layer_name}_example_projs_data_{n_neurons}rand_{d}dim{noise_suffix}.p'
        projs_model_cache = cache_folder / f'{layer_name}_example_projs_model_{n_neurons}rand_{d}dim{noise_suffix}.p'
        if evals_cache.exists() and projs_data_cache.exists() and projs_model_cache.exists():
            print(f'  {model_name} {layer_name}: all cached, skipping')
            continue

        if np.isin(model_name,['pixels','alexnet','random_alexnet']) or model_name.startswith('alt_alexnet'):
            resp_train_trials, resp_test_trials, _, idx_resp = model_train_test_responses(model_name, layer_name, \
                                                                                    stim_root, stim_name, \
                                                                                    monitor, corr_thre, cache_folder, \
                                                                                    inv_snr=inv_snr, noise_model=noise_model)
            resp_train_trials = resp_train_trials[idx_resp]
            resp_test_trials = resp_test_trials[idx_resp]
        else:
            #to speed up: for first layer of each network, load all responsive neurons
            if layer_name==model_df[model_df['network']==model_name]['layer'].values[0]:
                date = model_name[-10:]
                resp_train_trials0, resp_test_trials0, _, idx_resp = \
                    mouse_all_train_test_responses(data_root / date, model_name, \
                                                    stim_root, stim_name, \
                                                    monitor, corr_thre, cache_folder)
                resp_train_trials0 = resp_train_trials0[idx_resp]
                resp_test_trials0 = resp_test_trials0[idx_resp]
                retinotopy_fn = retinotopy_root / f'{model_name}_{stim_name}.npz'
            
            #for later layers, just subselect neurons in that layer
            idx_area = find_area_neurons(retinotopy_fn, layer_name, idx_resp)
            resp_train_trials = resp_train_trials0[idx_area]
            resp_test_trials = resp_test_trials0[idx_area]

        #save average responses and trial-to-trial variability for later use
        save_average_resp_n_cc(resp_train_trials, resp_test_trials, cache_folder, layer_name, noise_suffix=noise_suffix)

        #randomly subsample of n_neurons neurons
        n_resp, n_train, n_rot, _ = resp_train_trials.shape
        _, n_test, _, n_trials_test = resp_test_trials.shape
        n_neurons0 = min(n_neurons, resp_train_trials.shape[0])
        k_rand = np.min([np.max([n_resp // n_neurons0, 1]), 100])
        if model_name=='alexnet' and layer_name=='conv5':
            #conv5 has only ~303 responsive units and we draw 250, so k_rand==1: there is
            #only one disjoint sample. Use a different draw than the default seed-4 path
            #(same scheme as find_kn_unique_samples, seed 7) for a different random subselection.
            k_samp = np.random.default_rng(seed=7).permutation(n_resp)[:n_neurons0]
        elif k_rand>3:
            k_samp = find_kn_unique_samples(n_neurons0, n_resp, k_rand)[3]
        else:
            k_samp = find_kn_unique_samples(n_neurons0, n_resp, k_rand)[0]
        x_train_trials_k = resp_train_trials[k_samp]
        x_test_trials_k = resp_test_trials[k_samp]

        #fit equivariance model to trial-averaged training data
        x_train_k = x_train_trials_k.mean(-1)
        proj, center, rot, rot_shift = calculate_proj_rot_shift(x_train_k, d)
        evals, eigenvec = find_eigendecomp(rot)        
        evals = evals / (2*np.pi/n_rot) #units of 2*pi/n_rot

        #save absolute eigenvalues
        tmp_df = pd.DataFrame({'x': np.arange(d)+1, 'y': np.abs(evals)})
        pkl.dump(tmp_df, open( cache_folder / f'{layer_name}_example_evals_abs_{n_neurons}rand_{d}dim{noise_suffix}.p', "wb" ) )

        #project testing data
        proj_df = pd.DataFrame()
        for j in range(d//2):
            for l in range(n_test):
                xval = np.einsum('j,jkl->kl', proj@eigenvec[:,2*j], x_test_trials_k[:,l]-center[:,np.newaxis,np.newaxis])
                yval = np.einsum('j,jkl->kl', proj@eigenvec[:,2*j+1], x_test_trials_k[:,l]-center[:,np.newaxis,np.newaxis])
                tmp_df = pd.DataFrame({'x': xval.mean(-1),
                                       'y': yval.mean(-1),
                                       'errx': np.sqrt((xval.std(-1)**2).sum()/n_trials_test),
                                       'erry': np.sqrt((yval.std(-1)**2).sum()/n_trials_test)})
                tmp_df['test'] = l
                tmp_df['eigenspace_index'] = 2*j+1
                tmp_df['eval'] = evals[2*j]
                proj_df = pd.concat([proj_df, tmp_df], axis=0, ignore_index=True)
        pkl.dump(proj_df, open( cache_folder / f'{layer_name}_example_projs_data_{n_neurons}rand_{d}dim{noise_suffix}.p', "wb" ) )

        #project equivariance model predictions
        x_test_k = x_test_trials_k.mean(-1)
        x_test_pred_k = calculate_predictions(x_test_k, proj, center, rot, rot_shift)
        proj_df = pd.DataFrame()
        for j in range(d//2):
            for l in range(n_test):
                xval = np.einsum('j,jk->k', proj@eigenvec[:,2*j], x_test_pred_k[:,l]-center[:,np.newaxis])
                yval = np.einsum('j,jk->k', proj@eigenvec[:,2*j+1], x_test_pred_k[:,l]-center[:,np.newaxis])
                tmp_df = pd.DataFrame({'x': xval, 'y': yval})
                tmp_df['errx'] = 0
                tmp_df['erry'] = 0
                tmp_df['test'] = l
                tmp_df['eigenspace_index'] = 2*j+1
                tmp_df['eval'] = evals[2*j]
                proj_df = pd.concat([proj_df, tmp_df], axis=0, ignore_index=True)
        pkl.dump(proj_df, open( cache_folder / f'{layer_name}_example_projs_model_{n_neurons}rand_{d}dim{noise_suffix}.p', "wb" ) )


def calculate_predictions(x, proj, proj_shift, rot, rot_shift):
    """
    Calculate predicted neuronal responses using fitted equivariance model.

    Parameters
    ----------
    x : np.ndarray
        trial-averaged neuronal responses of shape (n_neurons, n_stim, n_rot)
    proj : np.ndarray
        projection matrix of shape (n_neurons, d)
    proj_shift : np.ndarray
        projection shift vector of shape (n_neurons,)
    rot : np.ndarray
        rotation matrix of shape (d, d)
    rot_shift : np.ndarray
        rotation shift vector of shape (d,)

    Returns
    -------
    x_pred : np.ndarray
        predicted neuronal responses of shape (n_neurons, n_stim, n_rot)
    """
    _, n_stim, n_rot = x.shape
    d = proj.shape[1]

    x_proj = np.einsum('ji,jkl->ikl', proj, x-proj_shift[:,np.newaxis,np.newaxis])
    x_pred = np.zeros((d, n_stim, n_rot+1))
    x_pred[:,:,0] = x_proj[:,:,0]
    for j in range(n_rot):
        x_pred[:,:,j+1] = rot@x_pred[:,:,j]-rot_shift[:,np.newaxis]

    #replace initial response (x_proj[:,:,0]) with prediction for n_rot rotations
    x_pred = np.roll(x_pred[:,:,1:], 1, axis=2)

    #project back out
    x_pred = np.einsum('ij,jkl->ikl', proj, x_pred)+proj_shift[:,np.newaxis,np.newaxis]

    return x_pred


def calculate_proj_rot_shift(x, d):
    """
    Calculate projection matrix, rotation matrix, and shift vectors for equivariance model
    using orthogonal Procrustes problem.

    Parameters
    ----------
    x : np.ndarray
        trial-averaged neuronal responses of shape (n_neurons, n_stim, n_rot)
    d : int
        number of projected dimensions

    Returns
    -------
    proj : np.ndarray
        projection matrix of shape (n_neurons, d)
    proj_shift : np.ndarray
        projection shift vector of shape (n_neurons,)
    rot : np.ndarray
        rotation matrix of shape (d, d)
    rot_shift : np.ndarray
        rotation shift vector of shape (d,)
    """     
    n_neurons = x.shape[0]
    proj_shift = x.reshape((n_neurons,-1)).mean(-1)

    #compute A, B matrices for orthogonal Procustes problem
    A = (x-proj_shift[:,np.newaxis,np.newaxis]).reshape((n_neurons,-1))
    B = np.roll(x-proj_shift[:,np.newaxis,np.newaxis], -1, axis=2).reshape((n_neurons,-1))

    #estimate projection matrix proj
    M = B @ A.T
    proj, _, _ = randomized_svd(M, n_components=d, random_state=0)
    #project with proj to d-dimensions then estimate rotation matrix and the shift
    A_d = proj.T @ A
    B_d = proj.T @ B
    M_d = B_d @ A_d.T
    U_d, _, VT_d = randomized_svd(M_d, n_components=d, random_state=0)
    rot = U_d @ VT_d
    rot_shift = ( rot @ A_d - B_d ).mean(1)

    #enforce proper rotation
    if np.linalg.det(rot) < 0:        
        U_d[:,-1] = -U_d[:,-1]
        rot = U_d @ VT_d
        rot_shift = ( rot @ A_d - B_d ).mean(1)
        
    return proj, proj_shift, rot, rot_shift


def compute_single_k_quantification(args):
    """
    Helper function to compute equivariance quantification for a single sample of neurons.

    Parameters
    ----------
    args : tuple
        tuple containing (x_train_avg_k, x_test_avg, d)

    Returns
    -------
    evals_k : np.ndarray
        eigenvalues of shape (d,)
    var_dim_train_k : np.ndarray
        variance explained on training images of shape (d,)
    var_dim_test_k : np.ndarray
        variance explained on test images of shape (d,)
    cc_train_k : np.ndarray
        Pearson correlation on training images of shape (d,)
    cc_test_k : np.ndarray
        Pearson correlation on test images of shape (d,)
    """
    x_train_avg_k, x_test_avg_k, d = args
    n_rot = x_train_avg_k.shape[2]

    #normalize by std of training responses
    std_norm = x_train_avg_k.std(axis=(1,2), keepdims=True)
    idx_nonzero = std_norm.squeeze() > 0
    x_train_avg_k = x_train_avg_k[idx_nonzero] / std_norm[idx_nonzero]
    x_test_avg_k = x_test_avg_k[idx_nonzero] / std_norm[idx_nonzero]
    
    #compute quantities for this sample
    proj, proj_shift, rot, rot_shift = calculate_proj_rot_shift(x_train_avg_k, d)
    evals_k, eigenvec = find_eigendecomp(rot)
    #units of 2*pi/n_rot
    evals_k = evals_k / (2*np.pi/n_rot)

    #variance explained
    s_diag_train = (eigenvec.T@proj.T@x_train_avg_k.reshape((x_train_avg_k.shape[0],-1))).std(1)
    var_tot_train = (x_train_avg_k.reshape((x_train_avg_k.shape[0],-1)).std(1)**2).sum()
    var_dim_train_k = (s_diag_train**2) / var_tot_train
    s_diag_test = (eigenvec.T@proj.T@x_test_avg_k.reshape((x_test_avg_k.shape[0],-1))).std(1)
    var_tot_test = (x_test_avg_k.reshape((x_test_avg_k.shape[0],-1)).std(1)**2).sum()
    var_dim_test_k = (s_diag_test**2) / var_tot_test

    #Pearson correlation across rotations and stimuli (need to reshape)
    x_train_pred = calculate_predictions(x_train_avg_k, proj, proj_shift, rot, rot_shift)
    x_test_pred = calculate_predictions(x_test_avg_k, proj, proj_shift, rot, rot_shift)
    x_train_avg_proj = np.einsum('ji,jkl->ikl', proj@eigenvec, x_train_avg_k-proj_shift[:,np.newaxis,np.newaxis])
    x_train_pred_proj = np.einsum('ji,jkl->ikl', proj@eigenvec, x_train_pred-proj_shift[:,np.newaxis,np.newaxis])
    x_test_avg_proj = np.einsum('ji,jkl->ikl', proj@eigenvec, x_test_avg_k-proj_shift[:,np.newaxis,np.newaxis])
    x_test_pred_proj = np.einsum('ji,jkl->ikl', proj@eigenvec, x_test_pred-proj_shift[:,np.newaxis,np.newaxis])
    cc_train_k = calculate_cc_norm(x_train_avg_proj.reshape((d, 1, -1)), 
                                x_train_pred_proj.reshape((d, 1, -1)), 
                                np.ones((d,1)))
    cc_test_k = calculate_cc_norm(x_test_avg_proj.reshape((d, 1, -1)), 
                                x_test_pred_proj.reshape((d, 1, -1)), 
                                np.ones((d,1)))
    
    return evals_k, var_dim_train_k, var_dim_test_k, cc_train_k.mean(1), cc_test_k.mean(1)


def equivariance_quantification_parallel(x_train_avg, x_test_avg, n_neurons, d, n_max=1):
    """
    Quantify equivariance using eigenvalues, variance explained, and correlations.

    Parameters
    ----------
    x_train_avg : np.ndarray
        trial-averaged training responses of shape (n_neurons_tot, n_stim_train, n_rot)
    x_test_avg : np.ndarray
        trial-averaged testing responses of shape (n_neurons_tot, n_stim_test, n_rot)
    n_neurons : int
        number of neurons to sample
    d : int
        dimension of projected subspace
    n_max : int, default=1
        maximum number of random neuron samples

    Returns
    -------
    equivariance_df : pd.DataFrame
        dataframe with columns 'evals', 'var train', 'var test', 'pc train', 'pc test'
    """
    #in case x_train.shape[0] is smaller than n_neurons
    n_neurons = min(n_neurons, x_train_avg.shape[0])
    
    #randomly sample neurons
    k_rand = np.min([np.max([x_train_avg.shape[0] // n_neurons, 1]), n_max])
    k_samp = find_kn_unique_samples(n_neurons, x_train_avg.shape[0], k_rand)

    #prepare arguments for parallel processing
    args_list = [(x_train_avg[k_samp[k]].copy(), x_test_avg[k_samp[k]].copy(), d) for k in range(k_rand)]
    
    #process k_rand samples in parallel
    if k_rand <= 1:
        #fallback to sequential processing for single sample
        results = [compute_single_k_quantification(args_list[0])]
    else:
        n_workers = int(os.environ.get('LSB_MAX_NUM_PROCESSORS', 4))
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(compute_single_k_quantification, args_list))
    
    #average aggregated results
    evals = np.zeros(d)
    var_dim_train = np.zeros(d)
    var_dim_test = np.zeros(d)
    cc_train = np.zeros(d)
    cc_test = np.zeros(d)
    for evals_k, var_dim_train_k, var_dim_test_k, cc_train_k, cc_test_k in results:
        evals += evals_k / k_rand
        var_dim_train += var_dim_train_k / k_rand
        var_dim_test += var_dim_test_k / k_rand
        cc_train += cc_train_k / k_rand
        cc_test += cc_test_k / k_rand

    equivariance_df = pd.DataFrame({
                           'evals': evals,
                           'var train': var_dim_train,
                           'var test': var_dim_test,
                           'pc train': cc_train,
                           'pc test': cc_test
                           })

    return equivariance_df


def compute_single_k_performance(args):
    """
    Helper function to compute equivariance performance for a single sample of neurons.

    Parameters
    ----------
    args : tuple
        tuple containing (d, x_train_k, x_test_k, cc_max_test)

    Returns
    -------
    n_nonan : int
        number of non-NaN neurons
    cc_all : np.ndarray
        correlation coefficients for performance evaluation
    """
    d, x_train_k, x_test_k, cc_max_test_k = args

    #normalize by std of training responses
    std_norm = x_train_k.std(axis=(1,2), keepdims=True)
    idx_nonzero = std_norm.squeeze() > 0
    x_train_k = x_train_k[idx_nonzero] / std_norm[idx_nonzero]
    x_test_k = x_test_k[idx_nonzero] / std_norm[idx_nonzero]
    cc_max_test_k = cc_max_test_k[idx_nonzero]
    
    #fit equivariance model to training data
    proj, proj_shift, rot, rot_shift = calculate_proj_rot_shift(x_train_k, d)
    x_test_k_pred = calculate_predictions(x_test_k, proj, proj_shift, rot, rot_shift)

    #compute performance quantities
    cc_all = calculate_cc_norm(x_test_k.reshape((x_test_k.shape[0], 1, -1)), 
                               x_test_k_pred.reshape((x_test_k_pred.shape[0], 1, -1)), 
                               cc_max_test_k.mean(1).reshape((-1,1)))
    idx_nan = np.where(np.isnan(cc_all.mean(1)))[0]
    cc_all[idx_nan] = 0
    n_nonan = x_train_k.shape[0]-len(idx_nan)
        
    return n_nonan, cc_all


def equivariance_performance_parallel(x_train_avg, x_test_avg, cc_half_test, n_neurons, d_range, \
                                      n_shuffle=100, n_max=1, verbose=False):
    """
    Evaluate equivariance model performance and test against shuffled fit.

    Parameters
    ----------
    x_train_avg : np.ndarray
        trial-averaged training responses of shape (n_neurons_tot, n_stim_train, n_rot)
    x_test_avg : np.ndarray
        trial-averaged testing responses of shape (n_neurons_tot, n_stim_test, n_rot)
    cc_half_test : np.ndarray
        half correlation for test stimuli of shape (n_neurons_tot, n_stim_test)
    n_neurons : int
        number of neurons to sample
    d_range : list
        list of dimensions to evaluate
    n_shuffle : int, default=100
        number of shuffle permutations for statistical testing
    n_max : int, default=1
        maximum number of random neuron samples
    verbose : bool, default=False
        whether to print performance metrics

    Returns
    -------
    err_df : pd.DataFrame
        performance results on actual data with columns 'dim', 'n_nonan', 'cc_mean'
    err_df_perm : pd.DataFrame
        performance results on shuffled data with columns 'dim', 'n_nonan', 'cc_mean'
    p_vals_mean : np.ndarray
        mean p-values across random samples, of shape (len(d_range),)
    """
    #in case x_train.shape[0] is smaller than n_neurons
    n_neurons = min(n_neurons, x_train_avg.shape[0])
    _, n_test, n_rot = x_test_avg.shape

    #max correlation coefficient from half correlation
    cc_max_test = np.sqrt(2*cc_half_test/(1+cc_half_test))
    
    #randomly sample neurons
    k_rand = np.min([np.max([x_train_avg.shape[0] // n_neurons, 1]), n_max])
    k_samp = find_kn_unique_samples(n_neurons, x_train_avg.shape[0], k_rand)
    
    #main computation
    err_df = pd.DataFrame()
    err_df_perm = pd.DataFrame()
    id_perm_train = find_n_unique_perm(n_rot*x_train_avg.shape[1], n_shuffle)
    id_perm_test = find_n_unique_perm(n_rot*n_test, n_shuffle)
    p_vals = np.zeros((len(d_range), k_rand))
    #loop over dimensions
    for j in range(len(d_range)):
        cc_all = np.zeros(k_rand, dtype=object)
        cc_all_perm = np.zeros(k_rand, dtype=object)
        #main computation
        for i in range(n_shuffle+1):
            x_train_k = x_train_avg
            x_test_k = x_test_avg
            if i>0:
                x_train_k = ( x_train_k.reshape((x_train_avg.shape[0], -1))[:,id_perm_train[i-1]] ).reshape(x_train_avg.shape)
                x_test_k = ( x_test_k.reshape((x_train_avg.shape[0], -1))[:,id_perm_test[i-1]] ).reshape(x_test_avg.shape)
            
            #prepare arguments for parallel processing
            args_list = [(d_range[j], x_train_k[k_samp[k]].copy(), x_test_k[k_samp[k]].copy(), cc_max_test[k_samp[k]].copy()) 
                         for k in range(k_rand)]
            
            #process k_rand samples in parallel
            if k_rand <= 1:
                #fallback to sequential processing for single sample
                results = [compute_single_k_performance(args_list[0])]
            else:
                n_workers = int(os.environ.get('LSB_MAX_NUM_PROCESSORS', 4))
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(compute_single_k_performance, args_list))
            
            #collect and average results across parallel processes
            if i==0:
                errors_arr = np.zeros((1,3))
                for k in range(k_rand):
                    n_nonan, cc_all_k = results[k]
                    errors_arr += [d_range[j], n_nonan, np.nanmean(cc_all_k[:,0])] / k_rand
                    cc_all[k] = cc_all_k
            elif i==1:
                errors_arr = np.zeros((1,3))
                for k in range(k_rand):
                    n_nonan, cc_all_k = results[k]
                    errors_arr += [d_range[j], n_nonan, np.nanmean(cc_all_k[:,0])] / k_rand / n_shuffle
                    cc_all_perm[k] = cc_all_k / n_shuffle
            else:
                for k in range(k_rand):
                    n_nonan, cc_all_k = results[k]
                    errors_arr += [d_range[j], n_nonan, np.nanmean(cc_all_k[:,0])] / k_rand / n_shuffle
                    cc_all_perm[k] += cc_all_k / n_shuffle
            #add results to a DataFrame
            if i==0:
                tmp_df = pd.DataFrame(errors_arr, columns=['dim', 'n_nonan', 'cc_mean'])
                tmp_df['dim'] = np.round(tmp_df['dim']).astype(int)
                err_df = pd.concat([err_df, tmp_df], axis=0)
                # err01_df = pd.concat([err01_df, tmp01_df], axis=0)
                if verbose:
                    print(f'Dimension: {d_range[j]}')
                    print(f"Frac. of no-nan neurons: {tmp_df['n_nonan'].mean()/n_neurons:.2f} (data)")
                    print(f"Mean CCnorm test: {tmp_df['cc_mean'].mean():.2f} (data)")
            elif i==n_shuffle:
                tmp_df = pd.DataFrame(errors_arr, columns=['dim', 'n_nonan', 'cc_mean'])
                tmp_df['dim'] = np.round(tmp_df['dim']).astype(int)
                err_df_perm = pd.concat([err_df_perm, tmp_df], axis=0)
                for k in range(k_rand):
                    p_vals[j,k] = wilcoxon(cc_all[k][:,0], cc_all_perm[k][:,0], alternative='greater')[1]
                if verbose:
                    print(f"Frac. of no-nan neurons: {tmp_df['n_nonan'].mean()/n_neurons:.2f} (shuffle)")
                    print(f"Mean CCnorm test: {tmp_df['cc_mean'].mean():.2f} (shuffle)")
                    print(f"pval: {np.nanmean(p_vals[j]):.2f}")

    return err_df, err_df_perm, np.nanmean(p_vals, axis=1)


def equivariance_ccnorm_vs_dim_all(model_df, params_df, project_root, d_range=None):
    """
    Compute the equivariance model's predictive CCnorm as a function of the fitted
    subspace dimension d, for all models and layers specified in model_df.

    For each model/layer the subspace dimension is swept over d_range. At every
    dimension the equivariance model is fit on a leave-two-stimuli-out training set
    and the normalized correlation (CCnorm) is evaluated on the two held-out
    stimuli, exactly as equivariance_summary_all does at the single dimension d
    (so the d == n_procs point reproduces the cached
    {layer_name}_equivariance_{n_neurons}avg_{d}dim.csv result).

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model specifications with columns 'network', 'layer', 'monitor'
    params_df : pd.DataFrame
        dataframe containing parameters including 'n_neurons', 'n_max', 'n_procs'
    project_root : pathlib.Path
        path to the project root directory
    d_range : array-like, optional
        subspace dimensions to evaluate. Defaults to np.arange(2, 61, 2), i.e.
        even dimensions up to 60, as used for the preprint.

    Returns
    -------
    None
        saves results to CSV files in project_root / 'cache' / model_name:
        {layer_name}_ccnorm_vs_dim_{n_neurons}avg_{d}dim.csv with columns
        'dim', 'cc_mean', 'stim'.
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    n_max = int(params_df.loc['n_max'].value)
    d = int(params_df.loc['n_procs'].value)
    if d_range is None:
        #sweep even dimensions up to 60 (as in the preprint), independent of n_procs
        d_range = np.arange(2, 61, 2)
    d_range = [int(d0) for d0 in d_range]

    print(f'Starting CCnorm-vs-dimension analysis...')
    print(f'Parameters: n_neurons={n_neurons}, n_max={n_max}, d_range={d_range}')

    #loop over models & layers
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name
        print(f'Processing {model_name} {layer_name}')

        out_f = cache_folder / f'{layer_name}_ccnorm_vs_dim_{n_neurons}avg_{d}dim.csv'
        if out_f.exists():
            print(f'  {model_name} {layer_name}: cached, skipping')
            continue

        avg_train_f = cache_folder / f'{layer_name}_avg_resp_train.p'
        avg_test2x_f = cache_folder / f'{layer_name}_avg_resp_test2x.p'
        cc_train_f = cache_folder / f'{layer_name}_cc_half_train.csv'
        cc_test2x_f = cache_folder / f'{layer_name}_cc_half_test2x.csv'
        if (avg_train_f.exists()==False) | (avg_test2x_f.exists()==False) | (cc_train_f.exists()==False) | (cc_test2x_f.exists()==False):
            print(f'No average data found; run example_equiv_projs first')
            continue
        x_train_avg = pkl.load( open( avg_train_f, "rb" ) )
        x_test_avg = pkl.load( open( avg_test2x_f, "rb" ) )
        cc_half_train = np.loadtxt(cc_train_f, delimiter=',')
        cc_half_test = np.loadtxt(cc_test2x_f, delimiter=',')

        #combine train with test
        n_train = x_train_avg.shape[1]
        n_test = x_test_avg.shape[1]
        n_all = n_train + n_test
        x_all_avg = np.concatenate((x_train_avg, x_test_avg), axis=1)
        cc_half_all = np.concatenate((cc_half_train, cc_half_test), axis=1)

        #neuron subsampling (same scheme as equivariance_performance_parallel)
        n_neurons_l = min(n_neurons, x_all_avg.shape[0])
        k_rand = int(np.min([np.max([x_all_avg.shape[0] // n_neurons_l, 1]), n_max]))

        #leave-two-stimuli-out pairs, matching equivariance_summary_all
        pair_idx = [(i, n_train // 2 + i) for i in range(n_train // 2)] + [(n_train, n_train + 1)]

        out_df = pd.DataFrame()
        for stim, (a, b) in enumerate(pair_idx):
            mask = (np.arange(n_all) != a) & (np.arange(n_all) != b)
            x_train_i = x_all_avg[:, mask]
            x_test_i = x_all_avg[:, [a, b]]
            cc_test_i = cc_half_all[:, [a, b]]
            cc_max_test = np.sqrt(2 * cc_test_i / (1 + cc_test_i))
            k_samp = find_kn_unique_samples(n_neurons_l, x_all_avg.shape[0], k_rand)

            #sweep subspace dimension; cc averaged over neuron subsamples (no shuffle)
            for d0 in d_range:
                cc_d = 0.0
                for k in range(k_rand):
                    _, cc_all_k = compute_single_k_performance(
                        (d0, x_train_i[k_samp[k]].copy(), x_test_i[k_samp[k]].copy(),
                         cc_max_test[k_samp[k]].copy()))
                    cc_d += np.nanmean(cc_all_k[:, 0]) / k_rand
                out_df = pd.concat([out_df,
                                    pd.DataFrame({'dim': [d0], 'cc_mean': [cc_d], 'stim': [stim]})],
                                   axis=0)

        out_df.to_csv(out_f, index=False)
        print(f'  saved {out_f.name}')
