import numpy as np
import pandas as pd
import pickle as pkl
import os
from sklearn.utils.extmath import randomized_svd
from scipy.spatial.distance import pdist
from concurrent.futures import ProcessPoolExecutor

from utils.helper import find_kn_unique_samples, find_eigendecomp, \
    calculate_cc_norm, calculate_norm_dists, calculate_rel_center_dist
from utils.neuron_selectivity import save_average_resp_n_cc
from utils.model_loader import model_train_test_responses, is_model_row
from utils.data_loader import mouse_all_train_test_responses, find_area_neurons


def _n_workers():
    """Worker count for the per-subsample process pools (N_WORKERS, default 4).
    A value <= 1 runs the subsamples sequentially in-process, which avoids the per-pool
    interpreter spawn cost on Windows (each pool re-imports the module chain)."""
    return int(os.environ.get('N_WORKERS', 4))


def equivariance_summary_all(model_df, params_df, project_root, subset=None):
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
    subset : str or None, default=None
        None: all 92 images with leave-two-out pairs (one scene + one texture). 'scenes' or
        'textures': only the 46 images of that class (the 45 training images of the class
        and its 80x test image), fitted with leave-one-image-out folds; caches and 'stim'
        (the held-out image index within the class) get the suffix '_scenes' / '_textures'

    Returns
    -------
    None
        saves results to CSV files in project_root / 'cache' / model_name:
        - variance analysis: {layer_name}_var_dim_{n_neurons}avg_{d}dim.csv, one row per
          (held-out pair 'stim', neuron subsample 'sample', dimension 'dim'); see
          equivariance_quantification_parallel. Summarize with helper.band_summary.
        Each cache is only computed when missing.
    """
    
    n_neurons = int(params_df.loc['n_neurons'].value)
    n_max = int(params_df.loc['n_max'].value)
    d = int(params_df.loc['n_procs'].value)
    noise_model = params_df.loc['noise_model'].value
    #non-gaussian noise models read/write suffixed caches (written by example_equiv_projs)
    #so they do not clobber the default gaussian caches
    noise_suffix = '' if noise_model == 'gaussian' else f'_{noise_model}'
    if subset not in (None, 'scenes', 'textures'):
        raise ValueError(f"subset must be None, 'scenes' or 'textures', got {subset!r}")
    out_suffix = noise_suffix + (f'_{subset}' if subset else '')

    print(f'Starting equivariance analysis...' + (f' ({subset} only)' if subset else ''))
    print(f'Parameters: n_neurons={n_neurons}, n_max={n_max}, d={d}')

    #loop over models & layers
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name
        print(f'Processing {model_name} {layer_name}')

        var_cache = cache_folder / f'{layer_name}_var_dim_{n_neurons}avg_{d}dim{out_suffix}.csv'
        if var_cache.exists():
            print(f'  {model_name} {layer_name}: cached, skipping')
            continue

        avg_train_f = cache_folder / f'{layer_name}_avg_resp_train{noise_suffix}.p'
        avg_test2x_f = cache_folder / f'{layer_name}_avg_resp_test2x{noise_suffix}.p'
        if (avg_train_f.exists()==False) | (avg_test2x_f.exists()==False):
            print(f'No average data found; run example_equiv_projs first')
            continue
        x_train_avg = pkl.load( open( avg_train_f, "rb" ) )
        x_test_avg = pkl.load( open( avg_test2x_f, "rb" ) )

        #combine train with test and initialize dataframes
        n_train = x_train_avg.shape[1]
        n_test = x_test_avg.shape[1]
        n_all = n_train + n_test
        x_all_avg = np.concatenate((x_train_avg, x_test_avg), axis=1)
        var_df = pd.DataFrame()

        #leave-two-stimuli-out pairs (one COCO and one texture image; the final pair is
        #the two 80x test images); 'stim' is the pair index. Images are ordered scenes
        #(indices 0 .. n_train/2 - 1), textures (n_train/2 .. n_train - 1), then the 80x
        #scene and texture. For a class subset, leave-one-image-out over that class.
        if subset is None:
            folds = [[a, b] for a, b in [(i, n_train // 2 + i) for i in range(n_train // 2)] + [(n_train, n_train + 1)]]
            keep = np.arange(n_all)
        else:
            keep = (np.concatenate((np.arange(0, n_train // 2), [n_train])) if subset == 'scenes'
                    else np.concatenate((np.arange(n_train // 2, n_train), [n_train + 1])))
            folds = [[j] for j in keep]
        for stim, held in enumerate(folds):
            mask = np.isin(np.arange(n_all), keep) & ~np.isin(np.arange(n_all), held)
            x_train_i = x_all_avg[:, mask]
            x_test_i = x_all_avg[:, held]

            tmp_df = equivariance_quantification_parallel(x_train_i, x_test_i, n_neurons, d, n_max=n_max)
            tmp_df['stim'] = stim
            var_df = pd.concat((var_df, tmp_df), axis=0)

        var_df.to_csv(var_cache, index=False)
        

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

        if is_model_row(model_name):
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

        #fit equivariance model to trial-averaged training data. Normalize each neuron by its
        #training-response std first, matching the quantification (compute_single_k_quantification);
        #without this the example projections are fit on a different (raw) footing than the
        #reported Fig 4b correlations and systematically understate the fit quality.
        x_train_k = x_train_trials_k.mean(-1)
        std_norm = x_train_k.std(axis=(1, 2), keepdims=True)
        idx_nonzero = std_norm.squeeze() > 0
        x_train_k = x_train_k[idx_nonzero] / std_norm[idx_nonzero]
        x_test_trials_k = x_test_trials_k[idx_nonzero] / std_norm[idx_nonzero][..., np.newaxis]
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
    return per_dim_quantities(x_train_avg_k, x_test_avg_k, proj, proj_shift, rot, rot_shift)


def per_dim_quantities(x_train_avg_k, x_test_avg_k, proj, proj_shift, rot, rot_shift, pc_train=True):
    """
    Per-dimension quantities of a fitted equivariance model (the body of
    compute_single_k_quantification, shared with the vs-dimension runner of
    equivariance_validation): eigenphases of rot in units of 2*pi/n_rot (ordered by
    magnitude, find_eigendecomp), the share of the total (std-normalized) training and test
    variance along each eigen-dimension, and the Pearson correlation between the projected
    responses and the chain predictions along each eigen-dimension (training and test images;
    the training correlation is skipped when pc_train is False, returning NaN).

    Parameters
    ----------
    x_train_avg_k, x_test_avg_k : np.ndarray
        std-normalized trial-averaged responses (n_neurons, n_stim, n_rot)
    proj, proj_shift, rot, rot_shift : fitted model (calculate_proj_rot_shift on x_train_avg_k)

    Returns
    -------
    evals_k, var_dim_train_k, var_dim_test_k, cc_train_k, cc_test_k : np.ndarray, each (d,)
    """
    d = rot.shape[0]
    n_rot = x_train_avg_k.shape[2]
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
    x_test_pred = calculate_predictions(x_test_avg_k, proj, proj_shift, rot, rot_shift)
    x_test_avg_proj = np.einsum('ji,jkl->ikl', proj@eigenvec, x_test_avg_k-proj_shift[:,np.newaxis,np.newaxis])
    x_test_pred_proj = np.einsum('ji,jkl->ikl', proj@eigenvec, x_test_pred-proj_shift[:,np.newaxis,np.newaxis])
    cc_test_k = calculate_cc_norm(x_test_avg_proj.reshape((d, 1, -1)),
                                x_test_pred_proj.reshape((d, 1, -1)),
                                np.ones((d,1)))
    if pc_train:
        x_train_pred = calculate_predictions(x_train_avg_k, proj, proj_shift, rot, rot_shift)
        x_train_avg_proj = np.einsum('ji,jkl->ikl', proj@eigenvec, x_train_avg_k-proj_shift[:,np.newaxis,np.newaxis])
        x_train_pred_proj = np.einsum('ji,jkl->ikl', proj@eigenvec, x_train_pred-proj_shift[:,np.newaxis,np.newaxis])
        cc_train_k = calculate_cc_norm(x_train_avg_proj.reshape((d, 1, -1)),
                                    x_train_pred_proj.reshape((d, 1, -1)),
                                    np.ones((d,1))).mean(1)
    else:
        cc_train_k = np.full(d, np.nan)

    return evals_k, var_dim_train_k, var_dim_test_k, cc_train_k, cc_test_k.mean(1)


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
        one row per (neuron subsample, dimension) with columns 'sample', 'dim', 'evals',
        'var train', 'var test', 'pc train', 'pc test'; dimensions are ordered by
        |eigenphase| within each subsample (find_eigendecomp). Results are deliberately NOT
        averaged across subsamples: averaging sorted spectra rank-wise smears a true
        eigenplane across rank slots whenever noise planes fall above it in some subsamples
        and below it in others. Band counts, band
        variance sums and band correlation means must be computed per subsample
        (helper.band_summary) and only then averaged.
    """
    #in case x_train.shape[0] is smaller than n_neurons
    n_neurons = min(n_neurons, x_train_avg.shape[0])

    #randomly sample neurons (disjoint subsamples)
    k_rand = np.min([np.max([x_train_avg.shape[0] // n_neurons, 1]), n_max])
    k_samp = find_kn_unique_samples(n_neurons, x_train_avg.shape[0], k_rand)

    #prepare arguments for parallel processing
    args_list = [(x_train_avg[k_samp[k]].copy(), x_test_avg[k_samp[k]].copy(), d) for k in range(k_rand)]

    #process k_rand samples in parallel (sequentially for a single sample or a single worker)
    n_workers = _n_workers()
    if (k_rand <= 1) or (n_workers <= 1):
        results = [compute_single_k_quantification(args) for args in args_list]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(compute_single_k_quantification, args_list))

    #one row per (subsample, dimension); no averaging across subsamples
    frames = []
    for k, (evals_k, var_dim_train_k, var_dim_test_k, cc_train_k, cc_test_k) in enumerate(results):
        frames.append(pd.DataFrame({'sample': k,
                                    'dim': np.arange(d),
                                    'evals': evals_k,
                                    'var train': var_dim_train_k,
                                    'var test': var_dim_test_k,
                                    'pc train': cc_train_k,
                                    'pc test': cc_test_k}))
    equivariance_df = pd.concat(frames, axis=0, ignore_index=True)

    return equivariance_df
