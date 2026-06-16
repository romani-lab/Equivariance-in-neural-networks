import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.svm import LinearSVC
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from sklearn.utils.extmath import randomized_svd

from utils.helper import find_kn_unique_samples, find_eigendecomp, calculate_effect_size, \
    calculate_norm_dists, calculate_rel_center_dist
from utils.model_loader import model_train_test_responses
from utils.data_loader import mouse_all_train_test_responses, find_area_neurons
from utils.equivariance_functions import calculate_proj_rot_shift, calculate_predictions
from utils.stimulus_loader import find_n_imgs


def decode_all(model_df, params_df, project_root, c_range=[0.001], ref_index=0, equivariant=False, model=False, \
               orientation_decoding=True, class_decoding=True, verbose=False):
    """
    Run orientation and texture decoding analysis for all models and layers specified in model_df
    with the parameters specified in params_df.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model specifications with columns 'network', 'layer', 'monitor'
    params_df : pd.DataFrame
        dataframe containing parameters including
        'stimulus', 'corr_thre', 'inv_snr', 'n_neurons', 'n_max'
    project_root : pathlib.Path
        path to the project root directory
    c_range : list, default=[0.001]
        list of regularization parameters for linear SVM classifier
    ref_index : int, default=0
        reference orientation index for alignment
    equivariant : bool, default=False
        whether to use equivariant subspace projection (requires 'n_procs' in params_df)
        suffix = '_{n_procs}dim' will be added to output filenames if True
    model : bool, default=False
        whether to run decoder on projected model predictions or projected recorded responses
    orientation_decoding : bool, default=True
        whether to run orientation decoding analysis
    class_decoding : bool, default=True
        whether to run texture decoding analysis
    verbose : bool, default=False
        whether to print progress messages

    Returns
    -------
    None
        saves results to CSV files in project_root / 'cache' / model_name:
        - orientation decoding: {layer_name}_ori2x_decoder_{ref_index}ref_{n_neurons}avg{suffix}.csv
        - texture decoding: {layer_name}_class2x_decoder_{n_neurons}avg_{c_range[0]}C{suffix}.csv
    """
    #get parameters
    stim_name = params_df.loc['stimulus'].value
    corr_thre = float(params_df.loc['corr_thre'].value)
    inv_snr = float(params_df.loc['inv_snr'].value)
    n_neurons = int(params_df.loc['n_neurons'].value)
    n_max = int(params_df.loc['n_max'].value)
    noise_model = params_df.loc['noise_model'].value
    #non-gaussian noise models get a filename suffix so they do not clobber the
    #default gaussian caches
    noise_suffix = '' if noise_model == 'gaussian' else f'_{noise_model}'
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
        if model:
            suffix += '_model'
    else:
        d = None
        suffix = ''
    suffix += noise_suffix

    print(f'Starting decoding analysis...')
    print(f'Parameters: n_neurons={n_neurons}, n_max={n_max}')

    #define folders
    data_root = project_root / 'data'
    stim_root = data_root / 'stimuli'
    retinotopy_root = data_root / 'retinotopy'

    #loop over models & layers
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        monitor = model_df['monitor'].values[i0]
        cache_folder = project_root / 'cache' / model_name
        cache_folder.mkdir(parents=True, exist_ok=True)

        # Check if all required outputs already exist in cache
        ori_cache = cache_folder / f'{layer_name}_ori2x_decoder_{ref_index}ref_{n_neurons}avg{suffix}.csv'
        if len(c_range)==1:
            class_cache = cache_folder / f'{layer_name}_class2x_decoder_{n_neurons}avg_{c_range[0]}C{suffix}.csv'
        else:
            class_cache = cache_folder / f'{layer_name}_class2x_decoder_{n_neurons}avg_Crange{suffix}.csv'

        ori_needed = orientation_decoding and not ori_cache.exists()
        class_needed = class_decoding and not class_cache.exists()

        if not ori_needed and not class_needed:
            print(f'  {model_name} {layer_name}: all cached, skipping')
            continue

        print(f'Processing {model_name} {layer_name}')

        if np.isin(model_name,['pixels','alexnet','random_alexnet']) or model_name.startswith('alt_alexnet'):
            resp_train_trials, resp_test_trials, _, idx_resp = model_train_test_responses(model_name, layer_name, \
                                                                                    stim_root, stim_name, \
                                                                                    monitor, corr_thre, cache_folder, \
                                                                                    inv_snr=inv_snr, noise_model=noise_model)
            #subselect reliably responsive neurons
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
                #subselect reliably responsive neurons
                resp_train_trials0 = resp_train_trials0[idx_resp]
                resp_test_trials0 = resp_test_trials0[idx_resp]
                retinotopy_fn = retinotopy_root / f'{model_name}_{stim_name}.npz'

            #for later layers, just subselect neurons in that layer
            idx_area = find_area_neurons(retinotopy_fn, layer_name, idx_resp)
            resp_train_trials = resp_train_trials0[idx_area]
            resp_test_trials = resp_test_trials0[idx_area]

        #concatenate training and test responses
        #for test, select the first 2 out of 80 trials, to match the number of training trials
        resp_all_trials = np.concatenate((resp_train_trials, resp_test_trials[:,:,:,:2]), axis=1)

        if ori_needed:
            result_df = orientation2x_2d_decoder(resp_all_trials.mean(-1), n_neurons, d=d, n_max=n_max, model=model, ref_index=ref_index, verbose=verbose)
            result_df.to_csv(ori_cache, index=False)

        if class_needed:
            svm_df = texture2x_svm_decoder_parallel(resp_all_trials, n_neurons, c_range, d=d, n_max=n_max, model=model, verbose=verbose)
            svm_df.to_csv(class_cache, index=False)


def decoder_effect_sizes(model_df, params_df, project_root, C0=0.001, C1=1):
    """
    Compute and save effect sizes for SVM and orientation decoder
    for all models and layers specified in model_df with the parameters specified in params_df.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    C0 : float, default=0.001
        regularization parameter for linear SVM classifier on full space
    C1 : float, default=1
        regularization parameter for linear SVM classifier on subspace

        
    Returns
    -------
    None
        saves effect size results to CSV files in project_root / 'results_extended':
        - SVM decoding effect sizes: svm_effect_sizes_{n_procs}dim.csv
        - orientation decoding effect sizes: ori_effect_sizes_{n_procs}dim.csv
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    d = int(params_df.loc['n_procs'].value)
    results_folder = project_root / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)

    #loop over svm and ori decoders
    for i in range(2):
        if i==0:
            decoder_f0 = f'class2x_decoder_{n_neurons}avg_{C0}C'
            decoder_f1 = f'class2x_decoder_{n_neurons}avg_{C1}C'
        else:
            decoder_f0 = f'ori2x_decoder_0ref_{n_neurons}avg'
            decoder_f1 = f'ori2x_decoder_0ref_{n_neurons}avg'
            
        #collect results across layers
        results_df = pd.DataFrame()
        for i0 in range(model_df.shape[0]):
            model_name = model_df['network'].values[i0]
            layer_name = model_df['layer'].values[i0]
            cache_folder = project_root / f'cache' / model_name

            class_f = cache_folder / f'{layer_name}_{decoder_f0}.csv'
            tmp_df = pd.read_csv(class_f)
            tmp_df['model'] = model_name
            tmp_df['area'] = layer_name
            class_f2 = cache_folder / f'{layer_name}_{decoder_f1}_{d}dim.csv'
            tmp_df2 = pd.read_csv(class_f2)
            tmp_df2['model'] = model_name
            tmp_df2['area'] = layer_name

            if i==0:
                tmp_df['score'] = 1-tmp_df['acc test']
                tmp_df2['score'] = 1-tmp_df2['acc test']
                df = tmp_df.merge(tmp_df2, on=["class", "ori"], suffixes=("_1", "_2"))
            else:
                tmp_df['score'] = tmp_df['res']
                tmp_df2['score'] = tmp_df2['res']
                df = tmp_df.merge(tmp_df2, on=["class", "ori"], suffixes=("_1", "_2"))

            result_df = calculate_effect_size(df)
            result_df['model'] = model_name
            result_df['area'] = layer_name
            results_df = pd.concat([results_df, result_df], axis=0)

        results_df.to_csv(results_folder / f'{decoder_f0.split("_")[0]}_effect_sizes_{d}dim.csv', index=False)


def distance_all(model_df, params_df, project_root):
    """
    Compute and save normalized distances and relative center distances
    for all models and layers specified in model_df with the parameters specified in params_df.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model specifications with columns 'network', 'layer', 'monitor'
    params_df : pd.DataFrame
        dataframe containing parameters including
        'stimulus', 'corr_thre', 'inv_snr', 'n_neurons', 'n_max'
    project_root : pathlib.Path
        path to the project root directory

    Returns
    -------
    None
        saves results to CSV files in project_root / 'cache' / model_name / {layer_name}_distances_avg{suffix}.csv 
        for suffix in ['', f'_{d}dim_inv', f'_{d}dim_tun']
    """
    #get parameters
    n_max = int(params_df.loc['n_max'].value)
    d = int(params_df.loc['n_procs'].value)

    #loop over models & layers
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / 'cache' / model_name

        # Check if all required outputs already exist in cache
        suffixes = ['', f'_{d}dim_inv', f'_{d}dim_tun']
        all_cached = all((cache_folder / f'{layer_name}_distances_avg{s}.csv').exists() for s in suffixes)
        if all_cached:
            print(f'  {model_name} {layer_name}: all cached, skipping')
            continue

        avg_train_f = cache_folder / f'{layer_name}_avg_resp_train.p'
        avg_test_f = cache_folder / f'{layer_name}_avg_resp_test.p'
        x_train_avg = pkl.load( open( avg_train_f, "rb" ) )
        x_test_avg = pkl.load( open( avg_test_f, "rb" ) )
        n_neurons0, n_train, n_rot = x_train_avg.shape
        n_test = x_test_avg.shape[1]

        #normalize by std of training responses
        std = x_train_avg.std(axis=(1,2), keepdims=True)
        x_train_avg = x_train_avg / std
        x_test_avg = x_test_avg / std

        k_rand = np.min([np.max([x_train_avg.shape[0] // n_neurons0, 1]), n_max])
        k_samp = find_kn_unique_samples(n_neurons0, x_train_avg.shape[0], k_rand)

        #loop through full space, invariant subspace, tuned subspace
        for j0 in range(3):
            if j0==0:
                suffix = ''
            elif j0==1:
                suffix = f'_{d}dim_inv'
            else:
                suffix = f'_{d}dim_tun'

            dists_all = np.zeros((n_test+2, 2))
            rel_center = 0.0
            for k in range(k_rand):
                x_train_k = x_train_avg[k_samp[k]].copy()
                x_test_k = x_test_avg[k_samp[k]].copy()

                if j0>0:
                    proj_k, _, rot_k, _ = calculate_proj_rot_shift(x_train_k, d)
                    eigenval_k, eigenvec_k = find_eigendecomp(rot_k)
                    eigenval_k = eigenval_k / (2*np.pi/n_rot) #units of 2*pi/n_rot
                    if j0==1:
                        id0 = np.where( (np.abs(eigenval_k)<.5) )[0]
                        eigenvec1 = eigenvec_k[:,id0]
                    elif j0==2:
                        id1 = np.where( (np.abs(eigenval_k)>=.5)&(np.abs(eigenval_k)<1.5) )[0]
                        eigenvec1 = eigenvec_k[:,id1]
                    eigenproj_k = proj_k@eigenvec1
                    x_train_k = np.einsum('ji,jkl->ikl', eigenproj_k, x_train_k)
                    x_test_k = np.einsum('ji,jkl->ikl', eigenproj_k, x_test_k)

                dists_all[0] += calculate_norm_dists(x_train_k[:,:n_train//2]) / k_rand
                dists_all[1] += calculate_norm_dists(x_train_k[:,n_train//2:]) / k_rand
                for i in range(n_test):
                    dists_all[i+2] += calculate_norm_dists(x_test_k[:,i:i+1]) / k_rand

                rel_center += calculate_rel_center_dist(x_test_k) / k_rand
            results_df = pd.DataFrame(dists_all, columns=['norm_dists_mean', 'norm_dists_sem'], \
                                    index=['train1','train2']+[f'test{i+1}' for i in range(n_test)])
            results_df['rel_center_test'] = rel_center
            results_df.to_csv(cache_folder / f'{layer_name}_distances_avg{suffix}.csv')


def example_decoder_weights(model_df, params_df, project_root, C0=0.001, equivariant=False):
    """
    Compute and save example orientation PC 1-2 and SVM weights (img 91-92) for all models 
    and layers specified in model_df with the parameters specified in params_df.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model specifications with columns 'network', 'layer', 'monitor'
    params_df : pd.DataFrame
        dataframe containing parameters including
        'stimulus', 'corr_thre', 'inv_snr', 'n_neurons', 'n_max'
    project_root : pathlib.Path
        path to the project root directory
    C0 : float, default=0.001
        regularization parameter for linear SVM classifier
    equivariant : bool, default=False
        whether to use equivariant decoding weights
        
    Returns
    -------
    None
        saves weights to CSV files in project_root / 'cache' / model_name:
        - orientation weights: {layer_name}_ori12_weights_{n_neurons}rand{suffix}.csv
        - SVM weights: {layer_name}_svm12_weights_{n_neurons}rand{suffix}.csv
    """
    #get parameters
    stim_name = params_df.loc['stimulus'].value
    corr_thre = float(params_df.loc['corr_thre'].value)
    inv_snr = float(params_df.loc['inv_snr'].value)
    n_neurons = int(params_df.loc['n_neurons'].value)
    noise_model = params_df.loc['noise_model'].value
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
    else:
        d = None
        suffix = ''

    print(f'Starting example decoding analysis...')
    print(f'with one random choice of n_neurons={n_neurons}')

    #define folders
    data_root = project_root / 'data'
    stim_root = data_root / 'stimuli'
    retinotopy_root = data_root / 'retinotopy'

    #loop over models & layers
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        monitor = model_df['monitor'].values[i0]
        cache_folder = project_root / 'cache' / model_name
        cache_folder.mkdir(parents=True, exist_ok=True)

        # Check if all required outputs already exist in cache
        ori_cache = cache_folder / f'{layer_name}_ori12_weights_{n_neurons}rand{suffix}.csv'
        svm_cache = cache_folder / f'{layer_name}_svm12_weights_{n_neurons}rand{suffix}.csv'
        if ori_cache.exists() and svm_cache.exists():
            print(f'  {model_name} {layer_name}: all cached, skipping')
            continue

        print(f'Processing {model_name} {layer_name}')

        if np.isin(model_name,['pixels','alexnet','random_alexnet']):
            resp_train_trials, resp_test_trials, _, idx_resp = model_train_test_responses(model_name, layer_name, \
                                                                                    stim_root, stim_name, \
                                                                                    monitor, corr_thre, cache_folder, \
                                                                                    inv_snr=inv_snr, noise_model=noise_model)
            #subselect reliably responsive neurons
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
                #subselect reliably responsive neurons
                resp_train_trials0 = resp_train_trials0[idx_resp]
                resp_test_trials0 = resp_test_trials0[idx_resp]
                retinotopy_fn = retinotopy_root / f'{model_name}_{stim_name}.npz'

            #for later layers, just subselect neurons in that layer
            idx_area = find_area_neurons(retinotopy_fn, layer_name, idx_resp)
            resp_train_trials = resp_train_trials0[idx_area]
            resp_test_trials = resp_test_trials0[idx_area]

        #subselect number of neurons
        n_resp, _, n_rot, n_trials_train = resp_train_trials.shape
        n_neurons0 = min(n_neurons, resp_train_trials.shape[0])
        k_rand = np.min([np.max([n_resp // n_neurons0, 1]), 100])
        if k_rand>3:
            k_samp = find_kn_unique_samples(n_neurons0, n_resp, k_rand)[3]
        else:
            k_samp = find_kn_unique_samples(n_neurons0, n_resp, k_rand)[0]
        x_train_trials_k = resp_train_trials[k_samp]
        x_test_trials_k = resp_test_trials[k_samp]

        #find orientation PCs
        x_train_k = x_train_trials_k.mean(-1)
        std_norm = x_train_k.std(axis=(1,2), keepdims=True)
        idx_nonzero = std_norm.squeeze() > 0
        x_train_k = x_train_k[idx_nonzero] / std_norm[idx_nonzero]
        if equivariant:
            proj_k, proj_shift_k, rot_k, rot_shift_k = calculate_proj_rot_shift(x_train_k, d)
            eigenval_k, eigenvec_k = find_eigendecomp(rot_k)
            eigenval_k = eigenval_k / (2*np.pi/n_rot) #units of 2*pi/n_rot
            id1 = np.where( (np.abs(eigenval_k)>=.5)&(np.abs(eigenval_k)<1.5) )[0]
            eigenvec1 = eigenvec_k[:,id1]
            eigenproj_k = proj_k@eigenvec1
            x_train_k_pred = calculate_predictions(x_train_k, proj_k, proj_shift_k, rot_k, rot_shift_k)
            x_train_k = np.einsum('ij,ikl->jkl', eigenproj_k, x_train_k_pred-proj_shift_k[:,np.newaxis,np.newaxis])
        x_train_shift = x_train_k.copy()
        for i in range(x_train_k.shape[1]):
            x_train_shift[:,i,:] -= x_train_k[:,i,:].mean(axis=1)[:,np.newaxis] #center[:,np.newaxis] 
        x_train_shift = x_train_shift.reshape((x_train_k.shape[0], -1))
        w_ori, _, _ = randomized_svd(x_train_shift, n_components=2, random_state=0)
        # if equivariant:
        #     w_ori = eigenproj_k @ w_ori
        w_df = pd.DataFrame(w_ori, columns=['PC1','PC2'])
        if equivariant:
            w_df['proc_id'] = range(len(id1))
        w_df.to_csv(cache_folder / f'{layer_name}_ori12_weights_{n_neurons}rand{suffix}.csv', index=False)

        #find SVM weights
        x_all_trials = np.concatenate((x_train_trials_k, x_test_trials_k[:,:,:,:2]), axis=1)
        c_train = np.array(range(x_all_trials.shape[1])) 
        n_rot_train = n_rot - 1
        y_train = np.squeeze(np.repeat(c_train, n_rot_train*n_trials_train))
        x_train_k = x_all_trials[:,:,:n_rot_train,:]
        std_norm = x_train_k.std(axis=(1,2,3), keepdims=True)
        id_nonzero = std_norm.squeeze() > 0
        x_train_k = x_train_k[id_nonzero] / std_norm[id_nonzero]
        if equivariant:
            x_train_k_avg = x_train_k.mean(-1)
            proj_k, _, rot_k, _ = calculate_proj_rot_shift(x_train_k_avg, d)
            eigenval_k, eigenvec_k = find_eigendecomp(rot_k)
            eigenval_k = eigenval_k / (2*np.pi/n_rot) #units of 2*pi/n_rot
            id0 = np.where(np.abs(eigenval_k)<0.5)[0]
            eigenvec0 = eigenvec_k[:,id0]
            eigenproj_k = proj_k@eigenvec0
            x_train_k = np.einsum('ij,iklm->jklm', eigenproj_k, x_train_k)
        x_train_k = x_train_k.reshape((x_train_k.shape[0], -1))
        svm = LinearSVC(dual=False, C=C0, class_weight=None).fit(x_train_k.T, y_train)
        #normalize each class weight vector (column) to unit norm across neurons, so the
        #per-neuron weights stay small and centered (axis=0), as in the preprint. Using
        #axis=1 instead normalizes each neuron across the 2 classes and forces every point
        #onto the unit circle.
        w_svm = svm.coef_[-2:].T/np.linalg.norm(svm.coef_[-2:].T, axis=0, keepdims=True)
        # if equivariant:
        #     w_svm = eigenproj_k @ w_svm
        w_df = pd.DataFrame(w_svm, columns=['img91','img92'])
        if equivariant:
            w_df['proc_id'] = range(len(id0))
        w_df.to_csv(cache_folder / f'{layer_name}_svm12_weights_{n_neurons}rand{suffix}.csv', index=False)


def align_by_rotating(x):
    """
    Align the first response to the x-axis for each stimulus and pair of dimensions.

    Parameters
    ----------
    x : np.ndarray
        neuronal responses of shape (dim, n_stim, n_rot)
    
    Returns
    -------
    x_rot : np.ndarray
        aligned neuronal responses of shape (dim, n_stim, n_rot)
    """
    dim, n_stim, _ = x.shape
    x_rot = x.copy()
    for d in range(dim//2):
        for i in range(n_stim):
            x0 = x[2*d,i,0]
            y0 = x[2*d+1,i,0]
            r0_sq = x0**2+y0**2
            x_rot[2*d,i,:] = (x0*x[2*d,i,:]+y0*x[2*d+1,i,:])/r0_sq
            x_rot[2*d+1,i,:] = (-y0*x[2*d,i,:]+x0*x[2*d+1,i,:])/r0_sq

    return x_rot


def proj12_n_norm(x_train, x_test):
    """
    Project neuronal responses onto 2D plane using PCA and normalize to unit norm.

    Parameters
    ----------
    x_train : np.ndarray
        training neuronal responses of shape (n_neurons, n_train, n_rot)
    x_test : np.ndarray
        test neuronal responses of shape (n_neurons, n_test, n_rot)

    Returns
    -------
    x_test_proj : np.ndarray
        normalized 2D projected responses of shape (2, n_stim, n_rot)
    """
    n_neurons, _, n_rot = x_test.shape
    center = x_train.reshape((n_neurons, -1)).mean(axis=1)

    x_train_shift = x_train.copy()
    for i in range(x_train.shape[1]):
        x_train_shift[:,i,:] -= x_train[:,i,:].mean(axis=1)[:,np.newaxis] #center[:,np.newaxis] 
    x_train_shift = x_train_shift.reshape((n_neurons, -1))
    U_train, _, _ = randomized_svd(x_train_shift, n_components=2, random_state=0)

    x_test_proj = np.einsum('ij,ikl->jkl', U_train, x_test - center[:,np.newaxis,np.newaxis])
    x_test_proj = align_by_rotating(x_test_proj)
    if x_test_proj[1,:,1].mean()<0:
        x_test_proj[1] = -x_test_proj[1]

    x_test_proj = x_test_proj / np.linalg.norm(x_test_proj, axis=0) #, keepdims=True

    return x_test_proj


def orientation2x_2d_decoder(x_train, n_neurons, rot_step=24, ref_index=0, 
                             d=None, n_max=1, verbose=False, model=False):
    """
    Decode relative orientation from neuronal responses using 2D projection 
    with leave-one-out cross-validation.

    Parameters
    ----------
    x_train : np.ndarray
        neuronal responses of shape (n_neurons, n_train, n_rot)
    n_neurons : int
        number of neurons to randomly sample
    rot_step : int, default=24
        angular step in degrees
    ref_index : int, default=0
        reference orientation index for alignment
    d : int, default=None
        dimension for equivariant subspace projection
    n_max : int, default=1
        maximum number of random neuron samples
    verbose : bool, default=False
        whether to print performance metrics
    model : bool, default=False
        whether to run decoder on projected model predictions or projected recorded responses

    Returns
    -------
    results_df : pd.DataFrame
        dataframe with columns 'R2', 'res', 'class', 'ori'
    """    
    _, n_train, n_rot = x_train.shape

    #independent variables for decoding
    angles = np.linspace(rot_step,(n_rot-1)*rot_step,n_rot-1)
    n_angles = len(angles)
    y_test = np.vstack([np.cos(angles*np.pi/180),np.sin(angles*np.pi/180)]).T

    #randomly sample neurons
    n_neurons = min(n_neurons, x_train.shape[0])
    k_rand = np.min([np.max([x_train.shape[0] // n_neurons, 1]), n_max])
    k_samp = find_kn_unique_samples(n_neurons, x_train.shape[0], k_rand)

    results_df = pd.DataFrame()
    r2_test = np.zeros(n_train*n_angles)
    angle_err_test = np.zeros(n_train*n_angles)
    ct = 0
    for k in range(k_rand):
        x_train_k_all = x_train[k_samp[k]].copy()
        #shift reference orientation
        x_train_k_all = np.roll(x_train_k_all, -ref_index, axis=2)
        for i in range(n_train):
            train_mask = np.arange(n_train)!=i
            #normalize by std of training responses
            std_norm = x_train_k_all[:,train_mask].std(axis=(1,2), keepdims=True)
            idx_nonzero = std_norm.squeeze() > 0
            x_train_k_norm = x_train_k_all[idx_nonzero] / std_norm[idx_nonzero]

            #separate training and test sets
            x_train_k_i = x_train_k_norm[:,train_mask]
            x_test_k_i = x_train_k_norm[:,i:(i+1)]

            #if d is None, no projection
            if d is None:
                x_train_k_proj = x_train_k_i
                x_test_k_proj = x_test_k_i
            #if d is not None, project to tuned subspace
            else:
                proj_k, proj_shift_k, rot_k, rot_k_shift = calculate_proj_rot_shift(x_train_k_i, d)
                eigenval_k, eigenvec_k = find_eigendecomp(rot_k)
                eigenval_k = eigenval_k / (2*np.pi/n_rot) #units of 2*pi/n_rot
                id1 = np.where( (np.abs(eigenval_k)>=.5)&(np.abs(eigenval_k)<1.5) )[0]
                if len(id1)<2:
                    ct += 1
                    continue
                eigenvec1 = eigenvec_k[:,id1]
                eigenproj_k = proj_k@eigenvec1
                if model:
                    x_train_k_i_pred = calculate_predictions(x_train_k_i, proj_k, proj_shift_k, rot_k, rot_k_shift)
                    x_test_k_i_pred = calculate_predictions(x_test_k_i, proj_k, proj_shift_k, rot_k, rot_k_shift)
                    x_train_k_proj = np.einsum('ij,ikl->jkl', eigenproj_k, x_train_k_i_pred-proj_shift_k[:,np.newaxis,np.newaxis])
                    x_test_k_proj = np.einsum('ij,ikl->jkl', eigenproj_k, x_test_k_i_pred-proj_shift_k[:,np.newaxis,np.newaxis])
                else:
                    x_train_k_proj = np.einsum('ij,ikl->jkl', eigenproj_k, x_train_k_i-proj_shift_k[:,np.newaxis,np.newaxis])
                    x_test_k_proj = np.einsum('ij,ikl->jkl', eigenproj_k, x_test_k_i-proj_shift_k[:,np.newaxis,np.newaxis])

            #dependent variables for decoding
            x_test_rot = proj12_n_norm(x_train_k_proj, x_test_k_proj)
            #remove reference response used for alignment
            x_test_rot = x_test_rot[:,:,1:].reshape((2, -1))

            #test R2 score
            u_test = ((y_test-x_test_rot.T)**2).sum()
            v_test = ((y_test-y_test.mean())**2).sum()
            r2_test[i*n_angles:(i+1)*n_angles] += (1-u_test/v_test) / k_rand

            #test angle errors
            angle_test = np.arctan2(y_test[:,1],y_test[:,0])
            angle_pred = np.arctan2(x_test_rot.T[:,1],x_test_rot.T[:,0])  
            angle_diff = np.abs(np.mod(angle_test-angle_pred+np.pi,2*np.pi)-np.pi)*180/np.pi
            angle_err_test[i*n_angles:(i+1)*n_angles] += angle_diff / k_rand

    results_df = pd.DataFrame({
        'R2': r2_test, 
        'res': angle_err_test, 
        'class': np.repeat(np.arange(n_train), n_angles),
        'ori': np.tile(np.arange(n_rot)[1:], n_train)
    })

    if verbose:
        if d is not None:
            print(f'Skipped {ct} samples because no tuned dimensions.')
        print(f"R2 test: {results_df['R2'].mean():.2f}")
        print(f"res test: {results_df['res'].mean():.2f}")

    return results_df


def svm2x_train_test(c, x_train_k, n_rot_test, d=None, model=False):
    """
    Train linear SVM on neuronal responses for all orientations 
    except for the last n_rot_test orientations.

    Parameters
    ----------
    c : float
        SVM regularization parameter
    x_train_k : np.ndarray
        neuronal responses of shape (n_neurons, n_train, n_rot, n_trials)
    n_rot_test : int
        number of test rotations to hold out
    d : int, default=None
        dimension for equivariant subspace projection
    model : bool, default=False
        whether to run decoder on projected model predictions or projected recorded responses

    Returns
    -------
    acc_train : float
        training accuracy
    acc_test : np.ndarray
        test accuracy for each stimulus class of shape (n_train,)
    conf_test : np.ndarray
        SVM decision confidence for each test prediction of shape (n_train,)
    """
    #get dimensions
    _, n_train, n_rot, n_trials_train = x_train_k.shape
    n_rot_train = n_rot - n_rot_test
    
    #class labels and independent variables for decoding
    c_train = np.array(range(n_train)) 
    y_train = np.squeeze(np.repeat(c_train, n_rot_train*n_trials_train))
    y_test = np.squeeze(np.repeat(c_train, n_rot_test))

    #normalize by std of training responses
    std_norm = x_train_k[:,:,:n_rot_train,:].std(axis=(1,2,3), keepdims=True)
    idx_nonzero = std_norm.squeeze() > 0
    x_train_k = x_train_k[idx_nonzero] / std_norm[idx_nonzero]
    n_neurons = x_train_k.shape[0]

    #separate training and test sets
    x_train_k_i = x_train_k[:,:,:n_rot_train]
    x_test_k_i = x_train_k[:,:,n_rot_train:].mean(-1)
    
    #if d is not None, project to invariant subspace
    if d is None:
        x_train_k_proj = x_train_k_i.reshape((n_neurons,-1))
        x_test_k_proj = x_test_k_i.reshape((n_neurons,-1))
    else:
        x_train_k_avg = x_train_k_i.mean(-1)
        proj_k, proj_k_shift, rot_k, rot_k_shift = calculate_proj_rot_shift(x_train_k_avg, d)
        eigenval_k, eigenvec_k = find_eigendecomp(rot_k)
        eigenval_k = eigenval_k / (2*np.pi/n_rot) #units of 2*pi/n_rot
        id0 = np.where(np.abs(eigenval_k)<0.5)[0]
        d0 = len(id0)
        eigenproj_k = proj_k@eigenvec_k[:,id0]
        if model:
            x_train_k_avg_pred = calculate_predictions(x_train_k_avg, proj_k, proj_k_shift, rot_k, rot_k_shift)
            x_train_k_avg_pred = np.repeat(x_train_k_avg_pred, n_trials_train).reshape((n_neurons, n_train, n_rot_train, n_trials_train))
            x_train_k_proj = np.einsum('ij,iklm->jklm', eigenproj_k, x_train_k_avg_pred).reshape((d0,-1))
            x_test_k_avg_pred = calculate_predictions(x_test_k_i, proj_k, proj_k_shift, rot_k, rot_k_shift)
            x_test_k_proj = np.einsum('ij,ikl->jkl', eigenproj_k, x_test_k_avg_pred).reshape((d0,-1))
        else:
            x_train_k_proj = np.einsum('ij,iklm->jklm', eigenproj_k, x_train_k_i).reshape((d0,-1))
            x_test_k_proj = np.einsum('ij,ikl->jkl', eigenproj_k, x_test_k_i).reshape((d0,-1))

    #train and test SVM
    svm = LinearSVC(dual=False, C=c, class_weight=None).fit(x_train_k_proj.T, y_train)
    acc_train = svm.score(x_train_k_proj.T, y_train)
    y_pred = svm.predict(x_test_k_proj.T)
    acc_test = (y_pred == y_test)
    confidence = svm.decision_function(x_test_k_proj.T)
    conf_test = np.diag(confidence)

    return acc_train, acc_test, conf_test


def compute_single_k_svm(args):
    """
    Helper function to compute SVM decoding for a single sample of neurons 
    with leave-one-out cross-validation across orientations.

    Parameters
    ----------
    args : tuple
        tuple containing (x_train, c, d, model)

    Returns
    -------
    acc_train_k : np.ndarray
        training accuracies of shape (n_rot, n_train)
    acc_test_k : np.ndarray  
        test accuracies of shape (n_rot, n_train)
    conf_test_k : np.ndarray
        test confidences of shape (n_rot, n_train)
    """
    x_train_k, c, d, model = args
    _, n_train, n_rot, _ = x_train_k.shape
    # print(f"Worker started with shape: {x_train_k.shape}")
    
    acc_train_k = np.zeros((n_rot, n_train))
    acc_test_k = np.zeros((n_rot, n_train))
    conf_test_k = np.zeros((n_rot, n_train))
    for r in range(n_rot):
        x_train_k_r = np.roll(x_train_k, shift=-r, axis=2)
        acc_train_res, acc_test_res, conf_test_res = svm2x_train_test(c, x_train_k_r, 1, d=d, model=model)
        acc_train_k[r] = acc_train_res
        acc_test_k[r] = acc_test_res
        conf_test_k[r] = conf_test_res
    
    return acc_train_k, acc_test_k, conf_test_k


def texture2x_svm_decoder_parallel(x_train, n_neurons, c_range, 
                                   d=None, model=False, n_max=1, verbose=False):
    """
    Decode image identity from neuronal responses using linear SVM 
    with leave-one-out cross-validation.

    Parameters
    ----------
    x_train : np.ndarray
        neuronal responses of shape (n_neurons, n_stim, n_rot, n_trials)
    n_neurons : int
        number of neurons to randomly sample
    c_range : list
        list of SVM regularization parameters  
    d : int, default=None
        dimension for equivariant subspace projection
    model : bool, default=False
        whether to run decoder on projected model predictions or projected recorded responses
    n_max : int, default=1
        maximum number of random neuron samples
    verbose : bool, default=False
        whether to print performance metrics

    Returns
    -------
    svm_df : pd.DataFrame
        dataframe with columns 'acc train', 'acc test', 'conf test', 'C', 'class', 'ori'
    """
    #get dimensions
    _, n_train, n_rot, _ = x_train.shape

    #randomly sample neurons
    n_neurons = min(n_neurons, x_train.shape[0])
    k_rand = np.min([np.max([x_train.shape[0] // n_neurons, 1]), n_max])
    k_samp = find_kn_unique_samples(n_neurons, x_train.shape[0], k_rand)
    
    #main computation
    svm_df = pd.DataFrame()
    #loop over regularization parameters
    for c in c_range:
        #prepare arguments for parallel processing
        args_list = [(x_train[k_samp[k]].copy(), c, d, model) for k in range(k_rand)]
        
        #process k_rand samples in parallel
        if k_rand <= 1:
            #fallback to sequential processing for single sample
            results = [compute_single_k_svm(args_list[0])]
        else:
            ctx = get_context("spawn")
            n_workers = int(os.environ.get('LSB_MAX_NUM_PROCESSORS', 4))
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
                results = list(executor.map(compute_single_k_svm, args_list))
        
        #aggregate mean results (across k_rand samples)
        acc_train = np.zeros((n_rot, n_train))
        acc_test = np.zeros((n_rot, n_train))
        conf_test = np.zeros((n_rot, n_train))
        for acc_train_k, acc_test_k, conf_test_k in results:
            acc_train += acc_train_k / k_rand
            acc_test += acc_test_k / k_rand
            conf_test += conf_test_k / k_rand
            
        #add results to a DataFrame
        tmp_df = pd.DataFrame({
            'acc train': np.squeeze(acc_train.reshape((-1, 1))),
            'acc test': np.squeeze(acc_test.reshape((-1, 1))),
            'conf test': np.squeeze(conf_test.reshape((-1, 1))),
            'C': c * np.ones(n_train * n_rot),
            'class': np.tile(np.arange(n_train), n_rot),
            'ori': np.repeat(np.arange(n_rot), n_train)
        })
        svm_df = pd.concat([svm_df, tmp_df], axis=0)
        if verbose:
            print('C: %.4f'%c)
            print(f"acc train: {tmp_df['acc train'].mean():.2f}")
            print(f"acc test: {tmp_df['acc test'].mean():.2f}")
            print(f"conf test: {tmp_df['conf test'].mean():.2f}")

    if verbose:
        print('Overall summary')
        print(f"acc train: {svm_df['acc train'].mean():.2f}")
        print(f"acc test: {svm_df['acc test'].mean():.2f}")
        print(f"conf test: {svm_df['conf test'].mean():.2f}")

    return svm_df
