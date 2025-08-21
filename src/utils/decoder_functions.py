import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from concurrent.futures import ProcessPoolExecutor
from sklearn.utils.extmath import randomized_svd

from utils.helper import find_kn_unique_samples, find_eigendecomp
from utils.model_loader import model_train_test_responses
from utils.data_loader import mouse_all_train_test_responses, find_area_neurons
from utils.equivariance_functions import calculate_proj_rot_shift, calculate_predictions


def decode_all(model_df, params_df, project_root, c_range=[0.001], equivariant=False):
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
    equivariant : bool, default=False
        whether to use equivariant subspace projection (requires 'n_procs' in params_df)
        suffix = '_{n_procs}dim' will be added to output filenames if True

    Returns
    -------
    None
        saves results to CSV files in project_root / 'cache' / model_name:
        - orientation decoding: {layer_name}_ori2x_decoder_{n_neurons}avg{suffix}.csv
        - texture decoding: {layer_name}_class2x_decoder_{n_neurons}avg_{c_range[0]}C{suffix}.csv
    """
    #get parameters
    stim_name = params_df.loc['stimulus'].value
    corr_thre = float(params_df.loc['corr_thre'].value)
    inv_snr = float(params_df.loc['inv_snr'].value)
    n_neurons = int(params_df.loc['n_neurons'].value)
    n_max = int(params_df.loc['n_max'].value)
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
    else:
        d = None
        suffix = ''

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
        print(f'Processing {model_name} {layer_name}')

        if np.isin(model_name,['pixels','alexnet']):
            resp_train_trials, resp_test_trials, _, _, idx_resp = model_train_test_responses(model_name, layer_name, \
                                                                                    stim_root, stim_name, \
                                                                                    monitor, corr_thre, cache_folder, \
                                                                                    inv_snr=inv_snr)
            resp_train_trials = resp_train_trials[idx_resp]
            resp_test_trials = resp_test_trials[idx_resp]
        else:
            #to speed up: for first layer of each network, load all responsive neurons
            if layer_name==model_df[model_df['network']==model_name]['layer'].values[0]:
                date = model_name[-10:]
                resp_train_trials0, resp_test_trials0, _, _, idx_resp = \
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

        #concatenate training and test responses (for test, select 2 out of 80 trials)
        resp_all_trials = np.concatenate((resp_train_trials, resp_test_trials[:,:,:,:2]), axis=1)

        result_df = orientation2x_2d_decoder(resp_all_trials.mean(-1), n_neurons, d=d, n_max=n_max)
        result_df.to_csv(cache_folder / f'{layer_name}_ori2x_decoder_{n_neurons}avg{suffix}.csv', index=False)

        svm_df = texture2x_svm_decoder_parallel(resp_all_trials, n_neurons, c_range, d=d, n_max=n_max)
        svm_df.to_csv(cache_folder / f'{layer_name}_class2x_decoder_{n_neurons}avg_{c_range[0]}C{suffix}.csv', index=False)


def align_by_rotating(x):
    """
    Align the first rotation value to the x-axis for each stimulus and dimension pair.

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
    x_rot : np.ndarray
        normalized 2D projected responses of shape (2, n_stim, n_rot)
    """
    n_neurons, n_test, n_rot = x_test.shape
    center = x_train.reshape((n_neurons, -1)).mean(axis=1)

    x_train_shift = x_train.copy()
    for i in range(x_train.shape[1]):
        x_train_shift[:,i,:] -= x_train[:,i,:].mean(axis=1)[:,np.newaxis]
    x_train_shift = x_train_shift.reshape((n_neurons, -1))
    U_train, _, _ = randomized_svd(x_train_shift, n_components=n_rot-1, random_state=0)

    x_test_proj = np.einsum('ij,ikl->jkl', U_train[:,:2], x_test - center[:,np.newaxis,np.newaxis])
    x_test_proj = align_by_rotating(x_test_proj)
    if x_test_proj[1,:,1].mean()<0:
        x_test_proj[1] = -x_test_proj[1]

    x_test_proj = x_test_proj / np.linalg.norm(x_test_proj, axis=0)

    return x_test_proj


def orientation2x_2d_decoder(x_train, n_neurons, rot_step=24, 
                             d=None, n_max=1, verbose=False):
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
    d : int, default=None
        dimension for equivariant subspace projection
    n_max : int, default=1
        maximum number of random neuron samples
    verbose : bool, default=False
        whether to print performance metrics

    Returns
    -------
    results_df : pd.DataFrame
        dataframe with columns 'R2', 'res', 'class', 'ori test'
    """    
    #in case x_train.shape[0] is smaller than n_neurons
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
    for k in range(k_rand):
        x_train_k = x_train[k_samp[k]]
        for i in range(n_train):
            x_train_k_i = x_train_k[:,np.arange(n_train)!=i]
            x_test_k_i = x_train_k[:,i:(i+1)]

            if d is None:
                x_train_k_proj = x_train_k_i
                x_test_k_proj = x_test_k_i
            #if d is not None, project to invariant subspace
            else:
                proj_k, proj_shift_k, rot_k, _ = calculate_proj_rot_shift(x_train_k_i, d)
                eigenval_k, eigenvec_k = find_eigendecomp(rot_k, n_rot)
                id1 = np.where( (np.abs(eigenval_k)>=.5)&(np.abs(eigenval_k)<1.5) )[0]
                eigenvec1 = eigenvec_k[:,id1]
                eigenproj_k = proj_k@eigenvec1
                x_train_k_proj = np.einsum('ij,ikl->jkl', eigenproj_k, x_train_k_i-proj_shift_k[:,np.newaxis,np.newaxis])
                x_test_k_proj = np.einsum('ij,ikl->jkl', eigenproj_k, x_test_k_i-proj_shift_k[:,np.newaxis,np.newaxis])

            #dependent variables for decoding
            x_test_rot = proj12_n_norm(x_train_k_proj, x_test_k_proj)
            #remove reference point because used for alignment
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
        'ori test': np.tile(np.arange(n_rot)[1:], n_train)
    })

    if verbose:
        print(f"R2 test: {results_df['R2'].mean():.2f}")
        print(f"res test: {results_df['res'].mean():.2f}")

    return results_df


def svm2x_train_test(c, x_train_k, n_rot_test, d=None):
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
    n_neurons, n_train, n_rot, n_trials_train = x_train_k.shape
    n_rot_train = n_rot - n_rot_test
    
    #class labels and independent variables for decoding
    c_train = np.array(range(n_train)) 
    y_train = np.squeeze(np.repeat(c_train, n_rot_train*n_trials_train))
    y_test = np.squeeze(np.repeat(c_train, n_rot_test))

    x_train_k_i = x_train_k[:,:,:n_rot_train]
    x_test_k_i = x_train_k[:,:,n_rot_train:].mean(-1)
    
    #If d is not None, project to invariant subspace
    if d is None:
        x_train_k_proj = x_train_k_i.reshape((n_neurons,-1))
        x_test_k_proj = x_test_k_i.reshape((n_neurons,-1))
    else:
        x_train_k_avg = x_train_k_i.mean(-1)
        proj_k, _, rot_k, _ = calculate_proj_rot_shift(x_train_k_avg, d)
        eigenval_k, eigenvec_k = find_eigendecomp(rot_k, n_rot)
        id0 = np.where(np.abs(eigenval_k)<0.5)[0]
        d0 = len(id0)
        eigenproj_k = proj_k@eigenvec_k[:,id0]
        x_train_k_proj = np.einsum('ij,iklm->jklm', eigenproj_k, x_train_k_i).reshape((d0,-1))
        x_test_k_proj = np.einsum('ij,ikl->jkl', eigenproj_k, x_test_k_i).reshape((d0,-1))

    # Train and test SVM
    svm = LinearSVC(dual=False, C=c, class_weight=None).fit(x_train_k_proj.T, y_train)
    acc_train = svm.score(x_train_k_proj.T, y_train)
    y_pred = svm.predict(x_test_k_proj.T)
    acc_test = (y_pred == y_test)
    confidence = svm.decision_function(x_test_k_proj.T)
    conf_test = np.diag(confidence)

    return acc_train, acc_test, conf_test


def compute_single_k_svm(args):
    """
    Helper function to compute SVM decoding for a single random neuron sample 
    with leave-one-out cross-validation across orientations.

    Parameters
    ----------
    args : tuple
        tuple containing (k, x_train, k_samp, c, d)

    Returns
    -------
    acc_train_k : np.ndarray
        training accuracies of shape (n_rot, n_train)
    acc_test_k : np.ndarray  
        test accuracies of shape (n_rot, n_train)
    conf_test_k : np.ndarray
        test confidences of shape (n_rot, n_train)
    """
    k, x_train, k_samp, c, d = args
    
    n_rot, n_train = x_train.shape[2], x_train.shape[1]
    x_train_k = x_train[k_samp[k]]
    
    acc_train_k = np.zeros((n_rot, n_train))
    acc_test_k = np.zeros((n_rot, n_train))
    conf_test_k = np.zeros((n_rot, n_train))
    
    for r in range(n_rot):
        x_train_k_r = np.roll(x_train_k, shift=-r, axis=2)
        acc_train_res, acc_test_res, conf_test_res = svm2x_train_test(c, x_train_k_r, 1, d=d)
        acc_train_k[r] = acc_train_res
        acc_test_k[r] = acc_test_res
        conf_test_k[r] = conf_test_res
    
    return acc_train_k, acc_test_k, conf_test_k


def texture2x_svm_decoder_parallel(x_train, n_neurons, c_range, 
                                   d=None, n_max=1, verbose=False):
    """
    Decode image class from neuronal responses using linear SVM 
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
        # Prepare arguments for parallel processing
        args_list = [(k, x_train, k_samp, c, d) for k in range(k_rand)]
        
        # Process k_rand samples in parallel
        if k_rand <= 1:
            # Fallback to sequential processing for single sample
            results = [compute_single_k_svm(args_list[0])]
        else:
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(compute_single_k_svm, args_list))
        
        # Aggregate results across k_rand samples
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