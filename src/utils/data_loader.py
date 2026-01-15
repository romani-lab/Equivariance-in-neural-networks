import numpy as np
import pandas as pd
import pickle as pkl
from scipy.stats import zscore 

from utils.stimulus_loader import find_n_trials, find_n_imgs, find_sort_idx
from utils.neuron_selectivity import reliable_neurons, corr_half


def recording_summary_all(model_df, params_df, project_root, verbose=True):
    """
    Run recording summary for all mouse recordings specified in model_df,
    assuming mouse model names start with 'TX',
    with parameters from params_df.
    Store xy positions of reliably responsive neurons and summary metrics 
    in cache folders under project_root/cache/model_name.

    Parameters
    ----------
    model_df : pd.DataFrame
        DataFrame containing model information.
    params_df : pd.DataFrame
        DataFrame containing parameters for each model.
    project_root : str
        Path to the project root directory.
    verbose : bool, default=True
        If True, print progress messages.

    Returns
    -------
    None
    """
    #get parameters
    stim_name = params_df.loc['stimulus'].value
    corr_thre = float(params_df.loc['corr_thre'].value)
    mouse_models = [model for model in model_df.network.unique() if model.startswith('TX')]
    model_df = model_df[model_df['network'].isin(mouse_models)].reset_index(drop=True)

    print(f'Starting single neuron analysis for {len(mouse_models)} mice...')
    print(f'Parameters: threshold for determining reliability={corr_thre}')
    
    #define folders
    data_root = project_root / 'data'
    stim_root = data_root / 'stimuli'
    retinotopy_root = data_root / 'retinotopy'

    #loop over models
    mouse_i0 = model_df.groupby('network').head(1).index.values
    for i0 in mouse_i0:
        model_name = model_df['network'].values[i0]
        monitor = model_df['monitor'].values[i0]
        cache_folder = project_root / 'cache' / model_name
        cache_folder.mkdir(parents=True, exist_ok=True)
        print(f'Processing {model_name}')

        date = model_name[-10:]
        resp_train_trials0, resp_test_trials0, xy_pos, _, idx_resp = \
            mouse_all_train_test_responses(data_root / date, model_name, \
                                        stim_root, stim_name, \
                                        monitor, corr_thre, cache_folder)
        
        #store position of reliably responsive neurons
        reliable_pos_df = pd.DataFrame(xy_pos, columns=['x', 'y'])
        reliable_pos_df['resp'] = reliable_pos_df.index.isin(idx_resp)
        pkl.dump(reliable_pos_df, open( cache_folder / f'reliable_{corr_thre}thre_pos.p', "wb" ) )

        #store summary metrics
        all_areas = model_df[model_df['network']==model_name]['layer'].unique()
        retinotopy_fn = retinotopy_root / f'{model_name}_{stim_name}.npz'
        sum_df = calculate_simple_metrics(resp_train_trials0, resp_test_trials0, idx_resp, all_areas, cache_folder, retinotopy_fn, verbose=verbose)
        pkl.dump(sum_df, open( cache_folder / f'reliable_{corr_thre}thre_summary.p', "wb" ) )
                     

def calculate_simple_metrics(resp_train_trials0, resp_test_trials0, idx_resp, all_areas, cache_folder, retinotopy_fn, verbose=True):
    """
    Calculate simple single-neuron metrics.

    Parameters
    ----------
    resp_train_trials0 : np.ndarray
        trial-by-trial neural responses for training data of shape (n_neurons, n_stim, n_rot, n_trials)
    resp_test_trials0 : np.ndarray
        trial-by-trial neural responses for testing data of shape (n_neurons, n_stim, n_rot, n_trials)
    idx_resp : np.ndarray
        indices of reliably responsive neurons of shape (n_neurons,)
    all_areas : list
        list of areas to analyze
    cache_folder : str
        path to the folder where the results will be saved
    retinotopy_fn : str
        path to the retinotopy file
    verbose : bool, default=True
        If True, print progress messages.

    Returns
    -------
    sum_df : pd.DataFrame
        DataFrame containing the calculated metrics for each neuron
        containing columns:
        - 'id': neuron index   
        - 'area': area of the neuron
        - 'CC_train': mean trial-to-trial correlation for training data
        - 'CC_test': mean trial-to-trial correlation for testing data
        - 'dp_train': class discrimination index for training data
        - 'dp_test': class discrimination index for testing data
        - 'OSI_train': orientation selectivity index for training data
        - 'OSI_test': orientation selectivity index for testing data
    """
    resp_train_avg = resp_train_trials0.mean(-1)
    resp_test_avg = resp_test_trials0.mean(-1)
    n_neurons_all = resp_train_avg.shape[0]
    n_train = resp_train_avg.shape[1]
    n_test = resp_test_avg.shape[1]
    n_rot = resp_train_avg.shape[2]

    #find number of reliably responsive neurons in each area
    n_neurons = np.zeros(len(all_areas),dtype=int)
    frac_neurons = np.zeros(len(all_areas))
    for i in range(len(all_areas)):
        idx_area = find_area_neurons(retinotopy_fn, all_areas[i], idx_resp)
        n_neurons[i] = idx_area.shape[0]
        idx_area_all = find_area_neurons(retinotopy_fn, all_areas[i], np.arange(n_neurons_all))
        frac_neurons[i] = idx_area.shape[0]/idx_area_all.shape[0]
        if verbose:
            print(f'Area {all_areas[i]}: {n_neurons[i]} reliably responsive neurons ({100*frac_neurons[i]:.1f}%)')
    tot_neurons = n_neurons.sum()

    #get cchalf of all reliably responsive neurons in each area
    id_neurons = np.zeros(tot_neurons,dtype=int)
    area_neurons = np.zeros(tot_neurons,dtype='object')
    cc_half_train0 = np.zeros(tot_neurons)
    cc_half_test0 = np.zeros(tot_neurons)
    ct = 0
    for i in range(len(all_areas)):
        idx_area = find_area_neurons(retinotopy_fn, all_areas[i], idx_resp)
        if idx_area.shape[0]>0:
            id_neurons[ct:(ct+n_neurons[i])] = idx_resp[idx_area]
            area_neurons[ct:(ct+n_neurons[i])] = all_areas[i]
            cc_f = cache_folder / f'{all_areas[i]}_cc_half_test.csv'
            if cc_f.exists():
                cc_half_test0[ct:(ct+n_neurons[i])] = np.loadtxt(cc_f, delimiter=',')[:,1]
            else:
                cc_half_test = corr_half(resp_test_trials0[idx_resp[idx_area]], n_perm=100).mean(-1)
                np.savetxt(cc_f, cc_half_test, delimiter=',')
                cc_half_test0[ct:(ct+n_neurons[i])] =  cc_half_test[:,1]
            cc_f2 = cache_folder / f'{all_areas[i]}_cc_half_train.csv'
            if cc_f2.exists():
                cc_half_train0[ct:(ct+n_neurons[i])] = np.loadtxt(cc_f2, delimiter=',').mean(1)
            else:
                cc_half_train = corr_half(resp_train_trials0[idx_resp[idx_area]], n_perm=100).mean(-1)
                np.savetxt(cc_f2, cc_half_train, delimiter=',')
                cc_half_train0[ct:(ct+n_neurons[i])] =  cc_half_train.mean(1)
            ct += n_neurons[i]

    #d prime
    dp_train0 = np.zeros((tot_neurons,n_train))
    for i in range(n_train):
        resp_one = resp_train_avg[id_neurons,i].reshape((tot_neurons,-1))
        resp_other = np.delete(resp_train_avg[id_neurons], i, axis=1).reshape((tot_neurons,-1))
        dp_train0[:,i] = np.array([np.abs(resp_one.mean(1)-resp_other[:,j*n_rot:(j+1)*n_rot].mean(1))/np.sqrt((resp_one.std(1)**2+resp_other[:,j*n_rot:(j+1)*n_rot].std(1)**2)/2) for j in range(n_train-1)]).mean(0)
    dp_train = dp_train0.mean(1)
    dp_test = np.abs(resp_test_avg[id_neurons,0].mean(1)-resp_test_avg[id_neurons,1].mean(1))/np.sqrt((resp_test_avg[id_neurons,0].std(1)**2+resp_test_avg[id_neurons,1].std(1)**2)/2)

    #orientation selectivity (OSI)
    OSI_train = np.zeros((tot_neurons, n_train))
    for i in range(n_train):
        pref_spin_train = np.argmax(resp_train_avg[id_neurons,i], axis=-1)
        resp_PO_train = resp_train_avg[id_neurons,i,pref_spin_train]
        resp_NPO_train = np.array([(np.roll(resp_train_avg[id_neurons[j],i], -pref_spin_train[j]-7,axis=-1)[0]+np.roll(resp_train_avg[id_neurons[j],i], -pref_spin_train[j]-8,axis=-1)[0])/2 for j in range(tot_neurons)])
        resp_min_train = resp_train_avg[id_neurons,i].min(-1)
        OSI_train[:,i] = (resp_PO_train-resp_NPO_train)/(resp_PO_train+resp_NPO_train-2*resp_min_train)
    OSI_train = OSI_train.mean(1)
    OSI_test = np.zeros((tot_neurons, n_test))
    for i in range(n_test):
        pref_spin_test = np.argmax(resp_test_avg[id_neurons,i], axis=-1)
        resp_PO_test = resp_test_avg[id_neurons,i,pref_spin_test]
        resp_NPO_test = np.array([(np.roll(resp_test_avg[id_neurons[j],i], -pref_spin_test[j]-7,axis=-1)[0]+np.roll(resp_test_avg[id_neurons[j],i], -pref_spin_test[j]-8,axis=-1)[0])/2 for j in range(tot_neurons)])
        resp_min_test = resp_test_avg[id_neurons,i].min(-1)
        OSI_test[:,i] = (resp_PO_test-resp_NPO_test)/(resp_PO_test+resp_NPO_test-2*resp_min_test)
    OSI_test = OSI_test.mean(1)

    sum_df = pd.DataFrame({'id': id_neurons,'area': area_neurons, \
                        'CC_train': cc_half_train0, 'CC_test': cc_half_test0,\
                        'dp_train': dp_train, 'dp_test': dp_test, \
                        'OSI_train': OSI_train, 'OSI_test': OSI_test})
    if verbose:
        print(sum_df.groupby('area').agg({'OSI_train': ['mean', 'std'], 'OSI_test': ['mean', 'std'],\
                                          'dp_train': ['mean', 'std'], 'dp_test': ['mean', 'std'],\
                                          'CC_train': ['mean', 'std'], 'CC_test': ['mean', 'std'], 'id': 'count'}))
    
    return sum_df


def find_area_neurons(retinotopy_f, area_name, idx_resp):
    """
    Find indices of responsive neurons in a specific mouse visual cortical area.

    Parameters
    ----------
    retinotopy_f : str
        path to the retinotopy file
    area_name : str
        name of a mouse visual cortical area
    idx_resp : np.ndarray
        array of indices of responsive neurons

    Returns
    -------
    idx : np.ndarray
        array of indices of idx_resp that are in area_name
    """
    #map area names to numbers used in retinotopy file
    area_dict = {'V1': 8, 'LM': 5, 'AL': 6, 'RL': 3, 'PM': 0, 'LI': 7, 'AM': 1}

    ret_df = np.load(retinotopy_f, allow_pickle=True)
    areas = ret_df['area_assign']
    idx_area = np.array([i for i in range(idx_resp.shape[0]) if areas[idx_resp[i]]==area_dict[area_name]])

    return idx_area


def mouse_all_train_test_responses(data_root, rec_name, stim_root, stim_name, monitor, corr_thre, cache_folder, n_perm=100):
    """
    Load normalized responses of a certain recording session, ordered by stimuli.
    Also return the xy positions of neurons and indices of reliably responsive neurons.

    Parameters
    ----------
    data_root : str
        path to the data folder
    rec_name : str
        name of the recording
    stim_root : str
        path to the stimulus file
    stim_name : str
        name of the stimulus file
    monitor : int
        index of the monitor (0: left, 1: front)
    corr_thre : float
        lower threshold for trial-to-trial correlation
    cache_folder : str
        path to the folder where the results will be saved
    n_perm : int, default=100
        number of random samples for computing trial-to-trial correlation

    Returns
    -------
    resp_train_trials_norm : np.ndarray
        array of size (n_neurons, n_stim_train, n_rot, n_trials_train) containing training responses
    resp_test_trials_norm : np.ndarray
        array of size (n_neurons, n_stim_test, n_rot, n_trials_test) containing testing responses
    xy_pos : np.ndarray
        array of size (n_neurons, 2) containing the xy-positions of neurons
    idx_resp : np.ndarray
        array of indices of responsive neurons
    """
    db = np.load(data_root / f'spk_slants_{rec_name}.npz', allow_pickle=True)

    resp_train_trials, resp_test_trials = sort_resp_by_train_test_trials(db['spks'].T, db['istim'], \
                                                                       stim_root, stim_name, monitor=monitor)
    resp_train_trials_norm = normalize_n_reorder_responses(resp_train_trials, db['uspont'], db['mu'], db['sd']) 
    resp_test_trials_norm = normalize_n_reorder_responses(resp_test_trials, db['uspont'], db['mu'], db['sd']) 
    xy_pos = np.array([db['xpos'],db['ypos']]).T

    #reformat responses into (n_neurons, n_stim, n_rot, n_trials)
    n_rot, n_train, n_test = find_n_imgs(stim_name)
    resp_train_trials_norm = resp_train_trials_norm.reshape((resp_train_trials_norm.shape[0], \
                                                             n_train, n_rot, resp_train_trials_norm.shape[2]))
    resp_test_trials_norm = resp_test_trials_norm.reshape((resp_test_trials_norm.shape[0], \
                                                              n_test, n_rot, resp_test_trials_norm.shape[2]))                                                           
    
    #find reliably responsive neurons
    cc_half_train_f = cache_folder / f'{rec_name}_cc_half_train_avg.csv'
    idx_resp = reliable_neurons(resp_train_trials_norm, cc_half_train_f, corr_thre, n_perm=n_perm)
    
    return resp_train_trials_norm, resp_test_trials_norm, xy_pos, idx_resp


def normalize_n_reorder_responses(resp, uspont, mu, std):  
    """
    Normalize responses by z-scoring across neurons (attempt to remove global gain), 
    project out spontaneous activity, and reorder.

    Parameters
    ----------
    resp : np.ndarray
        array of size (n_trials, n_stim, n_neurons) containing responses
    uspont : np.ndarray
        array of size (n_neurons, n_neurons) containing spontaneous PCs
    mu : np.ndarray
        array of size (n_neurons,) containing mean responses
    std : np.ndarray
        array of size (n_neurons,) containing standard deviation of responses

    Returns
    -------
    resp_norm : np.ndarray
        array of size (n_neurons, n_stim, n_trials) containing normalized responses
    """    
    #spontaneous PCs for orthogonal projection
    n_neurons = resp.shape[2]
    n_pc = uspont[0,:].size
    Usp = uspont[:,:n_pc]/np.linalg.norm( uspont[:,:n_pc] , axis=0)[np.newaxis,:]

    #project out spontaneous PCs
    resp_zscored = (resp-mu.flatten()[np.newaxis,np.newaxis,:])/std.flatten()[np.newaxis,np.newaxis,:]
    resp_zscored = resp_zscored.reshape((resp.shape[0]*resp.shape[1],n_neurons)).T
    resp_proj = resp_zscored-Usp @ (Usp.T @ resp_zscored)

    #normalize out global gain
    resp_norm = zscore(resp_proj,0).reshape((n_neurons,resp.shape[0],resp.shape[1]))
    resp_norm = np.transpose(resp_norm,[0,2,1])

    return resp_norm 


def sort_resp_by_train_test_trials(spks, istim, stim_root, stim_name, monitor=0):
    """
    Sort responses by training and testing stimuli.

    Parameters
    ----------
    spks : np.ndarray
        array of size (n_trials, n_neurons) containing responses
    istim : np.ndarray
        array of size n_trials containing stimulus indices
    stim_root : str
        path to the stimulus file
    stim_name : str
        name of the stimulus file
    monitor : int, default=0
        index of the monitor (0: left, 1: front)

    Returns
    -------
    resp_istim_train : np.ndarray
        array of size (n_trials_train, n_stim_train, n_neurons) containing responses for training trials
    resp_istim_test : np.ndarray
        array of size (n_trials_test, n_stim_test, n_neurons) containing responses for testing trials
    """    
    n_rot, n_train, n_test = find_n_imgs(stim_name)
    n_trials_train, n_trials_test = find_n_trials(stim_name)
        
    #shift in stimulus index, depending on monitor
    n_stim_train = n_rot*n_train
    n_stim_test = n_rot*n_test
    idx_shift_train = monitor*(n_stim_train + n_stim_test) + n_stim_test
    idx_shift_test = monitor*(n_stim_train + n_stim_test)
 
    #find trials where each stimulus was presented on the monitor
    istim_train = np.zeros((n_stim_train,n_trials_train), dtype=int)
    istim_test = np.zeros((n_stim_test,n_trials_test), dtype=int)
    for i in range(n_stim_train):
        istim_train[i,:]=np.nonzero( istim==idx_shift_train+i )[0]
    for i in range(n_stim_test):
        istim_test[i,:]=np.nonzero( istim==idx_shift_test+i )[0]
            
    #find corresponding responses
    n_neurons = spks.shape[1]
    resp_istim_train = np.zeros((n_trials_train,n_stim_train,n_neurons))
    resp_istim_test = np.zeros((n_trials_test,n_stim_test,n_neurons))
    for i in range(n_stim_train):
        resp_istim_train[:,i,:]=spks[istim_train[i,:],:]
    for i in range(n_stim_test):
        resp_istim_test[:,i,:]=spks[istim_test[i,:],:]

    #sort stimuli
    idx_train, idx_test = find_sort_idx(stim_root, stim_name)
    resp_sorted_train = resp_istim_train[:,idx_train-n_stim_test]
    resp_sorted_test = resp_istim_test[:,idx_test]

    return resp_sorted_train, resp_sorted_test