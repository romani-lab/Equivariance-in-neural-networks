import torch
from torchvision import models
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
import numpy as np

from utils.stimulus_loader import load_img_sorted, standard_img_transform, find_n_trials, find_n_imgs
from utils.neuron_selectivity import reliable_neurons


def model_train_test_responses(model_name, layer_name, stim_root, stim_name, size2, corr_thre, cache_folder, 
                               inv_snr=0, n_perm=100, noise_model='gaussian'):
    """
    Find responses of all artificial units in layer_name of model_name to images in stim_name 
    and their spatial positions.
    Split the responses into training and testing sets based on stimuli and add Gaussian noise.

    Parameters
    ----------
    model_name : str
        name of the model
    layer_name : str
        name of the intermediate layer
    stim_root : str
        path to the stimulus file
    stim_name : str
        name of the stimulus file
    size2 : int 
        half width of the image after scaling down
    corr_thre : float
        threshold for trial-to-trial correlation
    cache_folder : str
        path to the folder where the results will be saved
    inv_snr : float, default=0
        inverse signal-to-noise ratio
    n_perm : int, default=100
        number of permutations to compute
    noise_model : str, default='gaussian'
        type of noise model to use, options are 'gaussian', 'poisson', 'mult_gain'

    Returns
    -------
    resp_train_trials : np.ndarray
        array of size (n_neurons, n_stim_train, n_rot, n_trials_train) containing training responses
        with noise added
    resp_test_trials : np.ndarray
        array of size (n_neurons, n_stim_test, n_rot, n_trials_test) containing testing responses
        with noise added
    xy_pos : np.ndarray
        array of size (n_neurons, 2) containing spatial positions
    idx_resp : np.ndarray
        array of indices of responsive neurons    
    """
    resp_all, xy_pos = model_responses(model_name, layer_name, stim_root, \
                                    stim_name, size2=size2)
    resp_train_trials, resp_test_trials = train_test_split_w_noise(resp_all, stim_name, inv_snr=inv_snr, noise_model=noise_model)                                                   
    
    #find reliably responsive neurons
    cc_half_train_f = cache_folder / f'{layer_name}_cc_half_train_avg.csv'
    idx_resp = reliable_neurons(resp_train_trials, cc_half_train_f, 
                                                                             corr_thre, n_perm=n_perm)

    return resp_train_trials, resp_test_trials, xy_pos, idx_resp


def train_test_split_w_noise(resp_all, stim_name, inv_snr=0, noise_model='gaussian'):
    """
    Add noise to the responses and split into training and testing sets based on stimuli.

    Parameters
    ----------
    resp_all : np.ndarray
        array of size (n_neurons, n_stim*n_rot) containing responses
    stim_name : str
        name of the stimulus file
    inv_snr : float, default=0
        inverse signal-to-noise ratio
    noise_model : str, default='gaussian'
        type of noise model to use, options are 'gaussian', 'poisson', 'mult_gain'

    Returns
    -------
    resp_train_trials : np.ndarray
        array of size (n_neurons, n_stim_train, n_rot, n_trials_train) containing training responses
        with noise added according to the specified noise_model and inv_snr
    resp_test_trials : np.ndarray
        array of size (n_neurons, n_stim_test, n_rot, n_trials_test) containing testing responses
            with noise added according to the specified noise_model and inv_snr
    """
    n_rot, n_train, n_test = find_n_imgs(stim_name)
    resp_train = resp_all[:,:n_train*n_rot]
    resp_test = resp_all[:,n_train*n_rot:]

    sigma0 = np.sqrt(inv_snr)
    n_neurons = resp_all.shape[0]
    n_trials_train, n_trials_test = find_n_trials(stim_name)

    #add Gaussian noise with mean 0 and scale the train std by sigma0
    np.random.seed(4)
    if noise_model=='gaussian':
        dresp_train = np.array([np.random.normal(0, sigma0*resp_train[i].std(), n_train*n_rot*n_trials_train) for i in range(n_neurons)])
        dresp_test = np.array([np.random.normal(0, sigma0*resp_train[i].std(), n_test*n_rot*n_trials_test) for i in range(n_neurons)])
        resp_train_trials = np.repeat(resp_train, n_trials_train).reshape((n_neurons, -1)) + dresp_train
        resp_test_trials = np.repeat(resp_test, n_trials_test).reshape((n_neurons, -1)) + dresp_test
    elif noise_model=='poisson':
        dresp_train = np.array([np.random.poisson(lam=sigma0**2*resp_train[i].std()**2, size=(n_train*n_rot*n_trials_train)) for i in range(n_neurons)])
        dresp_test = np.array([np.random.poisson(lam=sigma0**2*resp_train[i].std()**2, size=(n_test*n_rot*n_trials_test)) for i in range(n_neurons)])
        resp_train_trials = np.repeat(resp_train, n_trials_train).reshape((n_neurons, -1)) + dresp_train
        resp_test_trials = np.repeat(resp_test, n_trials_test).reshape((n_neurons, -1)) + dresp_test
    elif noise_model=='mult_gain':
        from pathlib import Path
        local_dir = Path(__file__).resolve().parent
        gain_params_dir = local_dir / "gain_params"
        rng = np.random.default_rng(4)

        a_train = np.loadtxt(gain_params_dir / 'a_train.csv', delimiter=',')
        g_train = np.loadtxt(gain_params_dir / 'g_train.csv', delimiter=',')
        sample_a = make_a_sampler(a_train, rng=rng)
        resp_train_trials, _ = add_mult_gain(resp_train, g_train, sample_a)
        a_test = np.loadtxt(gain_params_dir / 'a_test.csv', delimiter=',')
        g_test = np.loadtxt(gain_params_dir / 'g_test.csv', delimiter=',')
        sample_a2 = make_a_sampler(a_test, rng=rng)
        resp_test_trials, _ = add_mult_gain(resp_test, g_test, sample_a2)
    
    #reshape
    resp_train_trials = resp_train_trials.reshape((n_neurons, n_train, n_rot, n_trials_train))
    resp_test_trials = resp_test_trials.reshape((n_neurons, n_test, n_rot, n_trials_test))

    return resp_train_trials, resp_test_trials


def make_a_sampler(a, rng=None):
    """
    a: (N,) neuron gain loadings from data
    Returns: sample_a(M) -> (M,)
    """
    rng = np.random.default_rng() if rng is None else rng
    a = np.asarray(a).ravel()

    def sample_a(M):
        a_samp = rng.choice(a, size=M, replace=True)

        # enforce sign convention (optional)
        if np.mean(a_samp) < 0:
            a_samp = -a_samp
        return a_samp

    return sample_a


def add_mult_gain(x, g, sample_a):
    """
    x: (M, n_stim)
    g: (n_stim, n_trials)
    Returns:
      y_trials: (M, n_stim, n_trials)
      a_samp: (M,)
    """
    x = np.asarray(x)
    g = np.asarray(g)

    M, n_stim = x.shape
    n_stim_g, _ = g.shape
    assert n_stim == n_stim_g, "Stimulus dimension mismatch"

    a_samp = sample_a(M)

    x_trials = x[:, :, None] * (1.0 + a_samp[:, None, None] * g[None, :, :])

    return x_trials, a_samp


def model_responses(model_name, layer_name, stim_root, stim_name, size2):
    """
    Find responses of all artificial units in layer_name of model_name to all images in stim_name 
    and their corresponding spatial positions.
    Only works for 'pixels' and 'alexnet' models.

    Parameters
    ----------
    model_name : str
        name of the model
    layer_name : str
        name of the intermediate layer
    stim_root : str
        path to the stimulus file
    stim_name : str
        name of the stimulus file
    size2 : int
        half width of the image, used for scaling down

    Returns
    -------
    resp_all : np.ndarray
        array of size (n_neurons, n_stim*n_rot*n_trials) containing responses
    xy_pos : np.ndarray
        array of size (n_neurons, 2) containing spatial positions
    """
    #load images and transform
    img_all_sorted = load_img_sorted(stim_root, stim_name)    
    img_all_transfo = standard_img_transform(img_all_sorted, size2=size2)

    #load unit responses and positions
    if model_name=='pixels':
        resp_all, xy_pos = pixel_responses(img_all_transfo)
    elif model_name=='alexnet':
        resp_all, xy_pos = alexnet_responses(layer_name, img_all_transfo)
    else:
        raise ValueError('Model not supported')
    
    #select units with xy_pos on the potentially resized image
    idx_center = np.where( (np.abs(xy_pos[:,0])<=size2/112) & (np.abs(xy_pos[:,1])<=size2/112) & (resp_all.std(1)>0) )[0]

    return resp_all[idx_center], xy_pos[idx_center]


def spatial_positions(n_features, n_channels):
    """
    Find spatial positions for n_features and n_channels, assuming uniform sampling.

    Parameters
    ----------
    n_features : int
        number of features
    n_channels : int
        number of channels

    Returns
    -------
    xy_pos : np.ndarray
        array of size (n_channels*n_features, 2) containing spatial positions
    """
    xv, yv = np.meshgrid( range(int(np.sqrt(n_features))), range(int(np.sqrt(n_features))) )
    xv = np.squeeze( xv.reshape((-1,1)) )
    yv = np.squeeze( yv.reshape((-1,1)) )
    xy_pos = np.zeros((n_channels, n_features, 2))
    xy_pos[:] = np.vstack([xv,yv]).T
    xy_pos = xy_pos.reshape((-1,2))
    #rescale to [-1,1]
    xy_pos = 2*(xy_pos/xy_pos.max(0)[np.newaxis,:]-.5)

    return xy_pos


def pixel_responses(img_all):
    """
    Convert image into pixel "responses" (from red channel) and spatial positions.

    Parameters
    ----------
    img_all : np.ndarray
        array of size n_all x 3 x n_y x n_x where
        n_all is the number of images
        n_y and n_x are the dimensions of the images

    Returns
    -------
    resp_all : np.ndarray
        array of size (n_all, n_y*n_x) containing pixel values
    xy_pos : np.ndarray
        array of size (n_y*n_x, 2) containing pixel values
    """
    resp_all = ( img_all[:,0].reshape((img_all.shape[0],img_all.shape[2]*img_all.shape[3])) ).T
    xy_pos = spatial_positions(img_all.shape[3]*img_all.shape[2], 1)
    
    return resp_all, xy_pos


def mid_getter_responses(mid_getter, img_all, layer_name, n_channels, n_features):
    """
    Responses of all artificial units in layer_name to images in img_all.

    Parameters
    ----------
    mid_getter : torch_intermediate_layer_getter.IntermediateLayerGetter
        object to get intermediate responses
    img_all : np.array
        array of size n_all x 3 x n_y x n_x where
        n_all is the number of images
        n_y and n_x are the dimensions of the images
    layer_name : str
        name of the intermediate layer
    n_channels : int    
        number of channels in corresponding layer
    n_features : int
        number of features in corresponding layer

    Returns
    ------- 
    resp_all : np.array
        array of size (n_all, n_channels, n_features) containing intermediate responses
    """
    resp_all = np.zeros((img_all.shape[0],n_channels,n_features))
    for i in range(img_all.shape[0]):
        batch_i = torch.unsqueeze(torch.from_numpy(img_all[i]), 0)
        mid_outputs, _ = mid_getter(batch_i)
        resp_all[i] = mid_outputs[layer_name].detach().numpy().reshape((n_channels,n_features))    
    resp_all = resp_all.reshape((img_all.shape[0],-1)).T

    return resp_all


def alexnet_responses(layer_name, img_all):
    """
    Responses of all artificial units in layer layer_name of AlexNet 
    to images in img_all and their spatial positions.

    Parameters
    ----------
    layer_name : str
        name of the intermediate layer
    img_all : np.array
        array of size (n_all, 3, n_y, n_x) where
        n_all is the number of images
        3 is for RGB channels
        n_y and n_x are the dimensions of the images    

    Returns
    -------
    resp_all : np.array
        array of size (n_all, n_channels, n_features) containing intermediate responses
    xy_pos : np.array
        array of size (n_channels*n_features, 2) containing spatial positions
    """
    #default weights are for pretraining on ImageNet
    model = models.alexnet(weights='DEFAULT')
    #conversion from feature name to layer name
    return_layers = {
    'features.0': 'conv1',
    'features.3': 'conv2',
    'features.6': 'conv3',
    'features.8': 'conv4',
    'features.10': 'conv5',
    }
    #number of features and channels in corresponding layer
    if layer_name=='conv1': 
        idx_feature=0
        n_features=55**2
    elif layer_name=='conv2': 
        idx_feature=3
        n_features=27**2
    elif layer_name=='conv3': 
        idx_feature=6
        n_features=13**2
    elif layer_name=='conv4': 
        idx_feature=8
        n_features=13**2
    elif layer_name=='conv5': 
        idx_feature=10
        n_features=13**2
    n_channels = model.features[idx_feature].out_channels    

    mid_getter = MidGetter(model, return_layers=return_layers, keep_output=False)
    resp_all = mid_getter_responses(mid_getter, img_all, layer_name, n_channels, n_features)
    xy_pos = spatial_positions(n_features, n_channels)

    return resp_all, xy_pos

