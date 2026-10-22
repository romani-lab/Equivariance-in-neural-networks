import torch
from torchvision import models
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
import numpy as np

from utils.stimulus_loader import load_img_sorted, standard_img_transform, find_n_trials, find_n_imgs
from utils.neuron_selectivity import reliable_neurons
from more_anns.feature_extractor import FeatureExtractor

RANDOM_ALEXNET_SEED = 0   # seed of the random-init AlexNet ('random_alexnet'), see alexnet_responses


def is_model_row(model_name):
    """Whether a params row is an ANN / pixel model (as opposed to a mouse recording).

    Used as the shared gate in equivariance_functions, equivariance_validation and
    decoder_functions so a new model prefix only has to be registered here.

    Recognises: 'pixels', 'alexnet', 'random_alexnet', the trained AlexNet variants
    ('alt_alexnet_*') and the CNNs with built-in equivariance ('eqv*': 'eqv-c15', 'eqv-c15r',
    'eqv-c1'; checkpoint at cache/more_anns/<name>/best_model.pth).
    """
    return (model_name in ('pixels', 'alexnet', 'random_alexnet')
            or model_name.startswith('alt_alexnet')
            or model_name.startswith('eqv'))


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
    resp_train_trials, resp_test_trials = train_test_split_w_noise(resp_all, stim_name, \
                                                                   inv_snr=inv_snr, noise_model=noise_model)                                                   
    
    #find reliably responsive neurons
    #the cache is noise-model-specific (non-gaussian models get a suffix so they do
    #not clobber the default gaussian caches)
    noise_suffix = '' if noise_model == 'gaussian' else f'_{noise_model}'
    cc_half_train_f = cache_folder / f'{layer_name}_cc_half_train_avg{noise_suffix}.csv'
    idx_resp = reliable_neurons(resp_train_trials, cc_half_train_f, corr_thre, n_perm=n_perm)

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
    if noise_model=='noiseless':
        #inv_snr = 0 clean limit: repeat the deterministic responses across trials with no noise
        #added. A distinct noise_model label (rather than gaussian + inv_snr=0) gives every cache
        #its own '_noiseless' suffix so it never collides with the default (gaussian, inv_snr>0)
        #caches. Split-half ceilings become 1, so CC_norm is the raw prediction correlation.
        resp_train_trials = np.repeat(resp_train, n_trials_train).reshape((n_neurons, -1))
        resp_test_trials = np.repeat(resp_test, n_trials_test).reshape((n_neurons, -1))
    elif noise_model=='gaussian':
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
    Create a sampler for neuron gain loadings.

    Parameters
    ----------
    a : np.ndarray
        neuron gain loadings of shape (n_neurons,)
    rng : np.random.Generator or None, default=None
        random number generator used for sampling

    Returns
    -------
    sample_a : callable
        function that samples M gain loadings and returns an array of shape (M,)
    """
    rng = np.random.default_rng() if rng is None else rng
    a = np.asarray(a).ravel()

    def sample_a(M):
        """
        Sample neuron gain loadings with replacement.

        Parameters
        ----------
        M : int
            number of gain loadings to sample

        Returns
        -------
        a_samp : np.ndarray
            sampled gain loadings of shape (M,)
        """
        a_samp = rng.choice(a, size=M, replace=True)

        # enforce sign convention (optional)
        if np.mean(a_samp) < 0:
            a_samp = -a_samp
        return a_samp

    return sample_a


def add_mult_gain(x, g, sample_a):
    """
    Add multiplicative gain to stimulus responses.

    Parameters
    ----------
    x : np.ndarray
        response array of shape (M, n_stim)
    g : np.ndarray
        stimulus-by-trial gain factors of shape (n_stim, n_trials)
    sample_a : callable
        function that takes M and returns sampled gain loadings of shape (M,)

    Returns
    -------
    x_trials : np.ndarray
        response array with multiplicative gain of shape (M, n_stim, n_trials)
    a_samp : np.ndarray
        sampled gain loadings of shape (M,)
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
        array of size (n_neurons, n_stim*n_rot) containing responses
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
    elif model_name=='random_alexnet':
        resp_all, xy_pos = alexnet_responses(layer_name, img_all_transfo, pretrained=False)
    elif model_name.startswith('alt_alexnet'):
        check_point_name = model_name[12:]
        checkpoint_path = stim_root.parent.parent / 'cache' / 'more_anns' / check_point_name / 'best_model.pth'
        if not checkpoint_path.exists():
            raise ValueError(f'Alternative AlexNet model should be saved as {checkpoint_path}')
        resp_all, xy_pos = alt_alexnet_responses(checkpoint_path, layer_name, img_all_transfo)
    elif model_name.startswith('eqv'):
        #equivariant CNN checkpoint (cache/more_anns/<network>/best_model.pth); the folder
        #name is the full network name (e.g. eqv-c15). Reuses alt_alexnet_responses,
        #whose only assumption is the layer_dict names and a (n_img, C, H, W) activation.
        checkpoint_path = stim_root.parent.parent / 'cache' / 'more_anns' / model_name / 'best_model.pth'
        if not checkpoint_path.exists():
            raise ValueError(f'Equivariant CNN model should be saved as {checkpoint_path}')
        resp_all, xy_pos = alt_alexnet_responses(checkpoint_path, layer_name, img_all_transfo)
    else:
        raise ValueError('Model not supported')
    
    #select units with xy_pos on the potentially resized image
    idx_center = np.where( (np.abs(xy_pos[:,0])<=size2/112) & (np.abs(xy_pos[:,1])<=size2/112) )[0] #& (resp_all.std(1)>0)

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
        array of size (n_y*n_x, n_all) containing pixel responses (red channel)
    xy_pos : np.ndarray
        array of size (n_y*n_x, 2) containing spatial positions
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
        array of size (n_channels*n_features, n_all) containing intermediate responses
    """
    resp_all = np.zeros((img_all.shape[0],n_channels,n_features))
    for i in range(img_all.shape[0]):
        batch_i = torch.unsqueeze(torch.from_numpy(img_all[i]), 0)
        mid_outputs, _ = mid_getter(batch_i)
        resp_all[i] = mid_outputs[layer_name].detach().numpy().reshape((n_channels,n_features))    
    resp_all = resp_all.reshape((img_all.shape[0],-1)).T

    return resp_all


def alt_alexnet_responses(checkpoint_path, layer_name, img_all):
    """
    Responses of all artificial units in layer layer_name of AlexNet
    with weights defined by checkpoint_path
    to images in img_all and their spatial positions.

    Parameters
    ----------
    checkpoint_path : pathlib.Path
        path to the checkpoint file (best_model.pth)
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
        array of size (n_channels*n_features, n_all) containing intermediate responses
    xy_pos : np.array
        array of size (n_channels*n_features, 2) containing spatial positions
    """
    #default weights are for pretraining on ImageNet
    extractor = FeatureExtractor(checkpoint_path)
    #conversion from feature name to layer name
    layer_dict = {
    'conv1': 'backbone.features.0',
    'conv2': 'backbone.features.3',
    'conv3': 'backbone.features.6',
    'conv4': 'backbone.features.8',
    'conv5': 'backbone.features.10',
    }

    backbone_layer = layer_dict[layer_name]
    activations = extractor.extract_activations(img_all, layers=[backbone_layer])
    act = activations[backbone_layer]  # shape: (n_all, n_channels, H, W)

    n_images = act.shape[0]
    n_channels = act.shape[1]
    n_features = act.shape[2] * act.shape[3]
    resp_all = act.reshape((n_images, n_channels * n_features)).T

    xy_pos = spatial_positions(n_features, n_channels)

    return resp_all, xy_pos


def alexnet_responses(layer_name, img_all, pretrained=True):
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
    pretrained : bool, default=True
        if True, use ImageNet-pretrained weights; if False, use random weights

    Returns
    -------
    resp_all : np.array
        array of size (n_channels*n_features, n_all) containing intermediate responses
    xy_pos : np.array
        array of size (n_channels*n_features, 2) containing spatial positions
    """
    if not pretrained:
        #fixed seed for the random-init network ('random_alexnet'), so that every analysis of
        #the same row (reliable-unit selection, equivariance, validation, decoding) sees the
        #same random weights and hence the same units
        torch.manual_seed(RANDOM_ALEXNET_SEED)
    model = models.alexnet(weights='DEFAULT' if pretrained else None)
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

