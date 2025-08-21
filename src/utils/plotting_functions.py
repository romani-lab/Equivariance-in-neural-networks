import numpy as np
import pickle as pkl
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helper import compute_significance
from utils.stimulus_loader import load_img_sorted


def plot_equivariance_pcorr(model_df, params_df, project_root, height=150, width=680, \
                           color_inv='rgba(247, 148, 29, 1.0)', color_tuned='rgba(139, 94, 60, 1.0)', color_err='rgba(169, 169, 169, 1.0)'):
    """
    Plot Pearson correlation of invariant and tuned subspaces across layers.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_inv : str, default='rgba(247, 148, 29, 1)'
        color for invariant dimensions
    color_tuned : str, default='rgba(139, 94, 60, 1)'
        color for tuned dimensions

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results' / 'pcorr_{n_neurons}avg_{d}dim.pdf'
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    d = int(params_df.loc['n_procs'].value)
    results_folder = project_root / 'results'
    results_folder.mkdir(parents=True, exist_ok=True)

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1, 'LM': 2, 'RL': 3, 'AL': 4, 'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9}
    #shift x-positions for invariant and tuned
    color_to_dx_dict = {f'{color_inv}': -0.1, f'{color_tuned}': 0.1}

    #collect results across layers
    var_df = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        var_f = cache_folder / f'{layer_name}_var_dim_{n_neurons}avg_{d}dim.csv'
        if not var_f.exists():
            continue
        tmp_df = pd.read_csv(var_f)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        var_df = pd.concat([var_df, tmp_df.copy()], axis=0)
    var_df = var_df.reset_index(drop=True)

    #significance for invariant dimensions
    inv_all_df = var_df[np.abs(var_df['evals'])<0.5]
    inv_pc_df = inv_all_df.groupby(['model','area','stim'])['pc test'].mean().reset_index()
    compute_significance(inv_pc_df.rename(columns={'pc test': 'score'}), results_folder / 'pcorr_inv_significance.csv', \
                         alternative='less')

    #significance for tuned dimensions
    tuned_all_df = var_df[(np.abs(var_df['evals'])>=0.5)&(np.abs(var_df['evals'])<1.5)]
    closest_idx = tuned_all_df.groupby(['model','area','stim'])['evals'].apply(\
                    lambda x: np.abs(x.abs()-1).nsmallest(2).index).explode().values
    tuned_pc_df = var_df.loc[closest_idx[::2], ['model','area', 'stim', 'pc test']].reset_index(drop=True)
    tuned_pc_df['pc test'] += var_df.loc[closest_idx[1::2], 'pc test'].values
    tuned_pc_df['pc test'] /= 2
    compute_significance(tuned_pc_df.rename(columns={'pc test': 'score'}), results_folder / 'pcorr_tuned_significance.csv', \
                         alternative='less')
    
    #compute means and sems in each layer
    inv_pc_avg_df = inv_pc_df.groupby(['model','area'])['pc test'].mean().reset_index().rename(columns={'pc test': 'pc'})
    inv_pc_avg_df['erry'] = inv_pc_df.groupby(['model','area'])['pc test'].std().values/np.sqrt(var_df['stim'].nunique())
    inv_pc_avg_df['cat'] = 0
    tuned_pc_avg_df = tuned_pc_df.groupby(['model','area'])['pc test'].mean().reset_index().rename(columns={'pc test': 'pc'})
    tuned_pc_avg_df['erry'] = tuned_pc_df.groupby(['model','area'])['pc test'].std().values/np.sqrt(var_df['stim'].nunique())
    tuned_pc_avg_df['cat'] = 1
    pc_df = pd.concat([inv_pc_avg_df, tuned_pc_avg_df], axis=0)

    #compute means and sems in each model
    plot_df = pc_df.groupby(['area','cat'])['pc'].mean().reset_index()
    plot_df['color'] = color_inv
    idx_tuned = plot_df[plot_df['cat']==1].index
    plot_df.loc[idx_tuned,'color'] = color_tuned
    plot_df['errcolor'] = color_err
    plot_df['y'] = plot_df['pc']
    plot_df['erry'] = pc_df.groupby(['area','cat'])['erry'].apply(lambda x: (x**2).sum()**0.5).values
    plot_df['erry'] = plot_df['erry']/np.sqrt(pc_df.groupby(['area','cat'])['pc'].count().values)
    plot_df['erry'] = plot_df['erry'].replace(np.nan, 0)
    plot_df['x'] = [area_to_x_dict[area] + color_to_dx_dict[color] for area, color in zip(plot_df['area'], plot_df['color'])]
    plot_df.sort_values('x', inplace=True)
    plot_df['errx'] = 0

    if plot_df.empty:
        print(f'No data found.')
        return
    fig = plot_points_w_errorbars(plot_df, y_dp=None, sz=8, height=height, width=width, alpha=.6)
    fig.update_layout(yaxis_title='Pearson correlation (1)')
    fig.update_xaxes(title_standoff=0, tickvals=list(area_to_x_dict.values()), ticktext=list(area_to_x_dict.keys()))
    fig.update_yaxes(title_standoff=0, range=[0,0.75])
    fig.write_image(results_folder / f'pcorr_{n_neurons}avg_{d}dim.pdf', scale=1, engine='kaleido')


def plot_equivariance_dim(model_df, params_df, project_root, height=150, width=680, \
                           color_inv='rgba(247, 148, 29, 1.0)', color_tuned='rgba(139, 94, 60, 1.0)', color_err='rgba(169, 169, 169, 1.0)'):
    """
    Plot dimension of the invariant and tuned subspaces across layers.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_inv : str, default='rgba(247, 148, 29, 1)'
        color for invariant dimensions
    color_tuned : str, default='rgba(139, 94, 60, 1)'
        color for tuned dimensions

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results' / 'dim_{n_neurons}avg_{d}dim.pdf'
    """
    
    n_neurons = int(params_df.loc['n_neurons'].value)
    d = int(params_df.loc['n_procs'].value)
    results_folder = project_root / 'results'
    results_folder.mkdir(parents=True, exist_ok=True)

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1, 'LM': 2, 'RL': 3, 'AL': 4, 'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9}
    #shift x-positions for invariant and tuned
    color_to_dx_dict = {f'{color_inv}': -0.1, f'{color_tuned}': 0.1}

    #collect results across layers
    dim_df = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        var_f = cache_folder / f'{layer_name}_var_dim_{n_neurons}avg_{d}dim.csv'
        if not var_f.exists():
            continue
        tmp_df = pd.read_csv(var_f)
        #find invariant and tuned dimensions
        tmp_df['dim'] = 0.0
        tmp_df['erry'] = 0.0
        tmp_df.loc[0,'dim'] = tmp_df.groupby('stim').apply(lambda x: (np.abs(x['evals'])<0.5).sum()).mean()
        tmp_df.loc[0,'erry'] = tmp_df.groupby('stim').apply(lambda x: (np.abs(x['evals'])<0.5).sum()).std()
        tmp_df.loc[0,'erry'] = tmp_df.loc[0,'erry']/np.sqrt(tmp_df['stim'].nunique())
        tmp_df.loc[1,'dim'] = tmp_df.groupby('stim').apply(lambda x: ((np.abs(x['evals'])>=0.5)&(np.abs(x['evals'])<1.5)).sum()).mean()
        tmp_df.loc[1,'erry'] = tmp_df.groupby('stim').apply(lambda x: ((np.abs(x['evals'])>=0.5)&(np.abs(x['evals'])<1.5)).sum()).std()
        tmp_df.loc[1,'erry'] = tmp_df.loc[1,'erry']/np.sqrt(tmp_df['stim'].nunique())
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        dim_df = pd.concat([dim_df, tmp_df.iloc[:2].reset_index(names='cat')], axis=0)  

    #find means and sems
    plot_df = dim_df.groupby(['area','cat'])['dim'].mean().reset_index()
    plot_df['color'] = color_inv
    idx_tuned = plot_df[plot_df['cat']==1].index
    plot_df.loc[idx_tuned,'color'] = color_tuned
    plot_df['errcolor'] = color_err
    plot_df['y'] = plot_df['dim']
    plot_df['erry'] = dim_df.groupby(['area','cat'])['erry'].apply(lambda x: (x**2).sum()**0.5).values
    plot_df['erry'] = plot_df['erry']/np.sqrt(dim_df.groupby(['area','cat'])['dim'].count().values)
    plot_df['erry'] = plot_df['erry'].replace(np.nan, 0)
    plot_df['x'] = [area_to_x_dict[area] + color_to_dx_dict[color] for area, color in zip(plot_df['area'], plot_df['color'])]
    plot_df.sort_values('x', inplace=True)
    plot_df['errx'] = 0

    if plot_df.empty:
        print(f'No data found.')
        return
    fig = plot_points_w_errorbars(plot_df, sz=8, height=height, width=width, alpha=.6)
    fig.update_layout(yaxis_title='dimension (1)')
    fig.update_xaxes(title_standoff=0, tickvals=list(area_to_x_dict.values()), ticktext=list(area_to_x_dict.keys()))
    fig.update_yaxes(title_standoff=0, range=[0,20], tickvals=[0,10,20], ticktext=['0','10','20'])
    fig.write_image(results_folder / f'dim_{n_neurons}avg_{d}dim.pdf', scale=1, engine='kaleido')


def plot_equivariance_ccnorm(model_df, params_df, project_root, height=150, width=680, \
                           color_pts='rgba(140, 140, 140, 1.0)', color_err='rgba(169, 169, 169, 1.0)'):
    """
    Plot normalized correlation coefficient across layers.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_pts : str, default='rgba(140, 140, 140, 1)'
        color for data points
    color_err : str, default='rgba(169, 169, 169, 1.0)'
        color for error bars

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results' / 'cc_{n_neurons}avg_{d}dim.pdf'
    """
   
    n_neurons = int(params_df.loc['n_neurons'].value)
    d = int(params_df.loc['n_procs'].value)
    results_folder = project_root / 'results'
    results_folder.mkdir(parents=True, exist_ok=True)

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1, 'LM': 2, 'RL': 3, 'AL': 4, 'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9}

    #collect results across layers
    performance_df = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        equi_f = cache_folder / f'{layer_name}_equivariance_{n_neurons}avg_{d}dim.csv'
        if not equi_f.exists():
            continue
        tmp_df = pd.read_csv(equi_f)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        performance_df = pd.concat([performance_df, tmp_df], axis=0)

    #compute mean and sem for each area
    plot_df = performance_df.groupby(['area'])[['cc_mean']].mean().reset_index()
    plot_df['y'] = plot_df['cc_mean']
    plot_df['erry'] = performance_df.groupby(['area'])['cc_mean'].std().reset_index()['cc_mean']
    plot_df['erry'] = plot_df['erry']/np.sqrt(performance_df.groupby(['area'])['cc_mean'].count().values)
    plot_df['erry'] = plot_df['erry'].replace(np.nan, 0)
    plot_df['x'] = [area_to_x_dict[area] for area in plot_df['area']]
    plot_df.sort_values('x', inplace=True)
    plot_df['errx'] = 0
    plot_df['color'] = color_pts
    plot_df['errcolor'] = color_err

    if plot_df.empty:
        print(f'No data found.')
        return
    fig = plot_points_w_errorbars(plot_df, y_dp=[0], sz=8, height=height, width=width, alpha=.6)
    fig.update_layout(yaxis_title='CCnorm (1)')
    fig.update_xaxes(title_standoff=0, tickvals=list(area_to_x_dict.values()), ticktext=list(area_to_x_dict.keys()))
    fig.update_yaxes(title_standoff=0, range=[-0.05,0.35], tickvals=[0,0.2], ticktext=['0','0.2'])
    fig.write_image(results_folder / f'cc_{n_neurons}avg_{d}dim.pdf', scale=1, engine='kaleido')  


def plot_example_projs(model_df, params_df, project_root, height=150, width=150, suffix='_data', \
                       color_test=['rgba(200, 50, 150, 1.0)', 'rgba(0, 174, 239, 1.0)']):
    """
    Plot example projections to all pairs of eigenspaces.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    height : int, default=150
        plot height in pixels
    width : int, default=150
        plot width in pixels
    suffix : str, default='_data'
        suffix for cache file identification
    color_test : list, default=['rgba(200, 50, 150, 1.0)', 'rgba(0, 174, 239, 1.0)']
        colors for test conditions

    Returns
    -------
    None
        saves plots as PDF in project_root / 'results' / 'example_projs' / '{model_name}_{layer_name}_example_projs{suffix}_{n_neurons}rand_{d}dim_{index}index.pdf'
    """

    n_neurons = int(params_df.loc['n_neurons'].value)
    d = int(params_df.loc['n_procs'].value)
    results_folder = project_root / 'results' / 'example_projs'
    results_folder.mkdir(parents=True, exist_ok=True)

    #loop over models & layers
    for i0 in range(model_df.shape[0]):
        
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name
        proj_df = pkl.load( open( cache_folder / f'{layer_name}_example_projs{suffix}_{n_neurons}rand_{d}dim.p', "rb" ) )
        xy_max = 1.1*proj_df[['x','y']].abs().max().max()
        for j in range(d//2):
          tmp_df = proj_df[proj_df['eigenspace_index']==2*j+1]

          plot_df = tmp_df[tmp_df['test']==0].copy()
          plot_df['color'] = color_test[0]
          plot_df['errcolor'] = color_test[0]
          fig = plot_points_w_errorbars(plot_df, lines=True, sz=4, highlight_first=True, alpha=.6)

          plot_df = tmp_df[tmp_df['test']==1].copy()
          plot_df['color'] = color_test[1]
          plot_df['errcolor'] = color_test[1]
          fig1 = plot_points_w_errorbars(plot_df, lines=True, sz=4, highlight_first=True, alpha=.6)
          fig.add_trace(fig1.data[0])
          fig.add_trace(fig1.data[1])
          
          fig.update_layout(height=height, width=width, xaxis_range=[-xy_max,xy_max], yaxis_range=[-xy_max,xy_max])
          fig.write_image(results_folder / f'{model_name}_{layer_name}_example_projs{suffix}_{n_neurons}rand_{d}dim_{2*j+1}index.pdf', scale=1, engine='kaleido')


def plot_example_evals_abs(model_df, params_df, project_root, height=150, width=200, \
                           color_pts='rgba(140, 140, 140, 1.0)', color_inv='rgba(247, 148, 29, 1.0)', color_tuned='rgba(139, 94, 60, 1.0)'):
    """
    Plot example absolute eigenphases.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    height : int, default=150
        plot height in pixels
    width : int, default=200
        plot width in pixels
    color_pts : str, default='rgba(140, 140, 140, 1)'
        default color for data points
    color_inv : str, default='rgba(247, 148, 29, 1)'
        color for invariant eigenvalues
    color_tuned : str, default='rgba(139, 94, 60, 1)'
        color for tuned eigenvalues

    Returns
    -------
    None
        saves plots as PDF files in project_root / 'results' / 'example_evals' / '{model_name}_{layer_name}_example_evals_abs_{n_neurons}rand_{d}dim.pdf'
    """
   
    n_neurons = int(params_df.loc['n_neurons'].value)
    d = int(params_df.loc['n_procs'].value)
    results_folder = project_root / 'results' / 'example_evals'
    results_folder.mkdir(parents=True, exist_ok=True)

    #loop over models & layers
    for i0 in range(model_df.shape[0]):
        
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        plot_df = pkl.load( open( cache_folder / f'{layer_name}_example_evals_abs_{n_neurons}rand_{d}dim.p', "rb" ) )
        plot_df['color'] = color_pts
        idx_inv = plot_df[plot_df['y'].values < .5].index
        plot_df.loc[idx_inv,'color'] = color_inv
        idx_tuned = plot_df[(plot_df['y'].values >= .5)&(plot_df['y'].values < 1.5)].index
        plot_df.loc[idx_tuned,'color'] = color_tuned

        if plot_df.empty:
            continue
        fig = plot_eval(plot_df, y_max=6, sz=6, height=height, width=width)
        fig.write_image(results_folder / f'{model_name}_{layer_name}_example_evals_abs_{n_neurons}rand_{d}dim.pdf', scale=1, engine='kaleido')


def plot_orientation_errors(model_df, params_df, project_root, height=150, width=680, 
                            color_pts='rgba(140, 140, 140, 1.0)', color_err='rgba(169, 169, 169, 1.0)',
                            equivariant=False):
    """
    Plot orientation decoding error across layers.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_pts : str, default='rgba(140, 140, 140, 1)'
        color for data points
    color_err : str, default='rgba(169, 169, 169, 1.0)'
        color for error bars
    equivariant : bool, default=False
        whether to use equivariant analysis

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results' / 'ori_{n_neurons}avg{suffix}.pdf'
    """

    n_neurons = int(params_df.loc['n_neurons'].value)
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
    else:
        suffix = ''
    results_folder = project_root / 'results'
    results_folder.mkdir(parents=True, exist_ok=True)

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1, 'LM': 2, 'RL': 3, 'AL': 4, 'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9}

    #collect results across layers
    results_df = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        ori_f = cache_folder / f'{layer_name}_ori2x_decoder_{n_neurons}avg{suffix}.csv'
        if not ori_f.exists():
            continue
        tmp_df = pd.read_csv(ori_f)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        tmp_df['res'] = np.abs(tmp_df['res'])
        results_df = pd.concat([results_df, tmp_df], axis=0)

    compute_significance(results_df.rename(columns={'res': 'score'}), results_folder / f'ori_significance{suffix}.csv')

    #compute mean and sem for each area
    plot_df = results_df.groupby(['area'])['res'].mean().reset_index()
    plot_df['y'] = plot_df['res']
    plot_df['erry'] = results_df.groupby(['area'])['res'].std().reset_index()['res']
    plot_df['erry'] = plot_df['erry']/np.sqrt(results_df.groupby(['area'])['res'].count().values)
    plot_df['x'] = [area_to_x_dict[area] for area in plot_df['area']]
    plot_df.sort_values('x', inplace=True)
    plot_df['errx'] = 0
    plot_df['color'] = color_pts
    plot_df['errcolor'] = color_err

    if plot_df.empty:
        print(f'No data found.')
        return
    fig = plot_points_w_errorbars(plot_df, y_dp=[90], sz=8, height=height, width=width)
    fig.update_layout(yaxis_title='absolute angle estimation error (deg)')
    fig.update_xaxes(title_standoff=0, tickvals=list(area_to_x_dict.values()), ticktext=list(area_to_x_dict.keys()))
    fig.update_yaxes(title_standoff=0, range=[40,95], tickvals=[0,45,90,180], ticktext=['0 ','45 ','90 ','180 '])
    fig.write_image(results_folder / f'ori_{n_neurons}avg{suffix}.pdf', scale=1, engine='kaleido')


def plot_classification_errors(model_df, params_df, project_root, C0=0.001, height=150, width=680, 
                               color_pts='rgba(140, 140, 140, 1.0)', color_err='rgba(169, 169, 169, 1.0)',
                               equivariant=False):
    """
    Plot classification error across layers.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    C0 : float, default=0.01
        SVM regularization parameter
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_pts : str, default='rgba(140, 140, 140, 1)'
        color for data points
    color_err : str, default='rgba(169, 169, 169, 1.0)'
        color for error bars
    equivariant : bool, default=False
        whether to use equivariant analysis

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results' / 'svm_{C0}C_{n_neurons}avg{suffix}.pdf'
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
    else:
        suffix = ''
    results_folder = project_root / 'results'
    results_folder.mkdir(parents=True, exist_ok=True)

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1, 'LM': 2, 'RL': 3, 'AL': 4, 'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9}

    #collect results across layers
    svm_df = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        class_f = cache_folder / f'{layer_name}_class2x_decoder_{n_neurons}avg_{C0}C{suffix}.csv'
        if not class_f.exists():
          print(f'No classification data for {model_name} {layer_name} with {n_neurons} neurons and C={C0}')
          continue
        tmp_df = pd.read_csv(class_f)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        svm_df = pd.concat([svm_df, tmp_df], axis=0)

    results_df = svm_df[['model', 'area', 'acc test']].copy()
    results_df['score'] = 1 - results_df['acc test']
    compute_significance(results_df, results_folder / f'svm_significance{suffix}.csv')

    #compute mean and sem for each area
    svm_plot_df = svm_df.groupby(['area'])['acc test'].mean().reset_index()
    svm_plot_df['y'] = 1-svm_plot_df['acc test']
    svm_plot_df['erry'] = svm_df.groupby(['area'])['acc test'].std().reset_index()['acc test']
    svm_plot_df['erry'] = svm_plot_df['erry']/np.sqrt(svm_df.groupby(['area'])['acc test'].count().values)
    svm_plot_df['x'] = [area_to_x_dict[area] for area in svm_plot_df['area']]
    svm_plot_df.sort_values('x', inplace=True)
    svm_plot_df['errx'] = 0
    svm_plot_df['color'] = color_pts
    svm_plot_df['errcolor'] = color_err

    if svm_plot_df.empty:
        print(f'No data found.')
        return
    fig = plot_points_w_errorbars(svm_plot_df, y_dp=[1-1/91], sz=8, height=height, width=width, alpha=.6)
    fig.update_layout(yaxis_title='decoding error (1)')
    fig.update_xaxes(title_standoff=0, tickvals=list(area_to_x_dict.values()), ticktext=list(area_to_x_dict.keys()))
    fig.update_yaxes(title_standoff=0, range=[0.65,1.03], tickvals=[0,0.5,0.7,1], ticktext=['0','0.5','0.7','1'])
    fig.write_image(results_folder / f'svm_{C0}C_{n_neurons}avg{suffix}.pdf', scale=1, engine='kaleido')


def plot_summary_all(model_df, params_df, project_root, height=120, width=110):
    """
    Plot summary statistics for all mouse models, including CC half, d prime, and OSI.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (corr_thre)
    project_root : Path
        root directory of the project
    height : int, default=200
        plot height in pixels
    width : int, default=200
        plot width in pixels
        
    Returns
    -------
    None
        saves plots as PDF files in project_root / 'results' / 'mouse_summary' / '{model_name}_reliable_{corr_thre}thre_CChalf.pdf',
        '{model_name}_reliable_{corr_thre}thre_dp.pdf', and '{model_name}_reliable_{corr_thre}thre_OSI.pdf'
    """
    #get parameters
    corr_thre = float(params_df.loc['corr_thre'].value)
    mouse_models = [model for model in model_df.network.unique() if model not in ['pixels', 'alexnet']]
    model_df = model_df[model_df['network'].isin(mouse_models)].reset_index(drop=True)

    #define folders
    results_folder = project_root / 'results' / 'mouse_summary'
    results_folder.mkdir(parents=True, exist_ok=True)

    #loop over mouse recordings
    mouse_i0 = model_df.groupby('network').apply(lambda x: x.index[0]).values
    for i0 in mouse_i0:        
        model_name = model_df['network'].values[i0]
        cache_folder = project_root / 'cache' / model_name
        cache_folder.mkdir(parents=True, exist_ok=True)

        sum_df = pkl.load( open( cache_folder / f'reliable_{corr_thre}thre_summary.p', "rb" ) )
        if sum_df.empty:
            continue

        CC_df = sum_df.rename(columns={'area': 'x', 'CC_test': 'y2', 'CC_train': 'y1'})
        fig = plot_v_boxes(CC_df, height=height, width=width)
        fig.update_layout(yaxis_title='CC half', yaxis_range=[-.21,1])
        fig.update_yaxes(title_standoff=0)
        fig.write_image(results_folder / f'{model_name}_reliable_{corr_thre}thre_CChalf.pdf', scale=1, engine='kaleido')

        CC_df = sum_df.rename(columns={'area': 'x', 'dp_test': 'y2', 'dp_train': 'y1'})
        fig = plot_v_boxes(CC_df, height=height, width=width)
        fig.update_layout(yaxis_title='d prime', yaxis_range=[-0.05,3.40])
        fig.update_yaxes(title_standoff=0)
        fig.write_image(results_folder / f'{model_name}_reliable_{corr_thre}thre_dp.pdf', scale=1, engine='kaleido')

        CC_df = sum_df.rename(columns={'area': 'x', 'OSI_test': 'y2', 'OSI_train': 'y1'})
        fig = plot_v_boxes(CC_df, height=height, width=width)
        fig.update_layout(yaxis_title='OSI', yaxis_range=[-0.05,1.05])
        fig.update_yaxes(title_standoff=0)
        fig.write_image(results_folder / f'{model_name}_reliable_{corr_thre}thre_OSI.pdf', scale=1, engine='kaleido')


def plot_v_boxes(df_plot, height=150, width=100):
    """
    Plot vertical boxes for two sets of data (e.g. train and test).

    Parameters
    ----------
    df_plot : pd.DataFrame
        dataframe containing 'x', 'y1', and 'y2' columns for plotting
    height : int, default=150
        plot height in pixels
    width : int, default=100
        plot width in pixels

    Returns
    -------
    fig : plotly.graph_objects.Figure
        figure object containing the box plots
    """
    fig=go.Figure()
    fig.add_trace(
        go.Box(
            x=df_plot['x']
            , y=df_plot['y1']
            , fillcolor='rgba(128, 128, 128, .5)'
            , line={'color': 'black', 'width': 0.5}
            , marker={"size": 1}
            , showlegend=False
        ))
    fig.add_trace(
        go.Box(
            x=df_plot['x']
            , y=df_plot['y2']
            , fillcolor='rgba(128, 128, 128, .5)'
            , line={'color': 'black', 'width': 0.5}
            , marker={"size": 1}
            , showlegend=False
        ))
    
    fig.update_layout(
            height=height
        , width=width
        , paper_bgcolor='rgba(255,255,255,255)'
        , plot_bgcolor='rgba(255,255,255,0)'
        , boxmode='group' 
        , boxgroupgap=0.2
        , boxgap=0.1
        , font={'size': 11, 'family': 'arial', 'color': 'black'}
        , autosize=False
        , margin=dict(
            l=20,
            r=20,
            b=20,
            t=20,
            pad=0
        )      
    )
    fig.update_xaxes(
            showline=True
        , showticklabels = True
        , showgrid=False
        , linewidth=.5
        , linecolor='black'
        , tickfont={
                'size': 10
            , 'family': 'arial'
            , 'color': 'black'
            }
        , ticks='outside'
        , ticklen=2
        , tickwidth=.5
        , tickangle=0
        )
    fig.update_yaxes(
            showline=True
        , showticklabels = True
        , showgrid=False
        , linewidth=.5
        , linecolor='black'
        , tickfont={
                'size': 10
            , 'family': 'arial'
            , 'color': 'black'
            }
        , ticks='outside'
        , ticklen=2
        , tickwidth=.5
    )

    return fig


def plot_neurons_all(model_df, params_df, project_root, height=200, width=200):
    """
    Plot neuron positions for all mouse models, highlighting reliably responsive neurons and area boundaries.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (corr_thre)
    project_root : Path
        root directory of the project
    height : int, default=200
        plot height in pixels
    width : int, default=200
        plot width in pixels
        
    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results' / 'mouse_summary' / '{model_name}' / 'reliable_{corr_thre}thre_positions.pdf'
    """
    #get parameters
    corr_thre = float(params_df.loc['corr_thre'].value)
    mouse_models = [model for model in model_df.network.unique() if model not in ['pixels', 'alexnet']]
    model_df = model_df[model_df['network'].isin(mouse_models)].reset_index(drop=True)

    #define folders
    retinotopy_root = project_root / 'data' / 'retinotopy'
    results_folder = project_root / 'results' / 'mouse_summary'
    results_folder.mkdir(parents=True, exist_ok=True)

    #retinotopy area boundaries, defined in a standard mouse
    boundaries = np.load( retinotopy_root / 'areas.npz', allow_pickle = True)['out']
    #relative shift in boundaries between mice
    shift_dict = {'TX108_05_31_2023': [-120,160], 'TX109_05_30_2023': [-490, 70], 'TX111_05_18_2023': [-80,300]}
    #check that mouse models are part of shift_dict
    for model in mouse_models:
        if model not in shift_dict:
            raise ValueError(f'No retinotopy shift defined for {model}. Please add it to shift_dict.')

    #loop over mouse recordings
    mouse_i0 = model_df.groupby('network').apply(lambda x: x.index[0]).values
    for i0 in mouse_i0:        
        model_name = model_df['network'].values[i0]
        cache_folder = project_root / 'cache' / model_name
        cache_folder.mkdir(parents=True, exist_ok=True)

        reliable_pos_df = pkl.load( open( cache_folder / f'reliable_{corr_thre}thre_pos.p', "rb" ) )
        if reliable_pos_df.empty:
            continue

        shift = shift_dict[model_name]
        idx_resp = np.where(reliable_pos_df['resp'])[0]
        fig = plot_neurons(reliable_pos_df[['x','y']].values, idx_resp, boundaries, shift, height=height, width=width)
        fig.write_image(results_folder / f'{model_name}_reliable_{corr_thre}thre_positions.pdf', scale=1, engine='kaleido')


def plot_neurons(xy_pos, idx_resp, boundaries, shift, width=200, height=200):
    """
    Plot neuron positions, highlighting reliably responsive neurons and area boundaries.

    Parameters
    ----------
    xy_pos : np.ndarray
        neuron positions of shape (n_neurons, 2)
    idx_resp : np.ndarray
        indices of responsive neurons
    boundaries : list
        boundary coordinates for visual areas
    shift : tuple
        coordinate shift values
    width : int, default=200
        plot width in pixels
    height : int, default=200
        plot height in pixels

    Returns
    -------
    fig : plotly.graph_objects.Figure
        plotly figure object
    """

    fig = go.Figure()
    # Scatter plot for all neurons
    fig.add_trace(go.Scatter(
        x=xy_pos[:, 0]
        , y=xy_pos[:, 1]
        , mode='markers'
        , marker=dict(size=1, color='lightgray', opacity=0.1)
        , showlegend=False
    ))
    # Scatter plot for responsive neurons
    fig.add_trace(go.Scatter(
        x=xy_pos[idx_resp, 0]
        , y=xy_pos[idx_resp, 1]
        , mode='markers'
        , marker=dict(size=1, color='darkgray', opacity=0.8)
        , showlegend=False
    ))
    # Plot area boundaries
    for j in range(len(boundaries)):
        fig.add_trace(go.Scatter(
            x=boundaries[j][:, 1] + shift[1]
            , y=boundaries[j][:, 0] + shift[0]
            , mode='lines'
            , line=dict(color='black', width=1)
            , opacity=0.4
            , showlegend=False
        ))
    fig.update_layout(
        title=f'Responsive/total neurons: {idx_resp.shape[0]}/{xy_pos.shape[0]}'
        , title_font={'size': 11, 'family': 'arial', 'color': 'black'}
        , xaxis_title=''
        , yaxis_title=''
        , paper_bgcolor='rgba(255,255,255,255)'
        , plot_bgcolor='rgba(255,255,255,0)'
        , font={'size': 11, 'family': 'arial', 'color': 'black'}
        , margin=dict(
            l=20,
            r=20,
            b=20,
            t=20,
            pad=0
        )
        , width=width
        , height=height
    )
    fig.update_xaxes(
        showline=False
        , showticklabels = False
        , showgrid=False
        )
    fig.update_yaxes(
        showline=False
        , showticklabels = False
        , showgrid=False
    )   

    return fig


def plot_eval(df_plot, y_max=7, sz=10, height=250, width=400):
    """
    Plot eigenvalue scatter plots with horizontal grid lines.

    Parameters
    ----------
    df_plot : pd.DataFrame
        dataframe containing x, y coordinates and color information
    y_max : int, default=7
        maximum y-axis value for grid lines
    sz : int, default=10
        marker size for scatter points
    height : int, default=250
        plot height in pixels
    width : int, default=400
        plot width in pixels

    Returns
    -------
    fig : plotly.graph_objects.Figure
        plotly figure object
    """

    fig=go.Figure()
    for i in range(y_max+1):
        fig.add_hline(y=i, line_width=1, line_color="lightgray", opacity=.8, line_dash='dot' )

    fig.add_trace(
        go.Scatter(
            x=df_plot['x']
          , y=df_plot['y']
          , mode='markers'
          , marker={"size": sz, 'color': df_plot['color']}
          , showlegend=False
        ))
    
    fig.update_layout(
          xaxis_range=[df_plot['x'].min()-1,df_plot['x'].max()+1]
        , yaxis_range=[-0.5,y_max+0.5]
        , height=height
        , width=width
        , paper_bgcolor='rgba(255,255,255,255)'
        , plot_bgcolor='rgba(255,255,255,0)'
        , xaxis_title='eigenvalue index i'    
        , yaxis_title='eigenphase/xx' #r'$|\Delta \theta/\Delta \phi|$'    
        , font={'size': 11, 'family': 'arial', 'color': 'black'}
        , autosize=False
        , margin=dict(
              l=20,
              r=20,
              b=20,
              t=20,
              pad=0
          )
    )
    fig.update_xaxes(
          showline=True
        , showticklabels = True
        , showgrid=False
        , linewidth=.5
        , linecolor='black'
        , tickfont={
              'size': 10
            , 'family': 'arial'
            , 'color': 'black'
          }
        , ticks='outside'
        , ticklen=2
        , tickwidth=.5
        , tickangle=0
        , mirror=True
        , tickvals=df_plot['x'].values[::2]
        , ticktext=[str(i) for i in df_plot['x'].values[::2]]
        , title_standoff=0
      )
    fig.update_yaxes(
          showline=True
        , showticklabels = True
        , showgrid=False
        , linewidth=.5
        , linecolor='black'
        , tickfont={
              'size': 10
            , 'family': 'arial'
            , 'color': 'black'
          }
        , ticks='outside'
        , ticklen=2
        , tickwidth=.5
        , mirror=True
        , tickvals=np.array(range(y_max+1))
        , ticktext=[str(i) for i in range(y_max+1)]
        , title_standoff=0
    )

    return fig


def plot_points_w_errorbars(df_plot, lines=False, y_dp=None, symb='circle', sz=10, height=250, width=400, \
                            highlight_first=False, highlight_last=False, alpha=0.2):
    """
    Plot bar plots and optional line connections.

    Parameters
    ----------
    df_plot : pd.DataFrame
        dataframe containing x, y, errx, erry, color, and errcolor columns
    lines : bool, default=False
        whether to connect points with lines
    y_dp : list or None, default=None
        y-coordinates for horizontal dotted reference lines
    symb : str, default='circle'
        marker symbol for scatter points
    sz : int, default=10
        marker size for scatter points
    height : int, default=250
        plot height in pixels
    width : int, default=400
        plot width in pixels
    highlight_first : bool, default=False
        whether to highlight the first point
    highlight_last : bool, default=False
        whether to highlight the last point
    alpha : float, default=0.2
        transparency for lines and error bars

    Returns
    -------
    fig : plotly.graph_objects.Figure
        plotly figure object
    """
    fig=go.Figure()
    if y_dp:
        for y_i in y_dp:
          fig.add_hline(y=y_i, line_width=1, line_color="lightgray", opacity=.8, line_dash='dot' )

    if lines:
      fig.add_trace(
          go.Scatter(
              x=df_plot['x']
            , y=df_plot['y']
            , error_x=dict(type='data', array=df_plot['errx'], width=2, color=df_plot['errcolor'].values[0][:-4]+f'{alpha})') 
            , error_y=dict(type='data', array=df_plot['erry'], width=2, color=df_plot['errcolor'].values[0][:-4]+f'{alpha})') 
            , mode='markers+lines'
            , marker={"size": sz, 'color': df_plot['color'], 'symbol': symb}
            , line={'color': df_plot['color'].values[0][:-4]+f'{alpha})', 'width': 1}
            , showlegend=False
          ))
    else:
      fig.add_trace(
          go.Scatter(
              x=df_plot['x']
            , y=df_plot['y']
            , error_x=dict(type='data', array=df_plot['errx'], width=2, color=df_plot['errcolor'].values[0][:-4]+f'{alpha})') 
            , error_y=dict(type='data', array=df_plot['erry'], width=2, color=df_plot['errcolor'].values[0][:-4]+f'{alpha})') 
            , mode='markers'
            , marker={"size": sz, 'color': df_plot['color'], 'symbol': symb}
            , showlegend=False
          ))
    if highlight_first:
      fig.add_trace(
          go.Scatter(
              x=df_plot['x'][:1]
              , y=df_plot['y'][:1]
              , mode='markers'
              , marker={'color': df_plot['color'].values, 'size': sz+1, 'symbol': symb, 'line': {'color': 'black', 'width': 0.5}}
              , showlegend=False
          ))
    if highlight_last:
      fig.add_trace(
          go.Scatter(
              x=df_plot['x'][-1:]
              , y=df_plot['y'][-1:]
              , mode='markers'
              , marker={'color': df_plot['color'].values, 'size': sz+1, 'symbol': symb, 'line': {'color': 'black', 'width': 0.5}}
              , showlegend=False
          ))
        
    fig.update_layout(
          height=height
        , width=width
        , paper_bgcolor='rgba(255,255,255,255)'
        , plot_bgcolor='rgba(255,255,255,0)'
        , font={'size': 11, 'family': 'arial', 'color': 'black'}
        , autosize=False
        , margin=dict(
              l=20,
              r=20,
              b=20,
              t=20,
              pad=0
          )
    )
    fig.update_xaxes(
          showline=True
        , showticklabels = True
        , showgrid=False
        , linewidth=.5
        , linecolor='black'
        , tickfont={
                'size': 10
            , 'family': 'arial'
            , 'color': 'black'
          }
        , ticks='outside'
        , ticklen=2
        , tickwidth=.5
        , tickangle=0
        , mirror=True
      )
    fig.update_yaxes(
          showline=True
        , showticklabels = True
        , showgrid=False
        , linewidth=.5
        , linecolor='black'
        , tickfont={
              'size': 10
            , 'family': 'arial'
            , 'color': 'black'
          }
        , ticks='outside'
        , ticklen=2
        , tickwidth=.5
        , mirror=True
    )

    return fig


def plot_classes_all(params_df, project_root):
    """
    Load and display all stimulus classes.
    
    Parameters
    ----------
    params_df : pd.DataFrame
        DataFrame containing analysis parameters with 'stimulus' column
    project_root : Path
        Root path to the project directory
        
    Returns
    -------
    None
        Saves the plot as a PDF file in project_root / 'results' / '{stim_name}_all_classes.pdf'
    """
    # Get stimulus parameters
    stim_name = params_df.loc['stimulus'].values[0]
    stim_root = project_root / 'data' / 'stimuli'

    results_folder = project_root / 'results'
    results_folder.mkdir(parents=True, exist_ok=True)

    # Load all images
    img_all = load_img_sorted(stim_root, stim_name) 
        
    fig = make_subplots(
        rows=7, cols=15,
        horizontal_spacing=0.01,
        vertical_spacing=0.02
    )
    
    # Training images
    for row in range(1, 7): 
        for col in range(1, 16): 
            img_idx = (row - 1) * 15 * 15 + (col - 1) * 15
            fig.add_trace(
                go.Heatmap(
                    z=img_all[img_idx],
                    colorscale='gray',
                    showscale=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            fig.update_yaxes(autorange="reversed", row=row, col=col)
    
    # Testing images
    for col, img_idx in enumerate([15*90, 15*91], start=1):
        fig.add_trace(
            go.Heatmap(
                z=img_all[img_idx],
                colorscale='gray',
                showscale=False,
                hoverinfo='skip'
            ),
            row=7, col=col
        )
        fig.update_yaxes(autorange="reversed", row=row, col=col)
    
    fig.update_layout(
        title=f"All Stimulus Classes in ({stim_name})",
        height=1400, 
        width=3000,  
        showlegend=False
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.write_image(results_folder / f'{stim_name}_all_classes.pdf', scale=1, engine='kaleido')


def plot_example_images(params_df, project_root):
    """
    Load and display example stimulus images.
    
    Parameters
    ----------
    params_df : pd.DataFrame
        DataFrame containing analysis parameters with 'stimulus' column
    project_root : Path
        Root path to the project directory
        
    Returns
    -------
    None
        Saves the plot as a PDF file in project_root / 'results' / '{stim_name}_example_images.pdf'
    """
    # Get stimulus parameters
    stim_name = params_df.loc['stimulus'].values[0]
    stim_root = project_root / 'data' / 'stimuli'

    results_folder = project_root / 'results'
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Load all images
    img_all = load_img_sorted(stim_root, stim_name) 
    
    fig = make_subplots(
        rows=5, cols=6,
        horizontal_spacing=0.02,
        vertical_spacing=0.02
    )
    
    # Define image indices for each row
    row1_images = [0, 1, 2, 3, None, 14]
    row2_images = [15*74+0, 15*74+1, 15*74+2, 15*74+3, None, 15*74+14]
    row3_images = [None, None, None, None, None, None]
    row4_images = [15*90+0, 15*90+1, 15*90+2, 15*90+3, None, 15*90+14]
    row5_images = [15*91+0, 15*91+1, 15*91+2, 15*91+3, None, 15*91+14]
    
    # Add images to subplots
    all_rows = [row1_images, row2_images, row3_images, row4_images, row5_images]
    for row_idx, row_images in enumerate(all_rows, start=1):
        for col_idx, img_idx in enumerate(row_images, start=1):
            if img_idx is not None and img_idx < len(img_all):
                fig.add_trace(
                    go.Heatmap(
                        z=img_all[img_idx],
                        colorscale='gray',
                        showscale=False,
                        hoverinfo='skip'
                    ),
                    row=row_idx, col=col_idx
                )
                fig.update_yaxes(autorange="reversed", row=row_idx, col=col_idx)
    
    fig.update_layout(
        title=f"Example Stimulus Images ({stim_name})",
        height=1000, 
        width=1200, 
        showlegend=False
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.write_image(results_folder / f'{stim_name}_example_images.pdf', scale=1, engine='kaleido')
