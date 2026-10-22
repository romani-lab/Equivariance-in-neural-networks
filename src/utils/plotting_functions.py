import os
import numpy as np
import pickle as pkl
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots

from utils.helper import compute_significance, load_var_dim, band_summary, band_summary_per_stim, band_summary_table
from utils.stimulus_loader import load_img_sorted, find_n_imgs


MODEL_COLORS = {
    'alexnet':                 'rgba(100, 100, 100, 1)',
    'random_alexnet':          'rgba(180, 180, 180, 1)',
    'alt_alexnet_rot-aug':     'rgba(44, 160, 44, 1)',
    'alt_alexnet_ori':         'rgba(227, 119, 194, 1)',
    'alt_alexnet_multi-task':  'rgba(214, 39, 40, 1)',
    # AlexNet-style CNNs with built-in rotation equivariance (SI Fig 6)
    'eqv-c15':                 'rgba(227, 119, 194, 1)',   # exact C_15 symmetry in every layer
    'eqv-c15r':                'rgba(214, 39, 40, 1)',     # symmetry relaxed across layers (C15 > C5 > C1)
    'eqv-c1':                  'rgba(0, 0, 0, 1)',         # no built-in symmetry (C1)
}


def _resolve_results_folder(project_root, results_folder, default_name):
    """
    Resolve and create the output folder for plot files.

    Parameters
    ----------
    project_root : Path
        root directory of the project
    results_folder : str or Path or None
        explicit output folder, or None to use project_root / default_name
    default_name : str
        default output folder name under project_root

    Returns
    -------
    folder : Path
        resolved output folder
    """
    folder = project_root / default_name if results_folder is None else Path(results_folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def model_to_color(model_name):
    """
    Return the standard color for a model name.

    Parameters
    ----------
    model_name : str
        model name to map to a color

    Returns
    -------
    color : str
        rgba color string for the model
    """
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    return MODEL_COLORS['alexnet']


def plot_models_overlay(models, layers, get_y_err_fn, ylabel, height=120, width=320,
                        yrange=None, y_dp=None, save_path=None, alpha=0.4, style_fn=None,
                        colors=None, get_y_err_fn_sub=None, dx=0.0):
    """Plot each model as colored markers across layers (conv1-conv5).

    Markers only (no connecting lines), at marker transparency ``alpha`` -- matching the
    equivariant-CNN rotation-robustness panel (plot_rotation_robustness, alpha=0.4).

    With ``get_y_err_fn_sub`` a second series per model is overlaid at the same x: the primary
    series (e.g. the decoder in the full space) becomes ``alpha``-transparent diamonds and the
    second (e.g. the same decoder in the estimated equivariant subspace) opaque circles drawn on
    top, both in the model colour -- the full-space-vs-subspace convention of the simulation
    panels (plot_simulation._overlay).

    Parameters
    ----------
    models : array-like
        List of model names.
    layers : array-like
        List of layer names for x-axis.
    get_y_err_fn : callable
        Function(model, layer) -> (y_value, error_value).
    ylabel : str
        Label for y-axis.
    height : int
        Plot height in pixels.
    width : int
        Plot width in pixels.
    yrange : list or None
        [ymin, ymax] for y-axis.
    y_dp : list or None
        Y-coordinates for horizontal dotted reference lines.
    save_path : str or Path or None
        If given, save figure to this path.
    alpha : float
        Marker transparency.
    style_fn : callable or None
        Optional Function(model) -> (marker_symbol, line_dash) to distinguish models
        that share a colour (e.g. a square marker for random-init variants). Only the
        marker symbol is used here (lines are not drawn); line_dash is ignored. If None,
        every model is drawn with circle markers.
    colors : dict or None
        Optional {model: 'rgba(...)'} overriding the built-in model_to_color palette
        (used by the equivariant-CNN overlays: eqv-c15 pink / eqv-c15r red / eqv-c1 black).
        Models absent from the dict fall back to model_to_color.
    get_y_err_fn_sub : callable or None
        Optional second series, Function(model, layer) -> (y_value, error_value). When given,
        ``get_y_err_fn`` is drawn as ``alpha``-transparent diamonds (style_fn symbols are not
        applied) and this series as opaque circles on top. Missing values (nan) are skipped.
    dx : float
        Horizontal dodge between models (layer spacing = 1): model i of ``models`` is drawn at
        x + (i - (n_models - 1) / 2) * dx, so the order of ``models`` is the left-to-right
        order within each layer; both series of a model share its offset. 0 = no dodge.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    #(extractor, marker symbol or None = style_fn/circle, marker alpha, marker size); with a
    #second series the primary is drawn first (diamonds), then the circles of every model on top.
    #single-series overlays (dim/pcorr/var/cc by_model) use size 6 to match the decoding panels'
    #primary diamonds.
    if get_y_err_fn_sub is None:
        series = [(get_y_err_fn, None, alpha, 6)]
    else:
        series = [(get_y_err_fn, 'diamond', alpha, 6), (get_y_err_fn_sub, 'circle', 1.0, 5)]
    offsets = {m: (i - (len(models) - 1) / 2) * dx for i, m in enumerate(models)}
    first = True
    for fn, symbol, a, sz in series:
        for model in models:
            ys, errs = [], []
            for layer in layers:
                y, err = fn(model, layer)
                ys.append(y)
                errs.append(err)

            c = colors.get(model, model_to_color(model)) if colors is not None else model_to_color(model)
            c_alpha = c[:-2] + str(a) + ')'
            df_i = pd.DataFrame({
                'x': np.arange(len(layers)) + offsets[model], 'y': ys,
                'errx': 0.0, 'erry': errs,
                'color': c, 'errcolor': c,
            })

            fig_i = plot_points_w_errorbars(df_i, lines=False, sz=sz, height=height, width=width, alpha=alpha)
            fig_i.data[0].marker.color = c_alpha
            if symbol is not None:
                fig_i.data[0].marker.symbol = symbol
            elif style_fn is not None:
                symbol_m, _dash = style_fn(model)
                fig_i.data[0].marker.symbol = symbol_m

            if first:
                fig = fig_i
                first = False
            else:
                fig.add_trace(fig_i.data[0])

    fig.update_layout(yaxis_title=ylabel)
    fig.update_yaxes(title_standoff=0)
    if yrange is not None:
        fig.update_yaxes(range=yrange)
    fig.update_xaxes(tickvals=np.arange(len(layers)), ticktext=list(layers))
    if y_dp:
        for y_i in y_dp:
            fig.add_hline(y=y_i, line_width=1, line_color="lightgray", opacity=.8, line_dash='dot')
    if save_path:
        fig.write_image(str(save_path), scale=1, engine='kaleido')
    fig.show()
    return fig


def make_df_extractor(df, model_col, layer_col, value_col, transform=None):
    """Create a (model, layer) -> (y, err) extractor function for plot_models_overlay.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    model_col : str
        Column name for model filtering.
    layer_col : str
        Column name for layer filtering.
    value_col : str
        Column name for the value to aggregate.
    transform : callable or None
        Optional transform applied to mean value, e.g. lambda x: 1-x for error.

    Returns
    -------
    _extract : callable
        Function(model, layer) -> (y_mean, y_sem).
    """
    if transform is None:
        transform = lambda x: x

    def _extract(model, layer):
        """
        Extract the aggregated y value and SEM for one model and layer.

        Parameters
        ----------
        model : str
            model name to select
        layer : str
            layer name to select

        Returns
        -------
        y : float
            transformed mean value, or np.nan if no data are found
        err : float
            standard error of the mean, or 0 if no data are found
        """
        data = df[(df[model_col] == model) & (df[layer_col] == layer)][value_col]
        if len(data) == 0:
            return np.nan, 0
        return transform(data.mean()), data.std() / np.sqrt(len(data))
    return _extract


def make_row_extractor(df, name_col, cat_col, cat_val, value_col, err_col, scale=1.0,
                       layer_col=None):
    """Create a (model, layer) -> (y, err) extractor for pre-aggregated DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame with one row per (model+layer, category).
    name_col : str
        Column containing model identifiers. If layer_col is None, expects
        '{model}_{layer}' format in this column.
    cat_col : str
        Column for category filtering (e.g. 'cat').
    cat_val : int or str
        Category value to select.
    value_col : str
        Column name for the y value.
    err_col : str
        Column name for the error value.
    scale : float
        Multiplier for both value and error (e.g. 100 for percentage).
    layer_col : str or None
        If given, filter by model and layer columns separately instead of
        using a combined '{model}_{layer}' name.

    Returns
    -------
    _extract : callable
        Function(model, layer) -> (y, err).
    """
    def _extract(model, layer):
        """
        Extract the stored y value and error for one model and layer.

        Parameters
        ----------
        model : str
            model name to select
        layer : str
            layer name to select

        Returns
        -------
        y : float
            scaled value, or np.nan if no row is found
        err : float
            scaled error value, or 0 if no row is found
        """
        if layer_col is not None:
            row = df[(df[name_col] == model) & (df[layer_col] == layer) & (df[cat_col] == cat_val)]
        else:
            name = f'{model}_{layer}'
            row = df[(df[name_col] == name) & (df[cat_col] == cat_val)]
        if row.empty:
            return np.nan, 0
        return scale * row[value_col].values[0], scale * row[err_col].values[0]
    return _extract


def make_robustness_df(angles, mean_vals, sem_vals, color):
    """Build DataFrame for plot_points_w_errorbars from robustness results.

    Parameters
    ----------
    angles : array-like
        X-axis values (rotation angles).
    mean_vals : array-like
        Y-axis values (mean error or accuracy).
    sem_vals : array-like
        Standard error of mean.
    color : str
        RGBA color string.

    Returns
    -------
    pd.DataFrame
    """
    return pd.DataFrame({
        'x': angles,
        'y': mean_vals,
        'errx': np.zeros(len(angles)),
        'erry': sem_vals,
        'color': color,
        'errcolor': color,
    })


def plot_relcenter_all(model_df, params_df, project_root, ymax=1.3, height=150, width=680, \
                      color_pts='rgba(140, 140, 140, 1)', color_err='rgba(169, 169, 169, 1.0)', show_mice=True):
    """
    Plot relative distance between means across layers.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    ymax : float, default=1.3
        maximum value for y-axis
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_pts : str, default='rgba(140, 140, 140, 1)'
        color for points
    color_err : str, default='rgba(169, 169, 169, 1.0)'
        color for error bars
    show_mice : bool, default=True
        whether to show individual mouse results separately

    Returns
    -------
    None
        saves plot as PDF file in project_root / 'results_extended' / 'relcenter_avg.pdf'
    """
    results_folder = project_root / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1.1, 'LM': 2.1, 'RL': 3.1, 'AL': 4.1, \
                      'TX108_V1': 0.95, 'TX108_LM': 1.95, 'TX108_RL': 2.95, 'TX108_AL': 3.95, \
                      'TX109_V1': 1, 'TX109_LM': 2, 'TX109_RL': 3, 'TX109_AL': 4, \
                      'TX111_V1': 1.05, 'TX111_LM': 2.05, 'TX111_RL': 3.05, 'TX111_AL': 4.05, \
                      'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9, \
                      }
    
    #collect results across layers
    dist_df = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        var_f = cache_folder / f'{layer_name}_distances_avg.csv'
        if not var_f.exists():
            continue
        tmp_df = pd.read_csv(var_f)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        dist_df = pd.concat([dist_df, tmp_df[:1].copy()], axis=0)
    dist_df = dist_df.reset_index(drop=True)

    # Check if there are no TX models, adjust pixels position
    if dist_df[dist_df.model.str.startswith('TX')].empty:
        area_to_x_dict['pixels'] = 4

    #find means and sems
    plot_df = dist_df.groupby(['area'])['rel_center_test'].mean().reset_index()

    plot_df['size'] = 8

    #show each mouse result separately
    if show_mice:
        plot_mouse_df = dist_df[dist_df.model.str.startswith('TX')].copy()
        plot_mouse_df['area'] = plot_mouse_df['model'].str[:5] + '_' + plot_mouse_df['area']
        plot_mouse_df['size'] = 5
        plot_df = pd.concat([plot_df, plot_mouse_df], axis=0).reset_index(drop=True)

    #compute other variables for plotting
    plot_df['color'] = color_pts
    plot_df['errcolor'] = color_err
    plot_df['y'] = plot_df['rel_center_test']
    plot_df['erry'] = 0
    plot_df['x'] = [area_to_x_dict[area] for area, color in zip(plot_df['area'], plot_df['color'])]
    plot_df.sort_values('x', inplace=True)
    plot_df['errx'] = 0

    if plot_df.empty:
        print(f'No data found.')
        return
    fig = plot_points_w_errorbars(plot_df, y_dp=None, sz=8, height=height, width=width, alpha=.6)
    fig.update_layout(yaxis_title='relative distance between means (1)')
    tick_df = plot_df[~plot_df['area'].str.startswith('TX')]
    fig.update_xaxes(title_standoff=0, tickvals=tick_df['x'].unique(), ticktext=tick_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[0,ymax], tickvals=[0,1,2], ticktext=['0','1','2'])
    fig.write_image(results_folder / f'relcenter_avg.pdf', scale=1, engine='kaleido')


def plot_equivariance_relcenter_all(model_df, params_df, project_root, ymax=10, height=150, width=680, \
                      color_inv='rgba(247, 148, 29, 1)', color_tuned='rgba(139, 94, 60, 1)', show_mice=True):
    """
    Plot relative distance between means across layers for invariant and tuned subspaces.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    ymax : float, default=10
        maximum value for y-axis
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_inv : str, default='rgba(247, 148, 29, 1)'
        color for invariant dimensions
    color_tuned : str, default='rgba(139, 94, 60, 1)'
        color for tuned dimensions
    show_mice : bool, default=True
        whether to show individual mouse results separately

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results_extended' / 'relcenter_avg_{d}dim.pdf'
    """
    d = int(params_df.loc['n_procs'].value)
    results_folder = project_root / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1.1, 'LM': 2.1, 'RL': 3.1, 'AL': 4.1, \
                      'TX108_V1': 0.95, 'TX108_LM': 1.95, 'TX108_RL': 2.95, 'TX108_AL': 3.95, \
                      'TX109_V1': 1, 'TX109_LM': 2, 'TX109_RL': 3, 'TX109_AL': 4, \
                      'TX111_V1': 1.05, 'TX111_LM': 2.05, 'TX111_RL': 3.05, 'TX111_AL': 4.05, \
                      'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9, \
                      }
    #shift x-positions for invariant and tuned
    color_to_dx_dict = {f'{color_inv}': -0.1, f'{color_tuned}': 0.1}

    #collect results across layers
    dist_df0 = pd.DataFrame()
    dist_df1 = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        var_f = cache_folder / f'{layer_name}_distances_avg_{d}dim_inv.csv'
        if not var_f.exists():
            continue
        tmp_df = pd.read_csv(var_f, index_col=0)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        tmp_df['color'] = color_inv
        dist_df0 = pd.concat([dist_df0, tmp_df[:1].copy()], axis=0)

        var_f = cache_folder / f'{layer_name}_distances_avg_{d}dim_tun.csv'
        if not var_f.exists():
            continue
        tmp_df = pd.read_csv(var_f, index_col=0)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        tmp_df['color'] = color_tuned
        dist_df1 = pd.concat([dist_df1, tmp_df[:1].copy()], axis=0)
    dist_df = pd.concat([dist_df0, dist_df1], axis=0).reset_index(drop=True)

    # Check if there are no TX models, adjust pixels position
    if dist_df[dist_df.model.str.startswith('TX')].empty:
        area_to_x_dict['pixels'] = 4

    #find means and sems
    plot_df = dist_df.groupby(['area','color'])['rel_center_test'].mean().reset_index()
    plot_df['size'] = 8

    #show each mouse result separately
    if show_mice:
        plot_mouse_df = dist_df[dist_df.model.str.startswith('TX')].copy()
        plot_mouse_df['area'] = plot_mouse_df['model'].str[:5] + '_' + plot_mouse_df['area']
        plot_mouse_df['size'] = 5
        plot_df = pd.concat([plot_df, plot_mouse_df], axis=0).reset_index(drop=True)

    #compute other variables for plotting
    plot_df['errcolor'] = 'rgba(200, 200, 200, 1.0)'
    plot_df['y'] = plot_df['rel_center_test']
    plot_df['x'] = [area_to_x_dict[area] + color_to_dx_dict[color] for area, color in zip(plot_df['area'], plot_df['color'])]
    plot_df['xtick'] = [area_to_x_dict[area] for area in plot_df['area']]
    plot_df.sort_values('x', inplace=True)
    plot_df['errx'] = 0
    plot_df['erry'] = 0

    if plot_df.empty:
        print(f'No data found.')
        return
    fig = plot_points_w_errorbars(plot_df, y_dp=None, sz=8, height=height, width=width, alpha=.6)
    fig.update_layout(yaxis_title='relative distance between means (1)')
    tick_df = plot_df[~plot_df['area'].str.startswith('TX')]
    fig.update_xaxes(title_standoff=0, tickvals=tick_df['xtick'].unique(), ticktext=tick_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[-1,ymax], tickvals=[0,5,10], ticktext=['0','5','10'])
    fig.write_image(results_folder / f'relcenter_avg_{d}dim.pdf', scale=1, engine='kaleido')


def plot_distance_all(model_df, params_df, project_root, ymax=1.3, height=150, width=680, \
                      colors=['rgba(0, 104, 56, 1.0)','rgba(141, 198, 63, 1.0)', \
                      'rgba(200, 50, 150, 1.0)', 'rgba(0, 174, 239, 1.0)']):
    """
    Plot normalized distance to mean across layers.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    ymax : float, default=1.3
        maximum value for y-axis
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    colors : list of str, default=['rgba(0, 104, 56, 1.0)','rgba(141, 198, 63, 1.0)', 'rgba(200, 50, 150, 1.0)', 'rgba(0, 174, 239, 1.0)']
        list of colors for scenes, textures, test1, and test2 respectively

    Returns
    -------
    None
        saves plot as PDF file in project_root / 'results_extended' / 'distances_avg.pdf'
    """
    results_folder = project_root / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1.1, 'LM': 2.1, 'RL': 3.1, 'AL': 4.1, \
                    #   'TX108_V1': 0.95, 'TX108_LM': 1.95, 'TX108_RL': 2.95, 'TX108_AL': 3.95, \
                    #   'TX109_V1': 1, 'TX109_LM': 2, 'TX109_RL': 3, 'TX109_AL': 4, \
                    #   'TX111_V1': 1.05, 'TX111_LM': 2.05, 'TX111_RL': 3.05, 'TX111_AL': 4.05, \
                      'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9, \
                      }
    #shift x-positions for invariant and tuned
    color_to_dx_dict = {f'{colors[0]}': -0.2, f'{colors[1]}': -0.1, f'{colors[2]}': 0.1, f'{colors[3]}': 0.2}
    
    #collect results across layers
    dist_df = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        var_f = cache_folder / f'{layer_name}_distances_avg.csv'
        if not var_f.exists():
            continue
        tmp_df = pd.read_csv(var_f)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        tmp_df['color'] = [colors[0], colors[1], colors[2], colors[3]]
        dist_df = pd.concat([dist_df, tmp_df.copy()], axis=0)
    dist_df = dist_df.reset_index(drop=True)

    # Check if there are no TX models, adjust pixels position
    if dist_df[dist_df.model.str.startswith('TX')].empty:
        area_to_x_dict['pixels'] = 4

    #find means and sems
    plot_df = dist_df.groupby(['area','color'])['norm_dists_mean'].mean().reset_index()
    plot_df['erry'] = dist_df.groupby(['area','color'])['norm_dists_sem'].apply(lambda x: (x**2).sum()**0.5).values
    plot_df['erry'] = plot_df['erry']/np.sqrt(dist_df.groupby(['area','color'])['norm_dists_mean'].count().values)
    plot_df['erry'] = plot_df['erry'].replace(np.nan, 0)

    # #show each mouse result separately
    # plot_mouse_df = dist_df[dist_df.model.str.startswith('TX')].copy()
    # plot_mouse_df['area'] = plot_mouse_df['model'].str[:5] + '_' + plot_mouse_df['area']
    # plot_mouse_df['erry'] = plot_mouse_df.groupby(['area','color'])['norm_dists_sem'].apply(lambda x: (x**2).sum()**0.5).values
    # plot_mouse_df['erry'] = plot_mouse_df['erry']/np.sqrt(plot_mouse_df.groupby(['area','color'])['norm_dists_mean'].count().values)
    # plot_mouse_df['erry'] = plot_mouse_df['erry'].replace(np.nan, 0)
    # plot_df = pd.concat([plot_df, plot_mouse_df], axis=0).reset_index(drop=True)

    #compute other variables for plotting
    plot_df['errcolor'] = plot_df['color']
    plot_df['y'] = plot_df['norm_dists_mean']
    plot_df['x'] = [area_to_x_dict[area] + color_to_dx_dict[color] for area, color in zip(plot_df['area'], plot_df['color'])]
    plot_df['xtick'] = [area_to_x_dict[area] for area in plot_df['area']]
    plot_df.sort_values('x', inplace=True)
    plot_df['errx'] = 0

    if plot_df.empty:
        print(f'No data found.')
        return
    fig = plot_points_w_errorbars(plot_df, y_dp=None, sz=8, height=height, width=width, alpha=.6)
    fig.update_layout(yaxis_title='normalized distance to mean (1)')
    fig.update_xaxes(title_standoff=0, tickvals=plot_df['xtick'].unique(), ticktext=plot_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[0,ymax], tickvals=[0,1,2], ticktext=['0','1','2'])
    fig.write_image(results_folder / f'distances_avg.pdf', scale=1, engine='kaleido')


def plot_equivariance_distance_all(model_df, params_df, project_root, ymax=9, height=150, width=680, \
                      color_inv='rgba(247, 148, 29, 1)', color_tuned='rgba(139, 94, 60, 1)', show_mice=True):
    """
    Plot normalized distance to mean across layers for invariant and tuned subspaces.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    ymax : float, default=9
        maximum value for y-axis
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_inv : str, default='rgba(247, 148, 29, 1)'
        color for invariant dimensions
    color_tuned : str, default='rgba(139, 94, 60, 1)'
        color for tuned dimensions
    show_mice : bool, default=True
        whether to show individual mouse results separately

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results_extended' / 'distances_avg_{d}dim.pdf'
    """
    d = int(params_df.loc['n_procs'].value)
    results_folder = project_root / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1.1, 'LM': 2.1, 'RL': 3.1, 'AL': 4.1, \
                      'TX108_V1': 0.95, 'TX108_LM': 1.95, 'TX108_RL': 2.95, 'TX108_AL': 3.95, \
                      'TX109_V1': 1, 'TX109_LM': 2, 'TX109_RL': 3, 'TX109_AL': 4, \
                      'TX111_V1': 1.05, 'TX111_LM': 2.05, 'TX111_RL': 3.05, 'TX111_AL': 4.05, \
                      'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9, \
                      }
    #shift x-positions for invariant and tuned
    color_to_dx_dict = {f'{color_inv}': -0.1, f'{color_tuned}': 0.1}

    #collect results across layers
    dist_df0 = pd.DataFrame()
    dist_df1 = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        var_f = cache_folder / f'{layer_name}_distances_avg_{d}dim_inv.csv'
        if not var_f.exists():
            continue
        tmp_df = pd.read_csv(var_f, index_col=0)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        tmp_df['color'] = color_inv
        tmp_df['y'] = (tmp_df.loc['test1','norm_dists_mean'] + tmp_df.loc['test2','norm_dists_mean']) / 2
        tmp_df['erry'] = np.sqrt((tmp_df.loc['test1','norm_dists_sem']**2 + tmp_df.loc['test2','norm_dists_sem']**2) / 2)
        dist_df0 = pd.concat([dist_df0, tmp_df[:1].copy()], axis=0)

        var_f = cache_folder / f'{layer_name}_distances_avg_{d}dim_tun.csv'
        if not var_f.exists():
            continue
        tmp_df = pd.read_csv(var_f, index_col=0)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        tmp_df['color'] = color_tuned
        tmp_df['y'] = (tmp_df.loc['test1','norm_dists_mean'] + tmp_df.loc['test2','norm_dists_mean']) / 2
        tmp_df['erry'] = np.sqrt((tmp_df.loc['test1','norm_dists_sem']**2 + tmp_df.loc['test2','norm_dists_sem']**2) / 2)
        dist_df1 = pd.concat([dist_df1, tmp_df[:1].copy()], axis=0)
    dist_df = pd.concat([dist_df0, dist_df1], axis=0).reset_index(drop=True)

    # Check if there are no TX models, adjust pixels position
    if dist_df[dist_df.model.str.startswith('TX')].empty:
        area_to_x_dict['pixels'] = 4

    #find means and sems
    plot_df = dist_df.groupby(['area','color'])['y'].mean().reset_index()
    plot_df['erry'] = dist_df.groupby(['area','color'])['erry'].apply(lambda x: (x**2).sum()**0.5).values
    plot_df['erry'] = plot_df['erry']/np.sqrt(dist_df.groupby(['area','color'])['y'].count().values)
    plot_df['erry'] = plot_df['erry'].replace(np.nan, 0)
    plot_df['size'] = 8

    #show each mouse result separately
    if show_mice:
        plot_mouse_df = dist_df[dist_df.model.str.startswith('TX')].copy()
        plot_mouse_df['area'] = plot_mouse_df['model'].str[:5] + '_' + plot_mouse_df['area']
        plot_mouse_df['size'] = 5
        plot_df = pd.concat([plot_df, plot_mouse_df], axis=0).reset_index(drop=True)

    #compute other variables for plotting
    plot_df['errcolor'] = 'rgba(200, 200, 200, 1.0)'
    plot_df['x'] = [area_to_x_dict[area] + color_to_dx_dict[color] for area, color in zip(plot_df['area'], plot_df['color'])]
    plot_df['xtick'] = [area_to_x_dict[area] for area in plot_df['area']]
    plot_df.sort_values('x', inplace=True)
    plot_df['errx'] = 0

    plot_df.loc[plot_df[(plot_df['area']=='pixels')&(plot_df['color']==color_tuned)].index, 'y'] = 8

    if plot_df.empty:
        print(f'No data found.')
        return
    fig = plot_points_w_errorbars(plot_df, y_dp=None, sz=8, height=height, width=width, alpha=.6)
    fig.update_layout(yaxis_title='normalized distance to mean (1)')
    tick_df = plot_df[~plot_df['area'].str.startswith('TX')]
    fig.update_xaxes(title_standoff=0, tickvals=tick_df['xtick'].unique(), ticktext=tick_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[0,ymax], tickvals=[0,2,4,6,8], ticktext=['0','2','4','6','8'])
    fig.write_image(results_folder / f'distances_avg_{d}dim.pdf', scale=1, engine='kaleido')


def plot_equivariance_variance(model_df, params_df, project_root, height=150, width=680, \
                           color_inv='rgba(247, 148, 29, 1)', color_tuned='rgba(139, 94, 60, 1)', show_mice=True,
                           results_folder=None, subset=None):
    """
    Plot variance explained of invariant and tuned subspaces across layers.

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
    show_mice : bool, default=True
        whether to show individual mouse results separately
    results_folder : str or Path or None, default=None
        output folder for saved files; defaults to project_root / 'results_extended'

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results_extended' / 'var_{n_neurons}avg_{d}dim.pdf'
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    d = int(params_df.loc['n_procs'].value)
    results_folder = _resolve_results_folder(project_root, results_folder, 'results_extended')

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1.1, 'LM': 2.1, 'RL': 3.1, 'AL': 4.1, \
                      'TX108_V1': 0.95, 'TX108_LM': 1.95, 'TX108_RL': 2.95, 'TX108_AL': 3.95, \
                      'TX109_V1': 1, 'TX109_LM': 2, 'TX109_RL': 3, 'TX109_AL': 4, \
                      'TX111_V1': 1.05, 'TX111_LM': 2.05, 'TX111_RL': 3.05, 'TX111_AL': 4.05, \
                      'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9, \
                      }
    #shift x-positions for invariant and tuned
    color_to_dx_dict = {f'{color_inv}': -0.1, f'{color_tuned}': 0.1}

    #collect results across layers
    var_df = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        #non-gaussian noise models read/write suffixed files so they do not clobber
        #the default gaussian caches and figures
        noise_suffix = '' if params_df.loc['noise_model'].value == 'gaussian' else f"_{params_df.loc['noise_model'].value}"
        noise_suffix += f'_{subset}' if subset else ''   #class subset caches (equivariance_summary_all(subset=...))
        var_f = cache_folder / f'{layer_name}_var_dim_{n_neurons}avg_{d}dim{noise_suffix}.csv'
        if not var_f.exists():
            continue
        tmp_df = load_var_dim(var_f)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        var_df = pd.concat([var_df, tmp_df.copy()], axis=0)
    var_df = var_df.reset_index(drop=True)

    #band variance sums per fit (neuron subsample x held-out pair), averaged over subsamples
    #per pair -- never rank-averaged across subsamples (helper.band_summary); the summary
    #table also holds the across-subsample and single-fit standard deviations
    stim_df = band_summary_per_stim(var_df, keys=['model','area'])
    band_summary_table(var_df, keys=['model','area']).to_csv(
        results_folder / f'var_{n_neurons}avg_{d}dim{noise_suffix}_summary.csv', index=False)

    #significance for invariant dimensions
    inv_pc_df = stim_df[['model','area','stim','var test inv']].rename(columns={'var test inv': 'var test'})
    results_df = compute_significance(inv_pc_df.rename(columns={'var test': 'score'}), alternative='less')
    results_df.to_csv(results_folder / f'var_inv_significance{noise_suffix}.csv', index=False)

    #significance for tuned dimensions
    tuned_pc_df = stim_df[['model','area','stim','var test tun']].rename(columns={'var test tun': 'var test'})
    results_df = compute_significance(tuned_pc_df.rename(columns={'var test': 'score'}), alternative='less')
    results_df.to_csv(results_folder / f'var_tuned_significance{noise_suffix}.csv', index=False)
    
    #show each mouse result separately
    if show_mice:
        inv_mouse_df = inv_pc_df[inv_pc_df.model.str.startswith('TX')].copy()
        inv_mouse_df['area'] = inv_mouse_df['model'].str[:5] + '_' + inv_mouse_df['area']
        inv_mouse_avg_df = inv_mouse_df.groupby(['model','area'])['var test'].mean().reset_index().rename(columns={'var test': 'var'})
        inv_mouse_avg_df['erry'] = inv_mouse_df.groupby(['model','area'])['var test'].std().values/np.sqrt(var_df['stim'].nunique())
        inv_mouse_avg_df['cat'] = 0
        tuned_mouse_df = tuned_pc_df[tuned_pc_df.model.str.startswith('TX')].copy()
        tuned_mouse_df['area'] = tuned_mouse_df['model'].str[:5] + '_' + tuned_mouse_df['area']
        tuned_mouse_avg_df = tuned_mouse_df.groupby(['model','area'])['var test'].mean().reset_index().rename(columns={'var test': 'var'})
        tuned_mouse_avg_df['erry'] = tuned_mouse_df.groupby(['model','area'])['var test'].std().values/np.sqrt(var_df['stim'].nunique())
        tuned_mouse_avg_df['cat'] = 1
        mouse_pc_df = pd.concat([inv_mouse_avg_df, tuned_mouse_avg_df], axis=0)

    #compute means and sems in each layer
    inv_pc_avg_df = inv_pc_df.groupby(['model','area'])['var test'].mean().reset_index().rename(columns={'var test': 'var'})
    inv_pc_avg_df['erry'] = inv_pc_df.groupby(['model','area'])['var test'].std().values/np.sqrt(var_df['stim'].nunique())
    inv_pc_avg_df['cat'] = 0
    tuned_pc_avg_df = tuned_pc_df.groupby(['model','area'])['var test'].mean().reset_index().rename(columns={'var test': 'var'})
    tuned_pc_avg_df['erry'] = tuned_pc_df.groupby(['model','area'])['var test'].std().values/np.sqrt(var_df['stim'].nunique())
    tuned_pc_avg_df['cat'] = 1
    pc_df = pd.concat([inv_pc_avg_df, tuned_pc_avg_df], axis=0)

    #add individual mouse results
    if show_mice:
        pc_df = pd.concat([pc_df, mouse_pc_df], axis=0).reset_index(drop=True)

    # Check if there are no TX models, adjust pixels position
    if pc_df[pc_df.model.str.startswith('TX')].empty:
        area_to_x_dict['pixels'] = 4

    #compute means and sems in each model
    plot_df = pc_df.groupby(['area','cat'])['var'].mean().reset_index()
    plot_df['color'] = color_inv
    idx_tuned = plot_df[plot_df['cat']==1].index
    plot_df.loc[idx_tuned,'color'] = color_tuned
    plot_df['errcolor'] = 'rgba(200, 200, 200, 1.0)'
    plot_df['y'] = plot_df['var']
    plot_df['erry'] = pc_df.groupby(['area','cat'])['erry'].apply(lambda x: (x**2).sum()**0.5).values
    plot_df['erry'] = plot_df['erry']/np.sqrt(pc_df.groupby(['area','cat'])['var'].count().values)
    plot_df['erry'] = plot_df['erry'].replace(np.nan, 0)
    plot_df['size'] = np.where(plot_df['area'].str.startswith('TX'), 5, 8)
    plot_df['x'] = [area_to_x_dict[area] + color_to_dx_dict[color] for area, color in zip(plot_df['area'], plot_df['color'])]
    plot_df['xtick'] = [area_to_x_dict[area] for area in plot_df['area']]
    plot_df.sort_values('x', inplace=True)
    plot_df['errx'] = 0

    if plot_df.empty:
        print(f'No data found.')
        return
    fig = plot_points_w_errorbars(plot_df, y_dp=None, sz=8, height=height, width=width, alpha=.6)
    fig.update_layout(yaxis_title='Variance explained (1)')
    tick_df = plot_df[~plot_df['area'].str.startswith('TX')]
    fig.update_xaxes(title_standoff=0, tickvals=tick_df['xtick'].unique(), ticktext=tick_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[0,0.2], tickvals=[0,0.1,0.2], ticktext=['0','10','20'])
    fig.write_image(results_folder / f'var_{n_neurons}avg_{d}dim{noise_suffix}.pdf', scale=1, engine='kaleido')


def plot_equivariance_pcorr(model_df, params_df, project_root, height=150, width=680, ymax=0.75, \
                           color_inv='rgba(247, 148, 29, 1)', color_tuned='rgba(139, 94, 60, 1)', show_mice=True,
                           results_folder=None, subset=None, pc_weighted=True,
                           overlay_params_df=None, alpha_faint=0.4, return_df=False):
    """
    Plot Pearson correlation of invariant and tuned subspaces across layers.

    The band correlation of each fit is the variance-weighted mean over the band's dimensions
    (pc_weighted=True, the paper's degrees of invariance and tuning, Fig 4b); pc_weighted=False
    gives the plain mean over dimensions.

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
    ymax : float, default=0.75
        maximum value for y-axis
    color_inv : str, default='rgba(247, 148, 29, 1)'
        color for invariant dimensions
    color_tuned : str, default='rgba(139, 94, 60, 1)'
        color for tuned dimensions
    show_mice : bool, default=True
        whether to show individual mouse results separately
    results_folder : str or Path or None, default=None
        output folder for saved files; defaults to project_root / 'results'

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results' / 'pcorr_{n_neurons}avg_{d}dim.pdf'
    """
    if overlay_params_df is not None:   #N-overlay figures show area means only
        show_mice = False
    n_neurons = int(params_df.loc['n_neurons'].value)
    d = int(params_df.loc['n_procs'].value)
    results_folder = _resolve_results_folder(project_root, results_folder, 'results')

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1.1, 'LM': 2.1, 'RL': 3.1, 'AL': 4.1, \
                      'TX108_V1': 0.95, 'TX108_LM': 1.95, 'TX108_RL': 2.95, 'TX108_AL': 3.95, \
                      'TX109_V1': 1, 'TX109_LM': 2, 'TX109_RL': 3, 'TX109_AL': 4, \
                      'TX111_V1': 1.05, 'TX111_LM': 2.05, 'TX111_RL': 3.05, 'TX111_AL': 4.05, \
                      'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9, \
                      }
    #shift x-positions for invariant and tuned
    color_to_dx_dict = {f'{color_inv}': -0.1, f'{color_tuned}': 0.1}

    #collect results across layers
    var_df = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        #non-gaussian noise models read/write suffixed files so they do not clobber
        #the default gaussian caches and figures
        noise_suffix = '' if params_df.loc['noise_model'].value == 'gaussian' else f"_{params_df.loc['noise_model'].value}"
        noise_suffix += f'_{subset}' if subset else ''   #class subset caches (equivariance_summary_all(subset=...))
        var_f = cache_folder / f'{layer_name}_var_dim_{n_neurons}avg_{d}dim{noise_suffix}.csv'
        if not var_f.exists():
            continue
        tmp_df = load_var_dim(var_f)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        var_df = pd.concat([var_df, tmp_df.copy()], axis=0)
    var_df = var_df.reset_index(drop=True)

    #band correlation means per fit (neuron subsample x held-out pair), averaged over
    #subsamples per pair -- never rank-averaged across subsamples (helper.band_summary);
    #pairs whose band is empty in every subsample are dropped, as before
    stim_df = band_summary_per_stim(var_df, keys=['model','area'], pc_weighted=pc_weighted)
    band_summary_table(var_df, keys=['model','area'], pc_weighted=pc_weighted).to_csv(
        results_folder / f'pcorr_{n_neurons}avg_{d}dim{noise_suffix}_summary.csv', index=False)

    #significance for invariant dimensions
    inv_pc_df = stim_df[['model','area','stim','pc test inv']].rename(columns={'pc test inv': 'pc test'}).dropna()
    results_df = compute_significance(inv_pc_df.rename(columns={'pc test': 'score'}), alternative='less')
    results_df.to_csv(results_folder / f'pcorr_inv_significance{noise_suffix}.csv', index=False)

    #significance for tuned dimensions
    tuned_pc_df = stim_df[['model','area','stim','pc test tun']].rename(columns={'pc test tun': 'pc test'}).dropna()
    results_df = compute_significance(tuned_pc_df.rename(columns={'pc test': 'score'}), alternative='less')
    results_df.to_csv(results_folder / f'pcorr_tuned_significance{noise_suffix}.csv', index=False)
    
    #show each mouse result separately
    if show_mice:
        inv_mouse_df = inv_pc_df[inv_pc_df.model.str.startswith('TX')].copy()
        inv_mouse_df['area'] = inv_mouse_df['model'].str[:5] + '_' + inv_mouse_df['area']
        inv_mouse_avg_df = inv_mouse_df.groupby(['model','area'])['pc test'].mean().reset_index().rename(columns={'pc test': 'pc'})
        inv_mouse_avg_df['erry'] = inv_mouse_df.groupby(['model','area'])['pc test'].std().values/np.sqrt(var_df['stim'].nunique())
        inv_mouse_avg_df['cat'] = 0
        tuned_mouse_df = tuned_pc_df[tuned_pc_df.model.str.startswith('TX')].copy()
        tuned_mouse_df['area'] = tuned_mouse_df['model'].str[:5] + '_' + tuned_mouse_df['area']
        tuned_mouse_avg_df = tuned_mouse_df.groupby(['model','area'])['pc test'].mean().reset_index().rename(columns={'pc test': 'pc'})
        tuned_mouse_avg_df['erry'] = tuned_mouse_df.groupby(['model','area'])['pc test'].std().values/np.sqrt(var_df['stim'].nunique())
        tuned_mouse_avg_df['cat'] = 1
        mouse_pc_df = pd.concat([inv_mouse_avg_df, tuned_mouse_avg_df], axis=0)

    #compute means and sems in each layer
    inv_pc_avg_df = inv_pc_df.groupby(['model','area'])['pc test'].mean().reset_index().rename(columns={'pc test': 'pc'})
    inv_pc_avg_df['erry'] = inv_pc_df.groupby(['model','area'])['pc test'].std().values/np.sqrt(var_df['stim'].nunique())
    inv_pc_avg_df['cat'] = 0
    tuned_pc_avg_df = tuned_pc_df.groupby(['model','area'])['pc test'].mean().reset_index().rename(columns={'pc test': 'pc'})
    tuned_pc_avg_df['erry'] = tuned_pc_df.groupby(['model','area'])['pc test'].std().values/np.sqrt(var_df['stim'].nunique())
    tuned_pc_avg_df['cat'] = 1
    pc_df = pd.concat([inv_pc_avg_df, tuned_pc_avg_df], axis=0)

    #add individual mouse results
    if show_mice:
        pc_df = pd.concat([pc_df, mouse_pc_df], axis=0).reset_index(drop=True)

    # Check if there are no TX models, adjust pixels position
    if pc_df[pc_df.model.str.startswith('TX')].empty:
        area_to_x_dict['pixels'] = 4

    #compute means and sems in each model
    plot_df = pc_df.groupby(['area','cat'])['pc'].mean().reset_index()
    plot_df['color'] = color_inv
    idx_tuned = plot_df[plot_df['cat']==1].index
    plot_df.loc[idx_tuned,'color'] = color_tuned
    plot_df['errcolor'] = 'rgba(200, 200, 200, 1.0)'
    plot_df['y'] = plot_df['pc']
    plot_df['erry'] = pc_df.groupby(['area','cat'])['erry'].apply(lambda x: (x**2).sum()**0.5).values
    plot_df['erry'] = plot_df['erry']/np.sqrt(pc_df.groupby(['area','cat'])['pc'].count().values)
    plot_df['erry'] = plot_df['erry'].replace(np.nan, 0)
    plot_df['size'] = np.where(plot_df['area'].str.startswith('TX'), 5, 8)
    plot_df['x'] = [area_to_x_dict[area] + color_to_dx_dict[color] for area, color in zip(plot_df['area'], plot_df['color'])]
    plot_df['xtick'] = [area_to_x_dict[area] for area in plot_df['area']]
    plot_df.sort_values('x', inplace=True)
    plot_df['errx'] = 0

    if plot_df.empty:
        print(f'No data found.')
        return
    if return_df:
        return plot_df
    if overlay_params_df is not None:
        #semi-transparent layer at the other neuron count (same x, area means only), drawn
        #first so the opaque params_df layer sits on top (SI N = 250 vs N = max figures)
        over_df = plot_equivariance_pcorr(model_df, overlay_params_df, project_root, show_mice=False,
                                          results_folder=results_folder, subset=subset, pc_weighted=pc_weighted, return_df=True)
        plot_df = pd.concat([_faint(over_df, alpha_faint), plot_df], ignore_index=True)
    fig = plot_points_w_errorbars(plot_df, y_dp=None, sz=8, height=height, width=width, alpha=.6)
    fig.update_layout(yaxis_title='Pearson correlation (1)')
    tick_df = plot_df[~plot_df['area'].str.startswith('TX')]
    fig.update_xaxes(title_standoff=0, tickvals=tick_df['xtick'].unique(), ticktext=tick_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[0,ymax])
    fig.write_image(results_folder / f'pcorr_{n_neurons}avg_{d}dim{noise_suffix}{_overlay_tag(overlay_params_df)}.pdf', scale=1, engine='kaleido')


def _both_null_bands(model_df, project_root, tag):
    """Chance recovered dimensions per band from the 'both' shuffle (image identity AND rotation
    order destroyed), averaged over rows. This structureless null is ~flat across areas (~1.8
    invariant, ~2.4 tuned dims -- it only reflects how many of the d fitted noise dimensions leak
    into each |eigenphase| band), so a single value per band is a fair reference line. Reads the
    precomputed results_extended/null_bands_{tag}.csv (plotting_validation.null_band_table) when
    present, else recomputes from the raw {area}_var_dim_null_{tag}.csv caches (slow). Returns
    (d_inv, d_tun), or (None, None) if neither is available."""
    summ = project_root / 'results_extended' / f'null_bands_{tag}.csv'
    if summ.exists():
        s = pd.read_csv(summ)
        if {'both d_inv', 'both d_tun'} <= set(s.columns) and s['both d_inv'].notna().any():
            return float(s['both d_inv'].mean()), float(s['both d_tun'].mean())
    d_inv, d_tun = [], []
    for i0 in range(model_df.shape[0]):
        f = project_root / 'cache' / model_df['network'].values[i0] / \
            f"{model_df['layer'].values[i0]}_var_dim_null_{tag}.csv"
        if not f.exists():
            continue
        nd = pd.read_csv(f, usecols=['kind', 'shuffle', 'stim', 'sample', 'evals'])
        nd = nd[nd['kind'] == 'both']
        if nd.empty:
            continue
        b = band_summary(nd, keys=['shuffle'])          #one fit per (shuffle, stim, sample)
        d_inv.append(b['d_inv'].mean())
        d_tun.append(b['d_tun'].mean())
    if not d_inv:
        return None, None
    return float(np.mean(d_inv)), float(np.mean(d_tun))


def plot_equivariance_dim(model_df, params_df, project_root, height=150, width=680, ymax=20, \
                           color_inv='rgba(247, 148, 29, 1)', color_tuned='rgba(139, 94, 60, 1)', show_mice=True,
                           mouse_gap_shift=0.10, results_folder=None, subset=None,
                           overlay_params_df=None, alpha_faint=0.4, return_df=False):
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
    ymax : float, default=20
        maximum value for y-axis
    color_inv : str, default='rgba(247, 148, 29, 1)'
        color for invariant dimensions
    color_tuned : str, default='rgba(139, 94, 60, 1)'
        color for tuned dimensions
    show_mice : bool, default=True
        whether to show individual mouse results separately
    results_folder : str or Path or None, default=None
        output folder for saved files; defaults to project_root / 'results'

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results' / 'dim_{n_neurons}avg_{d}dim.pdf'
    """
    if overlay_params_df is not None:   #N-overlay figures show area means only
        show_mice = False
    
    n_neurons = int(params_df.loc['n_neurons'].value)
    d = int(params_df.loc['n_procs'].value)
    results_folder = _resolve_results_folder(project_root, results_folder, 'results')

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1.1, 'LM': 2.1, 'RL': 3.1, 'AL': 4.1, \
                      'TX108_V1': 0.95, 'TX108_LM': 1.95, 'TX108_RL': 2.95, 'TX108_AL': 3.95, \
                      'TX109_V1': 1, 'TX109_LM': 2, 'TX109_RL': 3, 'TX109_AL': 4, \
                      'TX111_V1': 1.05, 'TX111_LM': 2.05, 'TX111_RL': 3.05, 'TX111_AL': 4.05, \
                      'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9, \
                      }
    #shift x-positions for invariant and tuned (0.15 so that with the mouse_gap_shift below the
    #tuned mouse cluster clears the invariant area mean: ... inv mice, gap, inv mean, tuned mice,
    #gap, tuned mean ...)
    color_to_dx_dict = {f'{color_inv}': -0.15, f'{color_tuned}': 0.15}

    #collect results across layers
    dim_df = pd.DataFrame()
    var_all_df = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        #non-gaussian noise models read/write suffixed files so they do not clobber
        #the default gaussian caches and figures
        noise_suffix = '' if params_df.loc['noise_model'].value == 'gaussian' else f"_{params_df.loc['noise_model'].value}"
        noise_suffix += f'_{subset}' if subset else ''   #class subset caches (equivariance_summary_all(subset=...))
        var_f = cache_folder / f'{layer_name}_var_dim_{n_neurons}avg_{d}dim{noise_suffix}.csv'
        if not var_f.exists():
            continue
        var_df = load_var_dim(var_f)
        var_df['model'] = model_name
        var_df['area'] = layer_name
        var_all_df = pd.concat([var_all_df, var_df], axis=0)
        #invariant and tuned dimension counts per fit (neuron subsample x held-out pair),
        #averaged over subsamples per pair -- never counted on rank-averaged spectra
        #(helper.band_summary); error bars are sems across held-out pairs, as before
        stim_df = band_summary_per_stim(var_df)
        n_stim = stim_df['stim'].nunique()
        tmp_df = pd.DataFrame({'dim': [stim_df['d_inv'].mean(), stim_df['d_tun'].mean()],
                               'erry': [stim_df['d_inv'].std()/np.sqrt(n_stim), stim_df['d_tun'].std()/np.sqrt(n_stim)]})
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        dim_df = pd.concat([dim_df, tmp_df.reset_index(names='cat')], axis=0)

    #summary table with the single-fit and across-subsample standard deviations
    if not dim_df.empty:
        table_df = band_summary_table(var_all_df, keys=['model','area'])
        table_df[table_df['quantity'].isin(['d_inv','d_tun'])].to_csv(
            results_folder / f'dim_{n_neurons}avg_{d}dim{noise_suffix}_summary.csv', index=False)

    #find means and sems
    plot_df = dim_df.groupby(['area','cat'])['dim'].mean().reset_index()
    plot_df['erry'] = dim_df.groupby(['area','cat'])['erry'].apply(lambda x: (x**2).sum()**0.5).values
    plot_df['erry'] = plot_df['erry']/np.sqrt(dim_df.groupby(['area','cat'])['dim'].count().values)
    plot_df['erry'] = plot_df['erry'].replace(np.nan, 0)

    plot_df['size'] = 8

    #show each mouse result separately
    if show_mice:
        plot_mouse_df = dim_df[dim_df.model.str.startswith('TX')].copy()
        plot_mouse_df['area'] = plot_mouse_df['model'].str[:5] + '_' + plot_mouse_df['area']
        plot_mouse_df['erry'] = plot_mouse_df.groupby(['area','cat'])['erry'].apply(lambda x: (x**2).sum()**0.5).values
        plot_mouse_df['erry'] = plot_mouse_df['erry']/np.sqrt(plot_mouse_df.groupby(['area','cat'])['dim'].count().values)
        plot_mouse_df['erry'] = plot_mouse_df['erry'].replace(np.nan, 0)
        plot_mouse_df['size'] = 5
        plot_df = pd.concat([plot_df, plot_mouse_df], axis=0).reset_index(drop=True)

    # Check if there are no TX models, adjust pixels position
    if dim_df[dim_df.model.str.startswith('TX')].empty:
        area_to_x_dict['pixels'] = 4

    #compute other variables for plotting
    plot_df['color'] = color_inv
    idx_tuned = plot_df[plot_df['cat']==1].index
    plot_df.loc[idx_tuned,'color'] = color_tuned
    plot_df['errcolor'] = 'rgba(200, 200, 200, 1.0)'
    plot_df['y'] = plot_df['dim']
    #mouse markers sit left of their area mean; shift the whole mouse cluster left by
    #mouse_gap_shift so the right-most mouse is further from the pooled mean, keeping the
    #inter-mouse spacing unchanged (right-most gap 0.05 -> 0.15 at the default 0.10 shift),
    #as in plot_ccnorm_pooled_equiv. In the mouse areas the invariant area mean sits between
    #the two mouse clusters, so there the whole tuned family is additionally pushed right so
    #the left-most tuned mouse marker clears the invariant mean.
    mouse_areas = ('V1', 'LM', 'RL', 'AL')
    plot_df['x'] = [area_to_x_dict[area] + color_to_dx_dict[color]
                    - (mouse_gap_shift if area.startswith('TX') else 0)
                    + (0.10 if (color == color_tuned and area.split('_')[-1] in mouse_areas) else 0)
                    for area, color in zip(plot_df['area'], plot_df['color'])]
    plot_df['xtick'] = [area_to_x_dict[area] for area in plot_df['area']]
    plot_df.sort_values('x', inplace=True)
    plot_df['errx'] = 0

    if plot_df.empty:
        print(f'No data found.')
        return
    if return_df:
        return plot_df
    if overlay_params_df is not None:
        #semi-transparent layer at the other neuron count (same x, area means only), drawn
        #first so the opaque params_df layer sits on top (SI N = 250 vs N = max figures)
        over_df = plot_equivariance_dim(model_df, overlay_params_df, project_root, show_mice=False, mouse_gap_shift=mouse_gap_shift,
                                        results_folder=results_folder, subset=subset, return_df=True)
        plot_df = pd.concat([_faint(over_df, alpha_faint), plot_df], ignore_index=True)
    fig = plot_points_w_errorbars(plot_df, sz=8, height=height, width=width, alpha=.6)
    #chance dimension per band from the 'both' shuffle (structureless null, ~flat across areas):
    #invariant (orange) ~1.8, tuned (brown) ~2.4 -- dotted reference lines like the decoding figures
    null_inv, null_tun = _both_null_bands(model_df, project_root, f'{n_neurons}avg_{d}dim{noise_suffix}')
    if null_inv is not None:
        fig.add_hline(y=null_inv, line_width=1, line_color=color_inv, opacity=.8, line_dash='dot')
        fig.add_hline(y=null_tun, line_width=1, line_color=color_tuned, opacity=.8, line_dash='dot')
    fig.update_layout(yaxis_title='dimension (1)')
    tick_df = plot_df[~plot_df['area'].str.startswith('TX')]
    fig.update_xaxes(title_standoff=0, tickvals=tick_df['xtick'].unique(), ticktext=tick_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[0,ymax], tickvals=[0,10,20], ticktext=['0','10','20'])
    fig.write_image(results_folder / f'dim_{n_neurons}avg_{d}dim{noise_suffix}{_overlay_tag(overlay_params_df)}.pdf', scale=1, engine='kaleido')


def plot_example_decoder_weights(model_df, params_df, project_root, height=150, width=150, equivariant=False):
    """
    Plot example decoder weights in PCA and SVM spaces.

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
    equivariant : bool, default=False
        whether to plot equivariant decoder weights

    Returns
    -------
    None
        saves plots as PDF in project_root / 'results_extended' / 'example_weights' / '{model_name}_{layer_name}_example_decoder_{n_neurons}rand_comp_{i}{j}.pdf'
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    results_folder = project_root / 'results_extended' / 'example_weights'
    results_folder.mkdir(parents=True, exist_ok=True)
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
    else:
        d = None
        suffix = ''

    #loop over models & layers
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name
        ori_df = pd.read_csv(cache_folder / f'{layer_name}_ori12_weights_{n_neurons}rand{suffix}.csv', index_col=False)
        svm_df = pd.read_csv(cache_folder / f'{layer_name}_svm12_weights_{n_neurons}rand{suffix}.csv', index_col=False)

        if not equivariant:
            #x-axis is the SVM weights and y-axis the orientation PC weights, which live on
            #different scales, so give each axis its own symmetric range to show all points
            xmax = 1.1*svm_df[['img91','img92']].abs().max().max()
            ymax = 1.1*ori_df[['PC1','PC2']].abs().max().max()
            for i in range(2):
                for j in range(2):
                    plot_df = pd.DataFrame({'x': svm_df[f'img9{i+1}'].values, 'y': ori_df[f'PC{j+1}'].values})
                    plot_df['errx'] = None
                    plot_df['erry'] = None
                    plot_df['color'] = 'rgba(100, 100, 100, 0.4)'
                    plot_df['errcolor'] = 'rgba(255, 255, 255, 0.0)'
                    fig = plot_points_w_errorbars(plot_df, sz=3, height=height, width=width, alpha=.6)
                    fig.add_hline(y=0, line_color='lightgrey', line_width=1, line_dash='dot')
                    fig.add_vline(x=0, line_color='lightgrey', line_width=1, line_dash='dot')
                    fig.update_layout(xaxis_range=[-xmax,xmax], yaxis_range=[-ymax,ymax])
                    fig.write_image(results_folder / f'{model_name}_{layer_name}_example_decoder_{n_neurons}rand{suffix}_comp_{i}{j}.pdf', scale=1, engine='kaleido')
        else:
            for i in range(2):
                if i==0:
                    xymax = 1.1*ori_df[['PC1','PC2']].abs().max().max()
                    plot_df = pd.DataFrame({'x': ori_df['PC1'].values, 'y': ori_df['PC2'].values})
                else:
                    xymax = 1.1*svm_df[['img91','img92']].abs().max().max()
                    plot_df = pd.DataFrame({'x': svm_df['img91'].values, 'y': svm_df['img92'].values})
                plot_df['errx'] = None
                plot_df['erry'] = None
                plot_df['color'] = 'rgba(100, 100, 100, 0.4)'
                plot_df['errcolor'] = 'rgba(255, 255, 255, 0.0)'
                fig = plot_points_w_errorbars(plot_df, sz=3, height=height, width=width, alpha=.6)
                fig.add_hline(y=0, line_color='lightgrey', line_width=1, line_dash='dot')
                fig.add_vline(x=0, line_color='lightgrey', line_width=1, line_dash='dot')
                fig.update_layout(xaxis_range=[-xymax,xymax], yaxis_range=[-xymax,xymax])
                fig.write_image(results_folder / f'{model_name}_{layer_name}_example_decoder_{n_neurons}rand{suffix}_comp_{i}.pdf', scale=1, engine='kaleido')


def plot_example_projs(model_df, params_df, project_root, height=150, width=150, suffix='_data', \
                       color_test=['rgba(200, 50, 150, 1.0)', 'rgba(0, 174, 239, 1.0)'], xymax=None):
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
    xymax : float, default=None
        maximum value for x and y axes

    Returns
    -------
    None
        saves plots as PDF in project_root / 'results' / 'example_projs' / '{model_name}_{layer_name}_example_projs{suffix}_{n_neurons}rand_{d}dim_{invariant|tuned}.pdf'
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    d = int(params_df.loc['n_procs'].value)
    stim_name = params_df.loc['stimulus'].value
    results_folder = project_root / 'results' / 'example_projs'
    results_folder.mkdir(parents=True, exist_ok=True)

    n_rot, _, n_test = find_n_imgs(stim_name)

    #loop over models & layers
    for i0 in range(model_df.shape[0]):
        
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name
        proj_df = pkl.load( open( cache_folder / f'{layer_name}_example_projs{suffix}_{n_neurons}rand_{d}dim.p', "rb" ) )

        id0 = np.argsort(np.abs(proj_df['eval'].values))[:n_test*n_rot]
        id1 = np.argsort(np.abs(np.abs(proj_df['eval'].values)-1))[:n_test*n_rot]
        if xymax is None:
            #common symmetric range across both subspaces AND both the data and model
            #caches, so all four plots for this area share the same x- and y-range
            xymax_local = 0
            for sfx in ['_data', '_model']:
                df_sfx = pkl.load( open( cache_folder / f'{layer_name}_example_projs{sfx}_{n_neurons}rand_{d}dim.p', "rb" ) )
                i0_sfx = np.argsort(np.abs(df_sfx['eval'].values))[:n_test*n_rot]
                i1_sfx = np.argsort(np.abs(np.abs(df_sfx['eval'].values)-1))[:n_test*n_rot]
                xymax_local = max(xymax_local, 1.1*df_sfx.loc[np.concatenate([i0_sfx, i1_sfx]), ['x','y']].abs().max().max())
        else:
            xymax_local = xymax

        #loop over invariant and tuned subspace
        for j, subspace_name in enumerate(['invariant', 'tuned']):
            idx = id0 if j==0 else id1
            tmp_df = proj_df.loc[idx].sort_index()
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
            fig.update_layout(height=height, width=width, xaxis_range=[-xymax_local,xymax_local], yaxis_range=[-xymax_local,xymax_local])
            fig.write_image(results_folder / f'{model_name}_{layer_name}_example_projs{suffix}_{n_neurons}rand_{d}dim_{subspace_name}.pdf', scale=1, engine='kaleido')


def plot_example_evals_abs(model_df, params_df, project_root, height=150, width=200, \
                           color_pts='rgba(140, 140, 140, 1)', color_inv='rgba(247, 148, 29, 1)', color_tuned='rgba(139, 94, 60, 1)'):
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
        fig = plot_eval(plot_df, sz=6, height=height, width=width)
        fig.write_image(results_folder / f'{model_name}_{layer_name}_example_evals_abs_{n_neurons}rand_{d}dim.pdf', scale=1, engine='kaleido')


def plot_orientation_errors(model_df, params_df, project_root, ref_index=0, height=150, width=680, ymin=40,
                            color_pts='rgba(140, 140, 140, 1)', color_err='rgba(169, 169, 169, 1.0)',
                            equivariant=False, model=False, symb='circle', show_mice=True,
                            results_folder=None, overlay_params_df=None, alpha_faint=0.4, return_df=False):
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
    ref_index : int, default=0
        reference orientation index for alignment
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    ymin : int, default=40
        minimum y-axis value
    color_pts : str, default='rgba(140, 140, 140, 1)'
        color for data points
    color_err : str, default='rgba(169, 169, 169, 1.0)'
        color for error bars
    equivariant : bool, default=False
        whether to use equivariant analysis
    model : bool, default=False
        whether to plot decoder from projected model predictions or projected recorded responses
    symb : str, default='circle'
        symbol for data points
    show_mice : bool, default=True
        whether to show individual mouse results separately
    results_folder : str or Path or None, default=None
        output folder for saved files; defaults to project_root / 'results'

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results' / 'ori_{ref_index}ref_{n_neurons}avg{suffix}.pdf'
    """
    if overlay_params_df is not None:   #N-overlay figures show area means only
        show_mice = False
    n_neurons = int(params_df.loc['n_neurons'].value)
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
        if model:
            suffix += '_model'
    else:
        suffix = ''
    #non-gaussian noise models read/write suffixed files so they do not clobber the
    #default gaussian caches and figures
    noise_model = params_df.loc['noise_model'].value
    suffix += '' if noise_model == 'gaussian' else f'_{noise_model}'
    results_folder = _resolve_results_folder(project_root, results_folder, 'results')

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1.1, 'LM': 2.1, 'RL': 3.1, 'AL': 4.1, \
                      'TX108_V1': 0.95, 'TX108_LM': 1.95, 'TX108_RL': 2.95, 'TX108_AL': 3.95, \
                      'TX109_V1': 1, 'TX109_LM': 2, 'TX109_RL': 3, 'TX109_AL': 4, \
                      'TX111_V1': 1.05, 'TX111_LM': 2.05, 'TX111_RL': 3.05, 'TX111_AL': 4.05, \
                      'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9, \
                      }

    #collect results across layers
    results_df = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        ori_f = cache_folder / f'{layer_name}_ori2x_decoder_{ref_index}ref_{n_neurons}avg{suffix}.csv'
        if not ori_f.exists():
            continue
        tmp_df = pd.read_csv(ori_f)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        results_df = pd.concat([results_df, tmp_df], axis=0)

    sign_df = compute_significance(results_df.rename(columns={'res': 'score'}), alternative='greater')
    sign_df.to_csv(results_folder / f'ori_significance_{ref_index}ref{suffix}.csv', index=False)

    #show each mouse result separately
    if show_mice:
        mouse_results_df = results_df[results_df.model.str.startswith('TX')].copy()
        mouse_results_df['area'] = mouse_results_df['model'].str[:5] + '_' + mouse_results_df['area']
        plot_mouse_df = mouse_results_df.groupby(['area'])['res'].mean().reset_index()
        plot_mouse_df['erry'] = mouse_results_df.groupby(['area'])['res'].std().reset_index()['res']
        plot_mouse_df['erry'] = plot_mouse_df['erry']/np.sqrt(mouse_results_df.groupby(['area'])['res'].count().values)
        plot_mouse_df['color'] = 'rgba(180, 180, 180, 1.0)'
        plot_mouse_df['errcolor'] = 'rgba(209, 209, 209, 1.0)'
        plot_mouse_df['symbol'] = 'diamond'
        plot_mouse_df['size'] = 5

    #compute mean and sem for each area
    plot_df = results_df.groupby(['area'])['res'].mean().reset_index()
    plot_df['erry'] = results_df.groupby(['area'])['res'].std().reset_index()['res']
    plot_df['erry'] = plot_df['erry']/np.sqrt(results_df.groupby(['area'])['res'].count().values)
    plot_df['color'] = color_pts
    plot_df['errcolor'] = color_err
    plot_df['symbol'] = 'diamond'
    plot_df['size'] = 8

    #add individual mouse results
    if show_mice:
        plot_df = pd.concat([plot_df, plot_mouse_df[['area','res','erry','color','errcolor','symbol','size']]], axis=0).reset_index(drop=True)

    # Check if there are no TX models, adjust pixels position
    if results_df[results_df.model.str.startswith('TX')].empty:
        area_to_x_dict['pixels'] = 4

    #compute mean and sem for each area
    plot_df['y'] = plot_df['res']
    plot_df['x'] = [area_to_x_dict[area] for area in plot_df['area']]
    plot_df['xtick'] = [area_to_x_dict[area] for area in plot_df['area']]
    plot_df.sort_values('x', inplace=True)
    plot_df['errx'] = 0

    if plot_df.empty:
        print(f'No data found.')
        return
    if return_df:
        return plot_df
    if overlay_params_df is not None:
        #semi-transparent layer at the other neuron count (same x, area means only), drawn
        #first so the opaque params_df layer sits on top (SI N = 250 vs N = max figures)
        over_df = plot_orientation_errors(model_df, overlay_params_df, project_root, ref_index=ref_index, equivariant=equivariant,
                                          model=model, show_mice=False, results_folder=results_folder, return_df=True)
        plot_df = pd.concat([_faint(over_df, alpha_faint), plot_df], ignore_index=True)
    fig = plot_points_w_errorbars(plot_df, y_dp=[90], sz=8, height=height, width=width, symb='diamond')
    fig.update_layout(yaxis_title='absolute angle estimation error (deg)')
    tick_df = plot_df[~plot_df['area'].str.startswith('TX')]
    fig.update_xaxes(title_standoff=0, tickvals=tick_df['xtick'].unique(), ticktext=tick_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[ymin,95], tickvals=[0,45,90,180], ticktext=['0 ','45 ','90 ','180 '])
    fig.write_image(results_folder / f'ori_{ref_index}ref_{n_neurons}avg{suffix}{_overlay_tag(overlay_params_df)}.pdf', scale=1, engine='kaleido')


def plot_classification_errors(model_df, params_df, project_root, C0=0.001, height=150, width=680, ymin=0.65,
                               color_pts='rgba(140, 140, 140, 1)', color_err='rgba(169, 169, 169, 1.0)',
                               equivariant=False, model=False, symb='circle', show_mice=True,
                               results_folder=None, overlay_params_df=None, alpha_faint=0.4, return_df=False):
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
    C0 : float, default=0.001
        SVM regularization parameter
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    ymin : float, default=0.65
        minimum y-axis value
    color_pts : str, default='rgba(140, 140, 140, 1)'
        color for data points
    color_err : str, default='rgba(169, 169, 169, 1.0)'
        color for error bars
    equivariant : bool, default=False
        whether to use equivariant analysis
    model : bool, default=False
        whether to plot decoder from projected model predictions or projected recorded responses
    symb : str, default='circle'
        symbol for data points
    show_mice : bool, default=True
        whether to show individual mouse results separately
    results_folder : str or Path or None, default=None
        output folder for saved files; defaults to project_root / 'results'

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results' / 'svm_{C0}C_{n_neurons}avg{suffix}.pdf'
    """
    if overlay_params_df is not None:   #N-overlay figures show area means only
        show_mice = False
    n_neurons = int(params_df.loc['n_neurons'].value)
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
        if model:
            suffix += '_model'
    else:
        suffix = ''
    #non-gaussian noise models read/write suffixed files so they do not clobber the
    #default gaussian caches and figures
    noise_model = params_df.loc['noise_model'].value
    suffix += '' if noise_model == 'gaussian' else f'_{noise_model}'
    results_folder = _resolve_results_folder(project_root, results_folder, 'results')

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1.1, 'LM': 2.1, 'RL': 3.1, 'AL': 4.1, \
                      'TX108_V1': 0.95, 'TX108_LM': 1.95, 'TX108_RL': 2.95, 'TX108_AL': 3.95, \
                      'TX109_V1': 1, 'TX109_LM': 2, 'TX109_RL': 3, 'TX109_AL': 4, \
                      'TX111_V1': 1.05, 'TX111_LM': 2.05, 'TX111_RL': 3.05, 'TX111_AL': 4.05, \
                      'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9, \
                      }

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

    # Check if there are no TX models, adjust pixels position
    if svm_df[svm_df.model.str.startswith('TX')].empty:
        area_to_x_dict['pixels'] = 4

    results_df = svm_df[['model', 'area', 'acc test']].copy()
    results_df['score'] = 1 - results_df['acc test']
    results_df = compute_significance(results_df, alternative='greater')
    results_df.to_csv(results_folder / f'svm_significance{suffix}.csv', index=False)

    #show each mouse result separately
    if show_mice:
        svm_mouse_df = svm_df[svm_df.model.str.startswith('TX')].copy()
        svm_mouse_df['area'] = svm_mouse_df['model'].str[:5] + '_' + svm_mouse_df['area']
        mouse_plot_df = svm_mouse_df.groupby(['area'])['acc test'].mean().reset_index()
        mouse_plot_df['erry'] = svm_mouse_df.groupby(['area'])['acc test'].std().values
        mouse_plot_df['erry'] = mouse_plot_df['erry']/np.sqrt(svm_mouse_df.groupby(['area'])['acc test'].count().values)
        mouse_plot_df['color'] = 'rgba(180, 180, 180, 1.0)'
        mouse_plot_df['errcolor'] = 'rgba(209, 209, 209, 1.0)'
        mouse_plot_df['symbol'] = 'diamond'
        mouse_plot_df['size'] = 5

    #compute mean and sem for each area
    svm_plot_df = svm_df.groupby(['area'])['acc test'].mean().reset_index()
    svm_plot_df['erry'] = svm_df.groupby(['area'])['acc test'].std().values
    svm_plot_df['erry'] = svm_plot_df['erry']/np.sqrt(svm_df.groupby(['area'])['acc test'].count().values)
    svm_plot_df['color'] = color_pts
    svm_plot_df['errcolor'] = color_err
    svm_plot_df['symbol'] = 'diamond'
    svm_plot_df['size'] = 8

    #add individual mouse results
    if show_mice:
        svm_plot_df = pd.concat([svm_plot_df, mouse_plot_df[['area','acc test','erry','color','errcolor','symbol','size']]], axis=0).reset_index(drop=True)

    #compute other variables for plotting
    svm_plot_df['y'] = 1-svm_plot_df['acc test']
    svm_plot_df['x'] = [area_to_x_dict[area] for area in svm_plot_df['area']]
    svm_plot_df['xtick'] = [area_to_x_dict[area] for area in svm_plot_df['area']]
    svm_plot_df.sort_values('x', inplace=True)
    svm_plot_df['errx'] = 0

    if svm_plot_df.empty:
        print(f'No data found.')
        return
    if return_df:
        return svm_plot_df
    if overlay_params_df is not None:
        #semi-transparent layer at the other neuron count (same x, area means only), drawn
        #first so the opaque params_df layer sits on top (SI N = 250 vs N = max figures)
        over_df = plot_classification_errors(model_df, overlay_params_df, project_root, C0=C0, equivariant=equivariant,
                                             model=model, show_mice=False, results_folder=results_folder, return_df=True)
        svm_plot_df = pd.concat([_faint(over_df, alpha_faint), svm_plot_df], ignore_index=True)
    fig = plot_points_w_errorbars(svm_plot_df, y_dp=[1-1/91], sz=8, height=height, width=width, alpha=.6, symb='diamond')
    fig.update_layout(yaxis_title='decoding error (1)')
    tick_df = svm_plot_df[~svm_plot_df['area'].str.startswith('TX')]
    fig.update_xaxes(title_standoff=0, tickvals=tick_df['xtick'].unique(), ticktext=tick_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[ymin,1.03], tickvals=[0,0.5,0.7,1], ticktext=['0','0.5','0.7','1'])
    fig.write_image(results_folder / f'svm_{C0}C_{n_neurons}avg{suffix}{_overlay_tag(overlay_params_df)}.pdf', scale=1, engine='kaleido')


def plot_classification_errors_acrossstim(model_df, params_df, project_root, C0=0.001, height=150, width=680, 
                               color_scenes='rgba(0, 104, 56, 1.0)', color_textures='rgba(141, 198, 63, 1.0)',
                               equivariant=False, model=False, symb='circle'):
    """
    Plot classification error across layers and stimuli.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    C0 : float, default=0.001
        SVM regularization parameter
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_scenes : str, default='rgba(0, 104, 56, 1.0)'
        color for scene data points
    color_textures : str, default='rgba(141, 198, 63, 1.0)'
        color for texture data points
    equivariant : bool, default=False
        whether to use equivariant analysis
    model : bool, default=False
        whether to plot decoder from projected model predictions or projected recorded responses
    symb : str, default='circle'
        symbol for data points

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results_extended' / 'svm_stim_{C0}C_{n_neurons}avg{suffix}.pdf'
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
        if model:
            suffix += '_model'
    else:
        suffix = ''
    results_folder = project_root / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1, 'LM': 2, 'RL': 3, 'AL': 4, \
                      'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9}

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

    #split by stim
    scene_df = svm_df[(svm_df['class']<45)|(svm_df['class']==90)].copy()
    text_df = svm_df[(svm_df['class']>=45)&(svm_df['class']!=90)].copy()

    #compute mean and sem for each area
    scene_plot_df = scene_df.groupby(['area'])['acc test'].mean().reset_index()
    scene_plot_df['erry'] = scene_df.groupby(['area'])['acc test'].std().values
    scene_plot_df['erry'] = scene_plot_df['erry']/np.sqrt(scene_df.groupby(['area'])['acc test'].count().values)
    scene_plot_df['color'] = color_scenes
    scene_plot_df['errcolor'] = color_scenes
    scene_plot_df['y'] = 1-scene_plot_df['acc test']
    scene_plot_df['x'] = [area_to_x_dict[area]-0.1 for area in scene_plot_df['area']]
    scene_plot_df['xtick'] = [area_to_x_dict[area] for area in scene_plot_df['area']]
    scene_plot_df.sort_values('x', inplace=True)
    scene_plot_df['errx'] = 0

    text_plot_df = text_df.groupby(['area'])['acc test'].mean().reset_index()
    text_plot_df['erry'] = text_df.groupby(['area'])['acc test'].std().values
    text_plot_df['erry'] = text_plot_df['erry']/np.sqrt(text_df.groupby(['area'])['acc test'].count().values)
    text_plot_df['color'] = color_textures
    text_plot_df['errcolor'] = color_textures
    text_plot_df['y'] = 1-text_plot_df['acc test']
    text_plot_df['x'] = [area_to_x_dict[area]+0.1 for area in text_plot_df['area']]
    text_plot_df['xtick'] = [area_to_x_dict[area] for area in text_plot_df['area']]
    text_plot_df.sort_values('x', inplace=True)
    text_plot_df['errx'] = 0

    svm_plot_df = pd.concat([scene_plot_df, text_plot_df], axis=0).reset_index(drop=True)
    
    # Check if there are no TX models, adjust pixels position
    if svm_df[svm_df.model.str.startswith('TX')].empty:
        area_to_x_dict['pixels'] = 4
        
    if svm_plot_df.empty:
        print(f'No data found.')
        return
    fig = plot_points_w_errorbars(svm_plot_df, y_dp=[1-1/91], sz=8, height=height, width=width, alpha=.6, symb=symb)
    fig.update_layout(yaxis_title='decoding error (1)')
    fig.update_xaxes(title_standoff=0, tickvals=svm_plot_df['xtick'].unique(), ticktext=svm_plot_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[0.5,1.03], tickvals=[0,0.5,0.75,1], ticktext=['0','0.5','0.75','1'])
    fig.write_image(results_folder / f'svm_stim_{C0}C_{n_neurons}avg{suffix}.pdf', scale=1, engine='kaleido')


def plot_orientation_errors_acrossstim(model_df, params_df, project_root, ref_index=0, height=150, width=680, 
                               color_scenes='rgba(0, 104, 56, 1.0)', color_textures='rgba(141, 198, 63, 1.0)',
                            equivariant=False, model=False, symb='circle', ymin=0):
    """
    Plot orientation decoding error across layers and stimuli.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    ref_index : int, default=0
        reference orientation index for alignment
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_scenes : str, default='rgba(0, 104, 56, 1.0)'
        color for scene data points
    color_textures : str, default='rgba(141, 198, 63, 1.0)'
        color for texture data points
    equivariant : bool, default=False
        whether to use equivariant analysis
    model : bool, default=False
        whether to plot decoder from projected model predictions or projected recorded responses
    symb : str, default='circle'
        symbol for data points
    ymin : int, default=0
        minimum y-axis value

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results_extended' / 'ori_stim_{ref_index}ref_{n_neurons}avg{suffix}.pdf'
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
        if model:
            suffix += '_model'
    else:
        suffix = ''
    results_folder = project_root / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1, 'LM': 2, 'RL': 3, 'AL': 4, \
                      'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9}
    
    #collect results across layers
    results_df = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        ori_f = cache_folder / f'{layer_name}_ori2x_decoder_{ref_index}ref_{n_neurons}avg{suffix}.csv'
        if not ori_f.exists():
            continue
        tmp_df = pd.read_csv(ori_f)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        results_df = pd.concat([results_df, tmp_df], axis=0)

    #split by stim
    scene_df = results_df[(results_df['class']<45)|(results_df['class']==90)].copy()
    text_df = results_df[(results_df['class']>=45)&(results_df['class']!=90)].copy()

    #compute mean and sem for each area
    scene_plot_df = scene_df.groupby(['area'])['res'].mean().reset_index()
    scene_plot_df['erry'] = scene_df.groupby(['area'])['res'].std().reset_index()['res']
    scene_plot_df['erry'] = scene_plot_df['erry']/np.sqrt(scene_df.groupby(['area'])['res'].count().values)
    scene_plot_df['color'] = color_scenes
    scene_plot_df['errcolor'] = color_scenes
    scene_plot_df['y'] = scene_plot_df['res']
    scene_plot_df['x'] = [area_to_x_dict[area]-0.1 for area in scene_plot_df['area']]
    scene_plot_df['xtick'] = [area_to_x_dict[area] for area in scene_plot_df['area']]
    scene_plot_df.sort_values('x', inplace=True)
    scene_plot_df['errx'] = 0

    text_plot_df = text_df.groupby(['area'])['res'].mean().reset_index()
    text_plot_df['erry'] = text_df.groupby(['area'])['res'].std().reset_index()['res']
    text_plot_df['erry'] = text_plot_df['erry']/np.sqrt(text_df.groupby(['area'])['res'].count().values)
    text_plot_df['color'] = color_textures
    text_plot_df['errcolor'] = color_textures
    text_plot_df['y'] = text_plot_df['res']
    text_plot_df['x'] = [area_to_x_dict[area]+0.1 for area in text_plot_df['area']]
    text_plot_df['xtick'] = [area_to_x_dict[area] for area in text_plot_df['area']]
    text_plot_df.sort_values('x', inplace=True)
    text_plot_df['errx'] = 0

    plot_df = pd.concat([scene_plot_df, text_plot_df], axis=0).reset_index(drop=True)
    
    # Check if there are no TX models, adjust pixels position
    if results_df[results_df.model.str.startswith('TX')].empty:
        area_to_x_dict['pixels'] = 4
        
    if plot_df.empty:
        print(f'No data found.')
        return
    fig = plot_points_w_errorbars(plot_df, y_dp=[90], sz=8, height=height, width=width, symb=symb)
    fig.update_layout(yaxis_title='absolute angle estimation error (deg)')
    fig.update_xaxes(title_standoff=0, tickvals=plot_df['xtick'].unique(), ticktext=plot_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[ymin,95], tickvals=[0,45,90,180], ticktext=['0 ','45 ','90 ','180 '])
    fig.write_image(results_folder / f'ori_stim_{ref_index}ref_{n_neurons}avg{suffix}.pdf', scale=1, engine='kaleido')


def plot_classification_errors_acrossC(model_df, params_df, project_root, color_dict, height=150, width=680, ymin=0.65,
                               color_err='rgba(169, 169, 169, 1.0)', equivariant=False, symb='circle'):
    """
    Plot classification error across layers for multiple values of C.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    color_dict : dict
        mapping of C values to colors
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    ymin : float, default=0.65
        minimum y-axis value
    color_err : str, default='rgba(169, 169, 169, 1.0)'
        color for error bars
    equivariant : bool, default=False
        whether to use equivariant analysis
    symb : str, default='circle'
        symbol for data points

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results_extended' / 'svm_Crange_{n_neurons}avg{suffix}.pdf'
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
    else:
        suffix = ''
    results_folder = project_root / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)

    #map layers to x-axis positions
    area_to_x_dict = {'pixels': 0, 'V1': 1, 'LM': 2, 'RL': 3, 'AL': 4, 'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9}

    #map C values to colors
    color_range = list(color_dict.values())
    color_to_dx_dict = {f'{color_range[0]}': -0.2, f'{color_range[1]}': -0.1, f'{color_range[2]}': 0, f'{color_range[3]}': 0.1, f'{color_range[4]}': 0.2}

    #collect results across layers
    svm_df = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        class_f = cache_folder / f'{layer_name}_class2x_decoder_{n_neurons}avg_Crange{suffix}.csv'
        if not class_f.exists():
          print(f'No classification data for {model_name} {layer_name} with {n_neurons} neurons')
          continue
        tmp_df = pd.read_csv(class_f)
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name

        for C1 in color_dict.keys():
            tmp_df_C = tmp_df[np.abs(tmp_df['C']-float(C1))<0.01*float(C1)]
            tmp_df_C['color'] = color_dict[C1]
            svm_df = pd.concat([svm_df, tmp_df_C], axis=0)

    # Check if there are no TX models, adjust pixels position
    if svm_df[svm_df.model.str.startswith('TX')].empty:
        area_to_x_dict['pixels'] = 4

    #compute mean and sem for each area
    svm_plot_df = svm_df.groupby(['area','color'])['acc test'].mean().reset_index()

    #compute other variables for plotting
    svm_plot_df['y'] = 1-svm_plot_df['acc test']
    svm_plot_df['erry'] = svm_df.groupby(['area','color'])['acc test'].std().reset_index()['acc test']
    svm_plot_df['erry'] = svm_plot_df['erry']/np.sqrt(svm_df.groupby(['area','color'])['acc test'].count().values)
    svm_plot_df['x'] = [area_to_x_dict[area] + color_to_dx_dict[color] for area, color in zip(svm_plot_df['area'], svm_plot_df['color'])]
    svm_plot_df['xtick'] = [area_to_x_dict[area] for area in svm_plot_df['area']]
    svm_plot_df.sort_values('x', inplace=True)
    svm_plot_df['errx'] = 0
    svm_plot_df['errcolor'] = color_err

    if svm_plot_df.empty:
        print(f'No data found.')
        return
    fig = plot_points_w_errorbars(svm_plot_df, y_dp=[1-1/91], sz=8, height=height, width=width, alpha=.6, symb=symb)
    fig.update_layout(yaxis_title='classification error (1)')
    fig.update_xaxes(title_standoff=0, tickvals=svm_plot_df['xtick'].unique(), ticktext=svm_plot_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[ymin,1.03], tickvals=[0,0.5,0.7,1], ticktext=['0','0.5','0.7','1'])
    fig.write_image(results_folder / f'svm_Crange_{n_neurons}avg{suffix}.pdf', scale=1, engine='kaleido')


def saturation_cap(area_df, value_col='y', atol=1e-6):
    """
    Find the number of neurons at which an area runs out of neurons.

    The decoder caps the requested neuron count at the area's actual neuron count
    (see decoder_functions.py), so the decoding value stops changing once N exceeds
    that count: every larger N reuses all available neurons and returns the identical
    (all-neuron) value. This returns the smallest N at which the area's value first
    reaches and then keeps that all-neuron value, i.e. where the curve should stop.
    For areas that never saturate within the sweep this returns the largest N.

    Parameters
    ----------
    area_df : pd.DataFrame
        rows for a single area, with columns 'N' and `value_col`
    value_col : str, default='y'
        column holding the plotted decoding value
    atol : float, default=1e-6
        absolute tolerance for matching the all-neuron value (which is exactly
        reproduced for every N above the neuron count)

    Returns
    -------
    cap : int
        smallest N belonging to the constant (all-neuron) tail of the curve
    """
    area_df = area_df.sort_values('N')
    Ns = area_df['N'].values
    vals = area_df[value_col].values
    v_top = vals[-1]  # value using all available neurons (largest N)
    cap = Ns[-1]
    for N, v in zip(Ns[::-1], vals[::-1]):
        if np.isclose(v, v_top, atol=atol, rtol=0):
            cap = N
        else:
            break
    return cap


def plot_classification_errors_acrossN(model_df, params_df, project_root, color_dict, C0=0.001, height=150, width=680,
                               color_err='rgba(169, 169, 169, 1.0)', equivariant=False, symb='circle'):
    """
    Plot classification error across layers for multiple values of N (number of neurons).

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    color_dict : dict
        mapping of N values to colors
    C0 : float, default=0.001
        SVM regularization parameter
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_err : str, default='rgba(169, 169, 169, 1.0)'
        color for error bars
    equivariant : bool, default=False
        whether to use equivariant analysis
    symb : str, default='circle'
        symbol for data points

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results_extended' / 'svm_{C0}C_Navg{suffix}.pdf'
    """
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
    else:
        suffix = ''
    results_folder = project_root / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)

    #collect results across neurons
    svm_df1 = pd.DataFrame()
    svm_df2 = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        files = os.listdir(cache_folder)
        neuron_files = []
        for f in files:
            if f.startswith(f'{layer_name}_class2x_decoder') and (f.endswith(f'avg_{C0}C{suffix}.csv')):
                try:
                    fstrings = f.split("_")
                    n_neurons = int(next(s for s in fstrings if 'avg' in s).replace('.csv', '')[:-3])
                    neuron_files.append((n_neurons, f))
                except ValueError:
                    continue        

        for n_neurons0, f in sorted(neuron_files):
            class_f = cache_folder / f
            tmp_df = pd.read_csv(class_f)     
            tmp_df['N'] = n_neurons0
            tmp_df['color'] = color_dict[layer_name]
            tmp_df['model'] = model_name
            tmp_df['area'] = layer_name
            if tmp_df['model'].str.startswith('TX').any() | (tmp_df['model']=='pixels').any():
                svm_df1 = pd.concat([svm_df1, tmp_df], axis=0)
            else:
                svm_df2 = pd.concat([svm_df2, tmp_df], axis=0)
    N_vals = np.sort(svm_df1['N'].unique())
    N_max = N_vals[-1]
    # the largest-N point is detached and drawn one step beyond the largest regular N value
    x_max = 2 * N_vals[-2] - N_vals[-3]

    #compute mean and sem for each area
    ct = 0
    for svm_df in [svm_df1, svm_df2]:
        svm_plot_df = svm_df.groupby(['N','color'])['acc test'].mean().reset_index()
        svm_plot_df['y'] = 1-svm_plot_df['acc test']
        svm_plot_df['erry'] = svm_df.groupby(['N','color'])['acc test'].std().reset_index()['acc test']
        svm_plot_df['erry'] = svm_plot_df['erry']/np.sqrt(svm_df.groupby(['N','color'])['acc test'].count().values)
        svm_plot_df['x'] = svm_plot_df['N']
        svm_plot_df.sort_values('x', inplace=True)
        svm_plot_df['errx'] = 0
        svm_plot_df['errcolor'] = color_err

        if svm_plot_df.empty:
            print(f'No data found.')
            return
        color_all = svm_plot_df['color'].unique()
        fig = None
        for color_i0 in color_all:
            #each curve stops where the area runs out of neurons (saturates)
            area_df = svm_plot_df[svm_plot_df['color']==color_i0]
            cap = saturation_cap(area_df)
            plot_df = area_df[(area_df['N']<N_max) & (area_df['N']<=cap)]
            fig0 = plot_points_w_errorbars(plot_df, lines=True, y_dp=[1-1/91], sz=8, height=height, width=width, alpha=1.0, symb=symb)
            if fig is None:
                fig = fig0
            else:
                fig.add_trace(fig0.data[0])
        for color_i0 in color_all:
            plot_df = svm_plot_df[(svm_plot_df['color']==color_i0) & (svm_plot_df['N']==N_max)].copy()
            if plot_df.empty:
                continue
            plot_df['x'] = x_max
            fig0 = plot_points_w_errorbars(plot_df, lines=False, y_dp=[1-1/91], sz=8, height=height, width=width, alpha=1.0, symb=symb)
            fig.add_trace(fig0.data[0])
        fig.update_layout(yaxis_title='classification error (1)')
        fig.update_yaxes(title_standoff=0, range=[0.5,1.0], tickvals=[0,0.5,1], ticktext=['0','0.5','1'])
        xticks = np.arange(250, N_vals[-2] + 1, 250)
        # drop ticks that would collide with the 'max' label, but always keep the largest regular N tick
        xticks = xticks[(x_max - xticks >= 250) | (xticks == N_vals[-2])]
        fig.update_xaxes(title_standoff=0, range=[0,x_max+50], tickvals=list(xticks) + [x_max],
                         ticktext=[str(int(t)) for t in xticks] + ['max'])
        if ct==0:
            fig.write_image(results_folder / f'svm1_{C0}C_Navg{suffix}.pdf', scale=1, engine='kaleido')
        else:
            fig.write_image(results_folder / f'svm2_{C0}C_Navg{suffix}.pdf', scale=1, engine='kaleido')
        ct += 1


def plot_classification_errors_acrossrelori(model_df, params_df, project_root, color_dict, C0=0.001, height=150, width=680, 
                               color_err='rgba(169, 169, 169, 1.0)', equivariant=False, symb='circle'):
    """
    Plot classification error across relative orientation.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    color_dict : dict
        mapping of layer names to colors
    C0 : float, default=0.001
        SVM regularization parameter
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_err : str, default='rgba(169, 169, 169, 1.0)'
        color for error bars
    equivariant : bool, default=False
        whether to use equivariant analysis
    symb : str, default='circle'
        symbol for data points

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results_extended' / 'svm_relori_{suffix}.pdf'
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
    else:
        suffix = ''
    results_folder = project_root / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)

    #collect results across neurons
    svm_df1 = pd.DataFrame()
    svm_df2 = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        svm_f = cache_folder / f'{layer_name}_class2x_decoder_{n_neurons}avg_{C0}C{suffix}.csv'
        if not svm_f.exists():
            continue
        tmp_df = pd.read_csv(svm_f)
        tmp_df['color'] = color_dict[layer_name]
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        if tmp_df['model'].str.startswith('TX').any() | (tmp_df['model']=='pixels').any():
            svm_df1 = pd.concat([svm_df1, tmp_df], axis=0)
        else:
            svm_df2 = pd.concat([svm_df2, tmp_df], axis=0)

    #compute mean and sem for each area
    ct = 0
    for svm_df in [svm_df1, svm_df2]:
        svm_plot_df = svm_df.groupby(['ori','color'])['acc test'].mean().reset_index()
        svm_plot_df['y'] = 1-svm_plot_df['acc test']
        svm_plot_df['erry'] = svm_df.groupby(['ori','color'])['acc test'].std().reset_index()['acc test']
        svm_plot_df['erry'] = svm_plot_df['erry']/np.sqrt(svm_df.groupby(['ori','color'])['acc test'].count().values)
        svm_plot_df['x'] = svm_plot_df['ori']
        svm_plot_df.sort_values('x', inplace=True)
        svm_plot_df['errx'] = 0
        svm_plot_df['errcolor'] = color_err

        if svm_plot_df.empty:
            print(f'No data found.')
            return
        color_all = svm_plot_df['color'].unique()
        plot_df = svm_plot_df[(svm_plot_df['color']==color_all[0])]
        fig = plot_points_w_errorbars(plot_df, lines=True, y_dp=[1-1/91], sz=8, height=height, width=width, alpha=1.0, symb=symb)
        for color_i0 in color_all[1:]:
            plot_df = svm_plot_df[(svm_plot_df['color']==color_i0)]
            fig0 = plot_points_w_errorbars(plot_df, lines=True, y_dp=[1-1/91], sz=8, height=height, width=width, alpha=1.0, symb=symb)
            fig.add_trace(fig0.data[0])
        fig.update_layout(yaxis_title='classification error (1)')
        fig.update_yaxes(title_standoff=0, range=[0.5,1.0], tickvals=[0,0.5,1], ticktext=['0','0.5','1'])
        fig.update_xaxes(title_standoff=0, range=[0-.5,plot_df['x'].max()+.5], \
                         tickvals=np.arange(plot_df['x'].max()+1), ticktext=[str(i*24) for i in range(int(plot_df['x'].max())+1)])   
        if ct==0:
            fig.write_image(results_folder / f'svm1_relori_{suffix}.pdf', scale=1, engine='kaleido')
        else:
            fig.write_image(results_folder / f'svm2_relori_{suffix}.pdf', scale=1, engine='kaleido')
        ct += 1


def plot_orientation_errors_acrossN(model_df, params_df, project_root, color_dict, height=150, width=680,
                               color_err='rgba(169, 169, 169, 1.0)', equivariant=False, symb='circle', ymin=40):
    """
    Plot orientation decoding error across layers for multiple values of N (number of neurons).

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    color_dict : dict
        mapping of N values to colors
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_err : str, default='rgba(169, 169, 169, 1.0)'
        color for error bars
    equivariant : bool, default=False
        whether to use equivariant analysis
    symb : str, default='circle'
        symbol for data points
    ymin : float, default=40
        minimum value of the y-axis

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results_extended' / 'ori_Navg{suffix}.pdf'
    """
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
    else:
        suffix = ''
    results_folder = project_root / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)

    #collect results across neurons
    results_df1 = pd.DataFrame()
    results_df2 = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        files = os.listdir(cache_folder)
        neuron_files = []
        for f in files:
            if f.startswith(f'{layer_name}_ori2x_decoder_') and (f.endswith(f'avg{suffix}.csv')):
                try:
                    fstrings = f.split("_")
                    n_neurons = int(next(s for s in fstrings if 'avg' in s).replace('.csv', '')[:-3])
                    neuron_files.append((n_neurons, f))
                except ValueError:
                    continue        

        for n_neurons0, f in sorted(neuron_files):
            class_f = cache_folder / f
            tmp_df = pd.read_csv(class_f)     
            tmp_df['N'] = n_neurons0
            tmp_df['color'] = color_dict[layer_name]
            tmp_df['model'] = model_name
            tmp_df['area'] = layer_name
            if tmp_df['model'].str.startswith('TX').any() | (tmp_df['model']=='pixels').any():
                results_df1 = pd.concat([results_df1, tmp_df], axis=0)
            else:
                results_df2 = pd.concat([results_df2, tmp_df], axis=0)
    N_vals = np.sort(results_df1['N'].unique())
    N_max = N_vals[-1]
    # the largest-N point is detached and drawn one step beyond the largest regular N value
    x_max = 2 * N_vals[-2] - N_vals[-3]

    #compute mean and sem for each area
    ct = 0
    for results_df in [results_df1, results_df2]:
        plot_df_all = results_df.groupby(['N','color'])['res'].mean().reset_index()
        plot_df_all['y'] = plot_df_all['res']
        plot_df_all['erry'] = results_df.groupby(['N','color'])['res'].std().reset_index()['res']
        plot_df_all['erry'] = plot_df_all['erry']/np.sqrt(results_df.groupby(['N','color'])['res'].count().values)
        plot_df_all['x'] = plot_df_all['N']
        plot_df_all.sort_values('x', inplace=True)
        plot_df_all['errx'] = 0
        plot_df_all['errcolor'] = color_err

        if plot_df_all.empty:
            print(f'No data found.')
            return
        color_all = plot_df_all['color'].unique()
        fig = None
        for color_i0 in color_all:
            #each curve stops where the area runs out of neurons (saturates)
            area_df = plot_df_all[plot_df_all['color']==color_i0]
            cap = saturation_cap(area_df)
            plot_df = area_df[(area_df['N']<N_max) & (area_df['N']<=cap)]
            fig0 = plot_points_w_errorbars(plot_df, lines=True, y_dp=[90], sz=8, height=height, width=width, alpha=1.0, symb=symb)
            if fig is None:
                fig = fig0
            else:
                fig.add_trace(fig0.data[0])
        for color_i0 in color_all:
            plot_df = plot_df_all[(plot_df_all['color']==color_i0) & (plot_df_all['N']==N_max)].copy()
            if plot_df.empty:
                continue
            plot_df['x'] = x_max
            fig0 = plot_points_w_errorbars(plot_df, lines=False, y_dp=[90], sz=8, height=height, width=width, alpha=1.0, symb=symb)
            fig.add_trace(fig0.data[0])
        fig.update_layout(yaxis_title='absolute angle estimation error (deg)')
        fig.update_yaxes(title_standoff=0, range=[ymin,95], tickvals=[0,45,90,180], ticktext=['0 ','45 ','90 ','180 '])
        xticks = np.arange(250, N_vals[-2] + 1, 250)
        # drop ticks that would collide with the 'max' label, but always keep the largest regular N tick
        xticks = xticks[(x_max - xticks >= 250) | (xticks == N_vals[-2])]
        fig.update_xaxes(title_standoff=0, range=[0,x_max+50], tickvals=list(xticks) + [x_max],
                         ticktext=[str(int(t)) for t in xticks] + ['max'])
        if ct==0:
            fig.write_image(results_folder / f'ori1_Navg{suffix}.pdf', scale=1, engine='kaleido')
        else:
            fig.write_image(results_folder / f'ori2_Navg{suffix}.pdf', scale=1, engine='kaleido')
        ct += 1


def plot_orientation_errors_acrossrefs(model_df, params_df, project_root, color_dict, height=150, width=680, 
                               color_err='rgba(169, 169, 169, 1.0)', equivariant=False, symb='circle'):
    """
    Plot orientation decoding error across reference orientations.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    color_dict : dict
        mapping of layer names to colors
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_err : str, default='rgba(169, 169, 169, 1.0)'
        color for error bars
    equivariant : bool, default=False
        whether to use equivariant analysis
    symb : str, default='circle'
        symbol for data points

    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results_extended' / 'ori_refs{suffix}.pdf'
    """
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
    else:
        suffix = ''
    results_folder = project_root / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)

    #collect results across neurons
    results_df1 = pd.DataFrame()
    results_df2 = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        files = os.listdir(cache_folder)
        ref_index_list = []
        for f in files:
            if f.startswith(f'{layer_name}_ori2x_decoder_') and ('ref' in f) and f.endswith(f'avg{suffix}.csv'):
                try:
                    fstrings = f.split("_")
                    ref_index = int(next(s for s in fstrings if 'ref' in s)[:-3])
                    ref_index_list.append((ref_index, f))
                except ValueError:
                    continue        

        for ref_index0, f in sorted(ref_index_list):
            class_f = cache_folder / f
            tmp_df = pd.read_csv(class_f)     
            tmp_df['ref'] = ref_index0
            tmp_df['color'] = color_dict[layer_name]
            tmp_df['model'] = model_name
            tmp_df['area'] = layer_name
            if tmp_df['model'].str.startswith('TX').any() | (tmp_df['model']=='pixels').any():
                results_df1 = pd.concat([results_df1, tmp_df], axis=0)
            else:
                results_df2 = pd.concat([results_df2, tmp_df], axis=0)

    #compute mean and sem for each area
    ct = 0
    for results_df in [results_df1, results_df2]:
        plot_df_all = results_df.groupby(['ref','color'])['res'].mean().reset_index()
        plot_df_all['y'] = plot_df_all['res']
        plot_df_all['erry'] = results_df.groupby(['ref','color'])['res'].std().reset_index()['res']
        plot_df_all['erry'] = plot_df_all['erry']/np.sqrt(results_df.groupby(['ref','color'])['res'].count().values)
        plot_df_all['x'] = plot_df_all['ref']
        plot_df_all.sort_values('x', inplace=True)
        plot_df_all['errx'] = 0
        plot_df_all['errcolor'] = color_err

        if plot_df_all.empty:
            print(f'No data found.')
            return
        color_all = plot_df_all['color'].unique()
        plot_df = plot_df_all[(plot_df_all['color']==color_all[0])]
        fig = plot_points_w_errorbars(plot_df, lines=True, y_dp=[90], sz=8, height=height, width=width, alpha=1.0, symb=symb)
        for color_i0 in color_all[1:]:
            plot_df = plot_df_all[(plot_df_all['color']==color_i0)]
            fig0 = plot_points_w_errorbars(plot_df, lines=True, y_dp=[90], sz=8, height=height, width=width, alpha=1.0, symb=symb)
            fig.add_trace(fig0.data[0])
        fig.update_layout(yaxis_title='absolute angle estimation error (deg)')
        fig.update_yaxes(title_standoff=0, range=[40,95], tickvals=[0,45,90,180], ticktext=['0 ','45 ','90 ','180 '])
        fig.update_xaxes(title_standoff=0, range=[-.5,plot_df['x'].max()+.5], \
                         tickvals=np.arange(plot_df['x'].max()+1), ticktext=[str(i*24) for i in range(int(plot_df['x'].max())+1)])   
        if ct==0:
            fig.write_image(results_folder / f'ori1_refs{suffix}.pdf', scale=1, engine='kaleido')
        else:
            fig.write_image(results_folder / f'ori2_refs{suffix}.pdf', scale=1, engine='kaleido')
        ct += 1


def plot_orientation_errors_acrossrelori(model_df, params_df, project_root, color_dict, ref_index=0, height=150, width=680, 
                               color_err='rgba(169, 169, 169, 1.0)', equivariant=False, symb='circle'):
    """
    Plot orientation decoding error across relative orientations.

    Parameters
    ----------
    model_df : pd.DataFrame
        dataframe containing model and layer information
    params_df : pd.DataFrame
        dataframe containing analysis parameters (n_neurons, n_procs)
    project_root : Path
        root directory of the project
    color_dict : dict
        mapping of layer names to colors
    ref_index : int, default=0
        reference orientation index to compute relative orientation (e.g., 0 for 0 deg, 1 for 24 deg, etc.)
    height : int, default=150
        plot height in pixels
    width : int, default=680
        plot width in pixels
    color_err : str, default='rgba(169, 169, 169, 1.0)'
        color for error bars
    equivariant : bool, default=False
        whether to use equivariant analysis
    symb : str, default='circle'
        symbol for data points
    Returns
    -------
    None
        saves plot as PDF files in project_root / 'results_extended' / 'ori_relori{suffix}.pdf'
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    if equivariant:
        d = int(params_df.loc['n_procs'].value)
        suffix = f'_{d}dim'
    else:
        suffix = ''
    results_folder = project_root / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)

    #collect results across neurons
    results_df1 = pd.DataFrame()
    results_df2 = pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        cache_folder = project_root / f'cache' / model_name

        ori_f = cache_folder / f'{layer_name}_ori2x_decoder_{ref_index}ref_{n_neurons}avg{suffix}.csv'
        if not ori_f.exists():
            continue
        tmp_df = pd.read_csv(ori_f)
        tmp_df['color'] = color_dict[layer_name]
        tmp_df['model'] = model_name
        tmp_df['area'] = layer_name
        if tmp_df['model'].str.startswith('TX').any() | (tmp_df['model']=='pixels').any():
            results_df1 = pd.concat([results_df1, tmp_df], axis=0)
        else:
            results_df2 = pd.concat([results_df2, tmp_df], axis=0)

    #compute mean and sem for each area
    ct = 0
    for results_df in [results_df1, results_df2]:
        plot_df_all = results_df.groupby(['ori','color'])['res'].mean().reset_index()
        plot_df_all['y'] = plot_df_all['res']
        plot_df_all['erry'] = results_df.groupby(['ori','color'])['res'].std().reset_index()['res']
        plot_df_all['erry'] = plot_df_all['erry']/np.sqrt(results_df.groupby(['ori','color'])['res'].count().values)
        plot_df_all['x'] = plot_df_all['ori']
        plot_df_all.sort_values('x', inplace=True)
        plot_df_all['errx'] = 0
        plot_df_all['errcolor'] = color_err

        if plot_df_all.empty:
            print(f'No data found.')
            return
        color_all = plot_df_all['color'].unique()
        plot_df = plot_df_all[(plot_df_all['color']==color_all[0])]
        fig = plot_points_w_errorbars(plot_df, lines=True, y_dp=[90], sz=8, height=height, width=width, alpha=1.0, symb=symb)
        for color_i0 in color_all[1:]:
            plot_df = plot_df_all[(plot_df_all['color']==color_i0)]
            fig0 = plot_points_w_errorbars(plot_df, lines=True, y_dp=[90], sz=8, height=height, width=width, alpha=1.0, symb=symb)
            fig.add_trace(fig0.data[0])
        fig.update_layout(yaxis_title='absolute angle estimation error (deg)')
        fig.update_yaxes(title_standoff=0, range=[0,100], tickvals=[0,45,90,180], ticktext=['0 ','45 ','90 ','180 '])
        fig.update_xaxes(title_standoff=0, range=[1-.5,plot_df['x'].max()+.5], \
                         tickvals=np.arange(plot_df['x'].max()+1), ticktext=[str(i*24) for i in range(int(plot_df['x'].max())+1)])   
        if ct==0:
            fig.write_image(results_folder / f'ori1_relori{suffix}.pdf', scale=1, engine='kaleido')
        else:
            fig.write_image(results_folder / f'ori2_relori{suffix}.pdf', scale=1, engine='kaleido')
        ct += 1        


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
    height : int, default=120
        plot height in pixels
    width : int, default=110
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
    mouse_i0 = model_df.groupby('network').head(1).index.values
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
    mouse_i0 = model_df.groupby('network').head(1).index.values
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


def plot_eval(df_plot, ymax=7, sz=10, height=250, width=400):
    """
    Plot eigenvalue scatter plots with horizontal grid lines.

    Parameters
    ----------
    df_plot : pd.DataFrame
        dataframe containing x, y coordinates and color information
    ymax : int, default=7
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
    for i in range(ymax+1):
        fig.add_hline(y=i, line_width=1, line_color="lightgray", opacity=.8, line_dash='dot' )

    fig.add_trace(
        go.Scatter(
            x=df_plot['x']
          , y=df_plot['y']
          , mode='markers'
          , marker={"size": sz, 'color': df_plot['color'], 'line': {'width': 0}}   #no white marker border (default)
          , showlegend=False
        ))
    
    fig.update_layout(
          xaxis_range=[df_plot['x'].min()-1,df_plot['x'].max()+1]
        , yaxis_range=[-0.5,ymax+0.5]
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
        , tickvals=np.array(range(ymax+1))
        , ticktext=[str(i) for i in range(ymax+1)]
        , title_standoff=0
    )

    return fig


def _faint(df, alpha):
    """Copy of a plot_df with its marker colours at opacity alpha: the semi-transparent layer of
    the N-overlay figures (opaque params_df layer on top of a faint overlay_params_df layer, as
    the full-space / subspace decoders of Fig 3c). plot_points_w_errorbars takes the error-bar
    colour of the single trace from the first row, so errcolor is left untouched."""
    out = df.copy()
    out['color'] = [c[:c.rfind(',')] + f', {alpha})' for c in out['color']]
    return out


def _overlay_tag(overlay_params_df):
    """File-name tag of an N-overlay figure ('' without overlay, '_overlay250' for 250 neurons)."""
    return '' if overlay_params_df is None else f"_overlay{int(overlay_params_df.loc['n_neurons'].value)}"


def plot_points_w_errorbars(df_plot, lines=False, y_dp=None, symb='circle', sz=10, height=250, width=400, \
                            highlight_first=False, highlight_last=False, alpha=0.2):
    """
    Plot bar plots and optional line connections. Markers have no border by default
    (marker line width 0); only the highlight_first/highlight_last markers get a thin
    black outline.

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

    marker_symbol = df_plot['symbol'] if 'symbol' in df_plot.columns else symb
    marker_size = df_plot['size'] if 'size' in df_plot.columns else sz

    if lines:
      fig.add_trace(
          go.Scatter(
              x=df_plot['x']
            , y=df_plot['y']
            , error_x=dict(type='data', array=df_plot['errx'], width=2, color=df_plot['errcolor'].values[0][:-4]+f'{alpha})')
            , error_y=dict(type='data', array=df_plot['erry'], width=2, color=df_plot['errcolor'].values[0][:-4]+f'{alpha})')
            , mode='markers+lines'
            , marker={"size": marker_size, 'color': df_plot['color'], 'symbol': marker_symbol, 'line': {'width': 0}}
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
            , marker={"size": marker_size, 'color': df_plot['color'], 'symbol': marker_symbol, 'line': {'width': 0}}
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


def plot_rotation_robustness(results_dict, angles, project_root, height=150, width=680, alpha=0.4,
                             results_folder=None, colors=None, filename='rotation_robustness.pdf',
                             yrange=(0.4, 1.03), showlegend=True):
    """Plot classification error (1 - accuracy) vs rotation angle for multiple models.

    Parameters
    ----------
    results_dict : dict
        {model_key: correct_array} where correct_array is (n_images, n_angles) bool.
    angles : array-like
        Rotation angles in degrees.
    project_root : Path
        Root directory used for the default output path.
    height : int
        Plot height in pixels.
    width : int
        Plot width in pixels.
    alpha : float
        Transparency for lines/error bars.
    results_folder : str or Path or None, default=None
        output folder for saved files; defaults to project_root / 'results_extended'
    colors : dict or None, default=None
        {model_key: 'rgba(...)'} overriding the built-in AlexNet-variant colours; keys not
        present fall back to grey. Used by the equivariant-CNN figure (eqv-c15r red /
        eqv-c15 pink / eqv-c1 black).
    filename : str, default='rotation_robustness.pdf'
        output PDF name inside ``results_folder``.
    yrange : tuple, default=(0.4, 1.03)
        y-axis (classification error) range.
    showlegend : bool, default=True
        whether to draw the per-model legend (the eqv figure passes False).

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    COLORS = colors if colors is not None else {
        'pretrained':   'rgba(100, 100, 100, 1)',
        'random':       'rgba(180, 180, 180, 1)',
        'rot-aug':      'rgba(44, 160, 44, 1)',
        'multi-task':   'rgba(214, 39, 40, 1)',
    }
    angles = np.asarray(angles)

    fig = go.Figure()
    for key, correct in results_dict.items():
        n_images = correct.shape[0]
        err = 1.0 - correct.mean(axis=0)
        sem = correct.std(axis=0) / np.sqrt(n_images)
        color = COLORS.get(key, 'rgba(100, 100, 100, 1)')

        df_plot = pd.DataFrame({
            'x': angles, 'y': err,
            'errx': np.zeros(len(angles)), 'erry': sem,
            'color': color, 'errcolor': color,
        })
        fig_i = plot_points_w_errorbars(df_plot, lines=True, sz=5, height=height, width=width, alpha=alpha)
        trace = fig_i.data[0]
        faint = color[:-2] + str(alpha) + ')'
        trace.marker.color = faint
        trace.line.color = faint
        # match the error bars to the line colour (plot_points_w_errorbars' own [:-4] strip
        # mangles rgba(..., 1) strings into plotly's navy default, so override it here)
        trace.error_y.color = faint
        trace.error_x.color = faint
        trace.name = key
        trace.showlegend = showlegend
        fig.add_trace(trace)

    fig.update_layout(
        height=height, width=width,
        showlegend=showlegend,
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(255,255,255,0)',
        font={'size': 11, 'family': 'arial', 'color': 'black'},
        margin=dict(l=20, r=20, b=20, t=20, pad=0),
        yaxis_title='classification error (1)',
    )
    fig.update_yaxes(title_standoff=0, range=list(yrange),
                     showline=True, showgrid=False, linewidth=0.5,
                     linecolor='black', mirror=True, ticks='outside',
                     ticklen=2, tickwidth=0.5,
                     tickfont={'size': 10, 'family': 'arial', 'color': 'black'})
    fig.update_xaxes(title='rotation angle (deg)', title_standoff=0,
                     tickvals=list(angles), ticktext=[str(a) for a in angles],
                     showline=True, showgrid=False, linewidth=0.5,
                     linecolor='black', mirror=True, ticks='outside',
                     ticklen=2, tickwidth=0.5,
                     tickfont={'size': 10, 'family': 'arial', 'color': 'black'})

    results_folder = _resolve_results_folder(project_root, results_folder, 'results_extended')
    fig.write_image(results_folder / filename, scale=1, engine='kaleido')
    return fig


def plot_orientation_robustness(results_dict, angles, project_root, height=150, width=680, alpha=0.4,
                                results_folder=None):
    """Plot orientation estimation error vs rotation angle for multiple models.

    Parameters
    ----------
    results_dict : dict
        {model_key: errors_array} where errors_array is (n_images, n_angles) in degrees.
    angles : array-like
        Rotation angles in degrees.
    project_root : Path
        Root directory used for the default output path.
    height : int
        Plot height in pixels.
    width : int
        Plot width in pixels.
    alpha : float
        Transparency for lines/error bars.
    results_folder : str or Path or None, default=None
        output folder for saved files; defaults to project_root / 'results_extended'

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    COLORS = {
        'random':       'rgba(180, 180, 180, 1)',
        'ori':          'rgba(227, 119, 194, 1)',
        'multi-task':   'rgba(214, 39, 40, 1)',
    }
    angles = np.asarray(angles)

    fig = go.Figure()
    for key, errors in results_dict.items():
        n_images = errors.shape[0]
        mean_err = errors.mean(axis=0)
        sem = errors.std(axis=0) / np.sqrt(n_images)
        color = COLORS.get(key, 'rgba(100, 100, 100, 1)')

        df_plot = pd.DataFrame({
            'x': angles, 'y': mean_err,
            'errx': np.zeros(len(angles)), 'erry': sem,
            'color': color, 'errcolor': color,
        })
        fig_i = plot_points_w_errorbars(df_plot, lines=True, sz=5, height=height, width=width, alpha=alpha)
        trace = fig_i.data[0]
        faint = color[:-2] + str(alpha) + ')'
        trace.marker.color = faint
        trace.line.color = faint
        trace.error_y.color = faint
        trace.error_x.color = faint
        trace.name = key
        trace.showlegend = True
        fig.add_trace(trace)

    fig.update_layout(
        height=height, width=width,
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(255,255,255,0)',
        font={'size': 11, 'family': 'arial', 'color': 'black'},
        margin=dict(l=20, r=20, b=20, t=20, pad=0),
        yaxis_title='orientation error (deg)',
    )
    fig.update_yaxes(title_standoff=0, range=[0, 180],
                     showline=True, showgrid=False, linewidth=0.5,
                     linecolor='black', mirror=True, ticks='outside',
                     ticklen=2, tickwidth=0.5,
                     tickfont={'size': 10, 'family': 'arial', 'color': 'black'})
    fig.update_xaxes(title='rotation angle (deg)', title_standoff=0,
                     tickvals=list(angles), ticktext=[str(a) for a in angles],
                     showline=True, showgrid=False, linewidth=0.5,
                     linecolor='black', mirror=True, ticks='outside',
                     ticklen=2, tickwidth=0.5,
                     tickfont={'size': 10, 'family': 'arial', 'color': 'black'})

    results_folder = _resolve_results_folder(project_root, results_folder, 'results_extended')
    fig.write_image(results_folder / 'orientation_robustness.pdf', scale=1, engine='kaleido')
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
        fig.update_yaxes(autorange="reversed", row=7, col=col)
    
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


def plot_hist_lines(df, val, col, hist_bins, color_df, height=400, width=500, norm=True):
    """
    Plot histogram as line traces, one per group.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing the data
    val : str
        column name for the values to histogram
    col : str
        column name for grouping
    hist_bins : np.ndarray
        bin edges for histogram
    color_df : pd.DataFrame
        dataframe indexed by group with a 'color' column
    height : int, default=400
        plot height in pixels
    width : int, default=500
        plot width in pixels
    norm : bool, default=True
        whether to normalize histogram counts to sum to 1

    Returns
    -------
    fig : plotly.graph_objects.Figure
        plotly figure object
    """
    all_groups = df[col].unique()

    fig = go.Figure()
    for group in all_groups:
        sub_df = df[df[col]==group]
        hist_counts, _ = np.histogram(sub_df[val], bins=hist_bins)
        if norm:
            hist_counts = hist_counts / hist_counts.sum()
        hist_df = pd.DataFrame({'bin': (hist_bins[:-1]+hist_bins[1:])/2, 'count': hist_counts})
        fig.add_trace(go.Scatter(x=hist_df['bin'], y=hist_df['count'], mode='markers+lines', name=str(group), \
                                marker=dict(color=color_df.loc[group,'color'], line=dict(width=0)),   #no white marker border (default)
                                line=dict(color=color_df.loc[group,'color'])))
    fig.update_layout(
        font=dict(family='arial', size=11, color='black')
        , xaxis_title=val
        , xaxis_range=[hist_bins[0], hist_bins[-1]]
        , paper_bgcolor="rgba(255,255,255,255)"
        , plot_bgcolor="rgba(255,255,255,255)"
        , height=height
        , width=width
    )
    fig.update_xaxes(
        showline=True
        , title_standoff=0
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickfont={
              'size': 10
            , 'family': 'arial'
            , 'color': 'black'
          }
    )
    fig.update_yaxes(
        showline=True
        , title_standoff=2
        , linewidth=1
        , linecolor='black'
        , mirror=True
        , ticks='outside'
        , ticklen=5
        , tickfont={
              'size': 10
            , 'family': 'arial'
            , 'color': 'black'
          }
    )

    return fig
