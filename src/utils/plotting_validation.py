"""
Figures and tables of the validation analyses of utils/equivariance_validation.py: pooled
(within-image centered) CC_norm of the equivariance model and of its control models, the
shuffle nulls of the dimension counts, and CC_norm against the fitted dimension.

All functions read the caches written by pooled_ccnorm_all / pooled_ccnorm_subset_all /
pooled_ccnorm_vs_dim_all and the per-subsample var_dim caches, and write into
results_extended (tables as CSV, figures as PDF via plot_points_w_errorbars).
"""
import numpy as np
import pandas as pd

from utils.helper import load_var_dim, band_summary
from utils.plotting_functions import plot_points_w_errorbars, _resolve_results_folder

AREA_TO_X = {'pixels': 0, 'V1': 1.1, 'LM': 2.1, 'RL': 3.1, 'AL': 4.1,
             'TX108_V1': 0.95, 'TX108_LM': 1.95, 'TX108_RL': 2.95, 'TX108_AL': 3.95,
             'TX109_V1': 1, 'TX109_LM': 2, 'TX109_RL': 3, 'TX109_AL': 4,
             'TX111_V1': 1.05, 'TX111_LM': 2.05, 'TX111_RL': 3.05, 'TX111_AL': 4.05,
             'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9}


def _tag(params_df):
    n_neurons = int(params_df.loc['n_neurons'].value)
    d = int(params_df.loc['n_procs'].value)
    noise_model = params_df.loc['noise_model'].value
    suffix = '' if noise_model == 'gaussian' else f'_{noise_model}'
    return f'{n_neurons}avg_{d}dim{suffix}', suffix


def _collect(model_df, params_df, project_root, kind, subset=None):
    """Concatenate {layer}_{kind}_{tag}[_{subset}].csv over rows, with 'model_name' and 'area'
    columns; only the within-image centered scores are kept."""
    tag, _ = _tag(params_df)
    sub = f'_{subset}' if subset else ''
    frames = []
    for i0 in range(model_df.shape[0]):
        model_name = model_df['network'].values[i0]
        layer_name = model_df['layer'].values[i0]
        f = project_root / 'cache' / model_name / f'{layer_name}_{kind}_{tag}{sub}.csv'
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if 'centered' in df.columns:
            df = df[df['centered'] == True]
        df['model_name'] = model_name
        df['area'] = layer_name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _mouse_area(df):
    out = df.copy()
    is_mouse = out['model_name'].str.startswith('TX')
    out['area_m'] = np.where(is_mouse, out['model_name'].str[:5] + '_' + out['area'], out['area'])
    return out


def pooled_ccnorm_table(model_df, params_df, project_root, results_folder=None, stat='median'):
    """
    Per (model_name, area, model, null) summary of the pooled per-neuron scores: median (or
    mean) and its sem over neurons of cc, cc_norm, feve, gain, plus the fraction of neurons
    with a valid ceiling; 'null' is 'data' or the shuffle kind. Written to
    results_extended/ccnorm_pooled_{tag}_table.csv and returned (ED Fig 9b statistics).
    """
    tag, _ = _tag(params_df)
    results_folder = _resolve_results_folder(project_root, results_folder, 'results_extended')
    df = _collect(model_df, params_df, project_root, 'ccnorm_pooled')
    if df.empty:
        print('No pooled caches found.')
        return df
    #shuffle tags 'rot3' -> 'rot'
    df['null'] = df['tag'].str.replace(r'\d+$', '', regex=True)
    keys = ['model_name', 'area', 'model', 'null']
    agg = getattr(df.groupby(keys)[['cc', 'cc_norm', 'feve', 'gain']], stat)()
    agg['n'] = df.groupby(keys)['cc'].count()
    agg['frac_valid_ceiling'] = df.groupby(keys)['cc_max'].apply(lambda x: x.notna().mean())
    agg['sem cc_norm'] = df.groupby(keys)['cc_norm'].std() / np.sqrt(df.groupby(keys)['cc_norm'].count())
    agg = agg.reset_index()
    agg.to_csv(results_folder / f'ccnorm_pooled_{tag}_table.csv', index=False)
    return agg


def plot_ccnorm_pooled_c1c2(model_df, params_df, project_root, stat='median',
                            height=150, width=680, ymax=0.3,
                            color_eqv='rgba(140, 140, 140, 1)', results_folder=None):
    """
    Pooled CC_norm of the equivariance model against its per-rotation control model (both
    grey), displaced by -/+0.15 at each x-tick as the two categories of the dimension figure
    (plot_equivariance_dim); the fully invariant model scores 0 by construction (dotted
    reference line). Per neuron subsample the stat (median) over the pooled per-neuron
    scores; the area point is the mean over subsamples (mice pooled) with the sem ACROSS
    SUBSAMPLES as the error bar (rows with a single subsample get none); individual mice as
    smaller symbols. Reads the {layer}_ccnorm_pooled caches (pooled_ccnorm_all) and writes
    results_extended/cc_pooled_c1c2_centered_{tag}.pdf (ED Fig 9b).
    """
    tag, _ = _tag(params_df)
    results_folder = _resolve_results_folder(project_root, results_folder, 'results_extended')
    df = _mouse_area(_collect(model_df, params_df, project_root, 'ccnorm_pooled'))
    if df.empty:
        print('No pooled caches found.')
        return
    models = ('equivariance', 'per_step')
    df = df[(df['tag'] == 'data') & df['model'].isin(models)]
    #one value per (row of model_params, model, neuron subsample): stat over the pooled
    #per-neuron scores of that subsample
    samp = getattr(df.groupby(['model_name', 'area', 'area_m', 'model', 'sample'])['cc_norm'], stat)().reset_index()

    colors = {'equivariance': color_eqv, 'per_step': color_eqv}
    dx = {'equivariance': -0.15, 'per_step': 0.15}

    def _agg(sub, key):
        g = sub.groupby([key, 'model'])['cc_norm']
        out = g.mean().rename('y').reset_index()
        out['erry'] = (g.std() / np.sqrt(g.count())).values   #sem across subsamples
        return out

    area_df = _agg(samp, 'area')
    area_df['size'] = 8
    mouse_df = _agg(samp[samp['model_name'].str.startswith('TX')], 'area_m').rename(columns={'area_m': 'area'})
    mouse_df['size'] = 5
    plot_df = pd.concat([area_df, mouse_df], ignore_index=True)
    plot_df['color'] = [colors[m] for m in plot_df['model']]
    plot_df['errcolor'] = 'rgba(200, 200, 200, 1.0)'
    plot_df['x'] = [AREA_TO_X[a] + dx[m] for a, m in zip(plot_df['area'], plot_df['model'])]
    plot_df['xtick'] = [AREA_TO_X[a] for a in plot_df['area']]
    plot_df['erry'] = plot_df['erry'].replace(np.nan, 0)
    plot_df['errx'] = 0
    plot_df['symbol'] = 'circle'
    plot_df = plot_df.sort_values('x')

    fig = plot_points_w_errorbars(plot_df, y_dp=[0], sz=8, height=height, width=width, alpha=.6, symb='circle')
    fig.update_layout(yaxis_title='CCnorm (1)')
    tick_df = plot_df[~plot_df['area'].str.startswith('TX')]
    fig.update_xaxes(title_standoff=0, tickvals=tick_df['xtick'].unique(), ticktext=tick_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[-0.02, ymax], tickvals=[0, 0.1, 0.2, 0.3], ticktext=['0', '0.1', '0.2', '0.3'])
    fig.write_image(results_folder / f'cc_pooled_c1c2_centered_{tag}.pdf', scale=1, engine='kaleido')
    return plot_df


def plot_ccnorm_pooled_equiv(model_df, params_df, project_root, stat='median',
                             height=150, width=680, ymax=0.3, mouse_gap_shift=0.10, size_scale=1.0,
                             color_area='rgba(140, 140, 140, 1)', color_err='rgba(169, 169, 169, 1.0)',
                             color_mouse='rgba(180, 180, 180, 1)', color_mouse_err='rgba(209, 209, 209, 1.0)',
                             results_folder=None, subset=None):
    """
    Pooled CC_norm of the equivariance model ONLY (no control models), across areas/layers,
    in the across-layer area/mouse grey style: area grey (size 8), individual mice lighter
    grey (size 5), the same x-tick map (AREA_TO_X), no per-model displacement. Per neuron
    subsample the stat (median) over the pooled per-neuron scores; the area point is the
    mean over subsamples with the sem ACROSS SUBSAMPLES as the error bar (rows with a single
    subsample get none). Reads the {layer}_ccnorm_pooled caches (pooled_ccnorm_all) and
    writes results_extended/cc_pooled_equiv_centered_{tag}[_{subset}].pdf (SI Fig 5f with
    the multiplicative-gain parameters, SI Fig 9 with a subset).

    subset : str or None, default=None
        None: the full-set caches (all 92 held-out images). 'scenes' or 'textures': the
        within-class leave-one-out caches from pooled_ccnorm_subset_all
        ({layer}_ccnorm_pooled_{tag}_{subset}.csv); the figure gets a matching '_{subset}' suffix.
    """
    tag, _ = _tag(params_df)
    results_folder = _resolve_results_folder(project_root, results_folder, 'results_extended')
    df = _mouse_area(_collect(model_df, params_df, project_root, 'ccnorm_pooled', subset=subset))
    if df.empty:
        print('No pooled caches found.')
        return
    df = df[(df['tag'] == 'data') & (df['model'] == 'equivariance')]
    if df.empty:
        print('No equivariance pooled rows found.')
        return
    #one value per (row of model_params, neuron subsample): stat over the pooled per-neuron
    #scores of that subsample
    samp = getattr(df.groupby(['model_name', 'area', 'area_m', 'sample'])['cc_norm'], stat)().reset_index()

    #class-subset figures use the two shades of green of the scene/texture decoding figures
    #(scenes dark green, textures light green): area and mouse markers share the class colour
    #and are told apart by size (8 vs 5); only when the caller kept the default area grey
    _class_color = {'scenes': 'rgba(0, 104, 56, 1)', 'textures': 'rgba(141, 198, 63, 1)'}
    if subset in _class_color and color_area == 'rgba(140, 140, 140, 1)':
        color_area = color_mouse = _class_color[subset]

    def _agg(sub, key):
        g = sub.groupby(key)['cc_norm']
        out = g.mean().rename('y').reset_index()
        out['erry'] = (g.std() / np.sqrt(g.count())).values   #sem across subsamples
        return out

    area_df = _agg(samp, 'area')
    area_df['size'] = 8 * size_scale
    area_df['color'] = color_area
    area_df['errcolor'] = color_err
    mouse_df = _agg(samp[samp['model_name'].str.startswith('TX')], 'area_m').rename(columns={'area_m': 'area'})
    mouse_df['size'] = 5 * size_scale
    mouse_df['color'] = color_mouse
    mouse_df['errcolor'] = color_mouse_err
    plot_df = pd.concat([area_df, mouse_df], ignore_index=True)
    #when there are no mouse rows (e.g. the model-only mult_gain run) move pixels next to the
    #conv layers, so there is no gap where V1..AL would be
    x_map = dict(AREA_TO_X)
    if not samp['model_name'].str.startswith('TX').any():
        x_map['pixels'] = 4
    #mouse markers sit left of their area mean; shift the whole cluster left by mouse_gap_shift
    #so the nearest (right-most) mouse is further from the pooled mean, keeping the inter-mouse
    #spacing unchanged
    def _x(a):
        return x_map[a] - mouse_gap_shift if a.startswith('TX') else x_map[a]
    plot_df['x'] = [_x(a) for a in plot_df['area']]
    plot_df['xtick'] = [x_map[a] for a in plot_df['area']]
    plot_df['erry'] = plot_df['erry'].replace(np.nan, 0)
    plot_df['errx'] = 0
    plot_df['symbol'] = 'circle'
    plot_df = plot_df.sort_values('x')

    #expand the y-axis if the data (e.g. pixels at 4000 neurons, ~0.4) exceeds the default cap
    ytop = float((plot_df['y'] + plot_df['erry']).max())
    if ytop > ymax:
        ymax = float(np.ceil(ytop / 0.05) * 0.05)
    ticks = np.round(np.arange(0, ymax + 1e-9, 0.1), 1)
    fig = plot_points_w_errorbars(plot_df, y_dp=[0], sz=8, height=height, width=width, alpha=.6, symb='circle')
    fig.update_layout(yaxis_title='CCnorm (1)')
    tick_df = plot_df[~plot_df['area'].str.startswith('TX')]
    fig.update_xaxes(title_standoff=0, tickvals=tick_df['xtick'].unique(), ticktext=tick_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[-0.02, ymax], tickvals=ticks, ticktext=[f'{t:g}' for t in ticks])
    name = f'cc_pooled_equiv_centered_{tag}{"_" + subset if subset else ""}.pdf'
    fig.write_image(results_folder / name, scale=1, engine='kaleido')
    return plot_df


def plot_ccnorm_pooled_equiv_overlay(model_df, params_df, project_root, n_list=(4000, 250),
                                     alpha_faint=0.4, stat='median', sz=9.6, show_err=False,
                                     height=150, width=680, ymax=0.3, color='140, 140, 140',
                                     color_err='rgba(169, 169, 169, 1.0)', results_folder=None):
    """
    Overlay of the pooled CC_norm of the equivariance model at two neuron counts on the
    SAME x (AREA_TO_X, no per-n displacement), AREA MEANS ONLY (no individual mouse markers).
    n_list[0] (default 4000) is drawn opaque and on top; the rest (default 250) at
    alpha_faint. Per area the mean over the per-subsample medians, with the sem across
    subsamples as the error bar. Reads the {layer}_ccnorm_pooled caches at each n_neurons;
    writes results_extended/cc_pooled_equiv_overlay_{ns}avg_centered_{d}dim{suffix}.pdf
    (SI Fig 8c).
    """
    d = int(params_df.loc['n_procs'].value)
    noise_model = params_df.loc['noise_model'].value
    suffix = '' if noise_model == 'gaussian' else f'_{noise_model}'
    results_folder = _resolve_results_folder(project_root, results_folder, 'results_extended')

    def _area_means(n):
        p = params_df.copy()
        p.loc['n_neurons', 'value'] = str(int(n))
        df = _mouse_area(_collect(model_df, p, project_root, 'ccnorm_pooled'))
        if df.empty:
            return pd.DataFrame()
        df = df[(df['tag'] == 'data') & (df['model'] == 'equivariance')]
        #per subsample the median over pooled neurons, then the mean/sem across subsamples per area
        samp = getattr(df.groupby(['model_name', 'area', 'area_m', 'sample'])['cc_norm'], stat)().reset_index()
        g = samp.groupby('area')['cc_norm']
        out = g.mean().rename('y').reset_index()
        out['erry'] = (g.std() / np.sqrt(g.count())).values
        out['n'] = int(n)
        return out

    frames = {int(n): _area_means(n) for n in n_list}
    frames = {n: f for n, f in frames.items() if not f.empty}
    if not frames:
        print('No pooled caches found for the requested neuron counts.')
        return
    opaque_n = int(n_list[0])
    #faint layers first, the emphasised (opaque) n_list[0] last so it is drawn on top
    ordered = [n for n in (int(x) for x in n_list) if n in frames and n != opaque_n]
    if opaque_n in frames:
        ordered.append(opaque_n)
    plot_df = pd.concat([frames[n] for n in ordered], ignore_index=True)
    plot_df['color'] = [f'rgba({color}, 1)' if n == opaque_n else f'rgba({color}, {alpha_faint})' for n in plot_df['n']]
    plot_df['size'] = sz
    plot_df['errcolor'] = color_err
    plot_df['x'] = [AREA_TO_X[a] for a in plot_df['area']]
    plot_df['xtick'] = plot_df['x']
    plot_df['errx'] = 0
    plot_df['erry'] = plot_df['erry'].replace(np.nan, 0) if show_err else 0.0

    #expand the y-axis if the data (e.g. pixels at 4000 neurons, ~0.4) exceeds the default cap
    ytop = float((plot_df['y'] + plot_df['erry']).max())
    if ytop > ymax:
        ymax = float(np.ceil(ytop / 0.05) * 0.05)
    ticks = np.round(np.arange(0, ymax + 1e-9, 0.1), 1)
    fig = plot_points_w_errorbars(plot_df, y_dp=[0], sz=sz, height=height, width=width, alpha=.6)
    if not show_err:                       #no whiskers on the overlay
        fig.data[0].error_y.visible = False
        fig.data[0].error_x.visible = False
    fig.update_layout(yaxis_title='CCnorm (1)')
    tick_df = plot_df[~plot_df['area'].str.startswith('TX')]
    fig.update_xaxes(title_standoff=0, tickvals=tick_df['xtick'].unique(), ticktext=tick_df['area'].unique())
    fig.update_yaxes(title_standoff=0, range=[-0.02, ymax], tickvals=ticks, ticktext=[f'{t:g}' for t in ticks])
    ns = '_'.join(str(int(n)) for n in n_list)
    fig.write_image(results_folder / f'cc_pooled_equiv_overlay_{ns}avg_centered_{d}dim{suffix}.pdf', scale=1, engine='kaleido')
    return plot_df


def plot_ccnorm_pooled_vs_dim(model_df, params_df, project_root, stat='median',
                              height=150, width=340, color_err='rgba(169, 169, 169, 1.0)', err_alpha=0.3,
                              ymax=None, results_folder=None, dims=(6, 20, 50, 100, 200)):
    """
    Pooled CC_norm of the equivariance model vs the fitted dimension d (ED Fig 9a): the grey
    shades pixels lightest .. AL black and conv1 .. conv5 the same range, lines+markers via
    plot_points_w_errorbars(lines=True, sz=6) with faint error bars, a log x-axis with ticks at
    the swept dims, y-range/ticks matching cc_pooled_c1c2. Reads
    {layer}_ccnorm_pooled_vsdim_{N}avg{suffix}.csv (pooled_ccnorm_vs_dim_all): the stat
    (median) over neurons per (sample, d) is the per-subsample unit, pooled over subsamples and
    mice per area. Two panels (mouse areas + pixels; AlexNet layers) with a shared y-range.
    Writes results_extended/ccnorm_pooled_vs_dim{1,2}_centered_{N}avg.pdf and the table
    ccnorm_pooled_vs_dim_centered_{N}avg.csv. dims: the fitted dimensions drawn (the published
    panel: 6, 20, 50, 100, 200; a cache swept more finely or further keeps only these).
    """
    n_neurons = int(params_df.loc['n_neurons'].value)
    noise_model = params_df.loc['noise_model'].value
    noise_suffix = '' if noise_model == 'gaussian' else f'_{noise_model}'
    results_folder = _resolve_results_folder(project_root, results_folder, 'results_extended')

    #the grey ramp of the decoding-vs-N figures (plot_extended_figures.ipynb color_dict)
    color_range = ['rgba(211, 211, 211, 1.0)', 'rgba(169, 169, 169, 1.0)', 'rgba(128, 128, 128, 1.0)',
                   'rgba(105, 105, 105, 1.0)', 'rgba(0, 0, 0, 1.0)']
    color_dict = {'pixels': color_range[0], 'V1': color_range[1], 'LM': color_range[2], 'RL': color_range[3],
                  'AL': color_range[4], 'conv1': color_range[0], 'conv2': color_range[1], 'conv3': color_range[2],
                  'conv4': color_range[3], 'conv5': color_range[4]}

    results_df1, results_df2 = pd.DataFrame(), pd.DataFrame()
    for i0 in range(model_df.shape[0]):
        mn = model_df['network'].values[i0]
        ar = model_df['layer'].values[i0]
        f = project_root / 'cache' / mn / f'{ar}_ccnorm_pooled_vsdim_{n_neurons}avg{noise_suffix}.csv'
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if 'centered' in df.columns:
            df = df[df['centered'] == True]
        #show every layer on the shared grid `dims` (a layer swept more finely or further keeps only these)
        df = df[df['dim'].isin(dims)]
        #per (sample, d) the stat over pooled neurons -> one value per subsample
        samp = getattr(df.groupby(['sample', 'dim'])['cc_norm'], stat)().reset_index().rename(columns={'cc_norm': 'cc_mean'})
        samp['color'] = color_dict.get(ar, color_range[-1])
        samp['area'] = ar
        samp['model'] = mn
        if mn.startswith('TX') or (mn == 'pixels'):
            results_df1 = pd.concat([results_df1, samp], axis=0)
        else:
            results_df2 = pd.concat([results_df2, samp], axis=0)
    if results_df1.empty and results_df2.empty:
        print('No pooled vs-dim caches found.')
        return
    pd.concat([results_df1, results_df2], ignore_index=True).to_csv(
        results_folder / f'ccnorm_pooled_vs_dim_centered_{n_neurons}avg.csv', index=False)

    #log x-axis: a sparse tick set within the data range, shared across panels
    _all_dims = [int(v) for rdf in (results_df1, results_df2) if not rdf.empty for v in rdf['dim'].unique()]
    dmin, dmax = min(_all_dims), max(_all_dims)
    xlo, xhi = 5, dmax * 1.4
    dims_all = [t for t in dims if dmin <= t <= dmax]
    if ymax is None:
        ymax = 0.3   #match cc_pooled_c1c2 y-range/ticks

    ct = 0
    for rdf in (results_df1, results_df2):
        if rdf.empty:
            ct += 1
            continue
        plot_df_all = rdf.groupby(['dim', 'color'])['cc_mean'].mean().reset_index()
        plot_df_all['y'] = plot_df_all['cc_mean']
        plot_df_all['erry'] = rdf.groupby(['dim', 'color'])['cc_mean'].std().reset_index()['cc_mean']
        plot_df_all['erry'] = plot_df_all['erry'] / np.sqrt(rdf.groupby(['dim', 'color'])['cc_mean'].count().values)
        plot_df_all['erry'] = plot_df_all['erry'].replace(np.nan, 0)
        plot_df_all['x'] = plot_df_all['dim']
        plot_df_all['errx'] = 0
        plot_df_all['errcolor'] = color_err
        plot_df_all.sort_values('x', inplace=True)

        fig = None
        for color_i0 in plot_df_all['color'].unique():
            plot_df = plot_df_all[plot_df_all['color'] == color_i0]
            fig0 = plot_points_w_errorbars(plot_df, lines=True, y_dp=[0], sz=6, height=height, width=width, alpha=1.0)
            err_rgba = color_err[:-4] + f'{err_alpha})'   #lines opaque, error bars faint
            fig0.data[0].error_y.color = err_rgba
            fig0.data[0].error_x.color = err_rgba
            fig = fig0 if fig is None else fig.add_trace(fig0.data[0])
        fig.update_layout(yaxis_title='CCnorm (1)', xaxis_title='dimension (1)')
        fig.update_yaxes(title_standoff=0, range=[-0.02, ymax],
                         tickvals=[0, 0.1, 0.2, 0.3], ticktext=['0', '0.1', '0.2', '0.3'])
        fig.update_xaxes(title_standoff=0, type='log', range=[np.log10(xlo), np.log10(xhi)],
                         tickvals=dims_all)
        suffix = '1' if ct == 0 else '2'
        fig.write_image(results_folder / f'ccnorm_pooled_vs_dim{suffix}_centered_{n_neurons}avg.pdf', scale=1, engine='kaleido')
        ct += 1
    return pd.concat([results_df1, results_df2], ignore_index=True)


def null_band_table(model_df, params_df, project_root, results_folder=None):
    """
    Data vs null band statistics per (model_name, area): invariant/tuned counts and band
    correlations of the leave-two-out fits (var_dim caches) against their shuffle nulls
    (var_dim_null caches: kinds rot / img / both). Written to
    results_extended/null_bands_{tag}.csv; the 'both' columns are the dashed shuffle-control
    lines of Fig 3b (plotting_functions.plot_equivariance_dim).

    'data q' is the mean over fits (held-out pairs x subsamples). The null columns describe
    the null distribution of that SAME statistic: '{kind} q' is the mean over shuffle
    replicates of the per-replicate mean over fits, '{kind} q q95' / '{kind} q max' its 95th
    percentile / maximum over replicates.
    """
    tag, _ = _tag(params_df)
    results_folder = _resolve_results_folder(project_root, results_folder, 'results_extended')
    rows = []
    for i0 in range(model_df.shape[0]):
        mn = model_df['network'].values[i0]
        ar = model_df['layer'].values[i0]
        cache = project_root / 'cache' / mn
        var_f = cache / f'{ar}_var_dim_{tag}.csv'
        if not var_f.exists():
            continue
        data = band_summary(load_var_dim(var_f))
        rec = {'model_name': mn, 'area': ar}
        for q in ['d_inv', 'd_tun', 'pc test inv', 'pc test tun', 'var test inv', 'var test tun']:
            rec[f'data {q}'] = data[q].mean()
        null_f = cache / f'{ar}_var_dim_null_{tag}.csv'
        if null_f.exists():
            nd = pd.read_csv(null_f)
            for kind, sub in nd.groupby('kind'):
                sub = sub.copy()
                sub['var train'] = np.nan; sub['pc train'] = np.nan
                b = band_summary(sub, keys=['shuffle'])                 #one fit per (shuffle, stim, sample)
                for q in ['d_inv', 'd_tun', 'pc test inv', 'pc test tun', 'var test inv', 'var test tun']:
                    m = b.groupby('shuffle')[q].mean()                  #null of the mean over fits
                    rec[f'{kind} {q}'] = m.mean()
                    rec[f'{kind} {q} q95'] = m.quantile(0.95)
                    rec[f'{kind} {q} max'] = m.max()
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(results_folder / f'null_bands_{tag}.csv', index=False)
    return out
