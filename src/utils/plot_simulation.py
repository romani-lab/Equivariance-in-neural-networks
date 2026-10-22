"""
Figures of the simulations (utils/simulation.py; src/simulations.ipynb).

1. Low-dimensional simulations: per grid axis the estimated invariant/tuned counts, the two
   decoders in the full space and in the estimated subspaces, and the pooled centered CC_norm
   of the equivariance model, as mean +- sem over seeds (ED Fig 8b-c; `summarize`, `plot_axis`),
   and the calibration of CC_norm against the noiseless ground truth (SI Fig 7;
   `plot_calibration`). Reads cache/<CACHE_DIR>, writes results_extended/simulation/.
2. Calibration simulations based on pixels: the pooled centered CC_norm of the equivariance model
   over the (N, d) grid -- vs the fitted dimension d at several N, and its peak over d vs N --
   for the pixels, the AlexNet layers, the pooled mouse areas and the simulated populations
   built from the pixels' equivariant structure (ED Fig 9c-h; `plot_*_vs_d`,
   `plot_*_peak_vs_N`), and for the equivariant CNNs (SI Fig 6f; `plot_eqv_net_peak_vs_N`).
   Reads the vsdim caches, writes results_extended/calibration/ (results_extended_anns/ for SI 6f).
3. Fit of simulations to data: the measured and calibrated generator parameters per population
   (ED Fig 10a; `plot_datafit_params`) and the structure-matched populations through the standard
   pipeline next to the data (ED Fig 10b; `plot_datafit_simulations`); results_extended/data_fit/.

Conventions of 2-3: centered CC_norm, the MEDIAN over pooled neurons per (subsample, d) (as the
paper's pooled-CC_norm figures), then mean +- sem over disjoint neuron subsamples (0-width where
only one subsample exists); plot_points_w_errorbars(lines=True), log x for d and N.
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.simulation import (DEFAULT, grid, CACHE_DIR, CALIBRATION_DIR, DATAFIT_DIR, DATAFIT_CACHE_DIR, SWEPT_IID_CSV,
                              TARGETS, TARGET_ORDER, AREAS, CONVS, MICE, CONV_N_REL, cached_n_levels)
from utils.helper import load_var_dim, band_summary
from utils.plotting_functions import plot_points_w_errorbars

# ----------------------------------------------------------------------------------------
# 1. low-dimensional simulations (ED Fig 8, SI Fig 7)
# ----------------------------------------------------------------------------------------
#axis -> (cells in plotting order, config key on the x axis). `reference` (= DEFAULT) is the
#model-abiding population and the anchor of every axis.
AXES = {
    #ED Fig 8b: fit dimension d, N, noise, angular irregularity xi, equivariant fraction f
    'fitdim': ([f'fitdim_{d}' for d in range(2, 21, 2)], 'd'),
    'N': (['N_4', 'N_8', 'N_16', 'N_50', 'reference', 'N_1000'], 'n_neurons'),   # log x-axis
    'snr': (['snr_0', 'snr_0.5', 'reference', 'snr_2', 'snr_3.9'], 'inv_snr'),
    'xi': (['reference', 'control_C2p_a100', 'control_C2p_a1'], 'c2_alpha'),
    'eqvfrac': (['reference', 'eqvfrac_0.75', 'eqvfrac_0.5', 'eqvfrac_0.25', 'eqvfrac_0.1'], None),
    #ED Fig 8c: true split, epsilon, delta, gain heterogeneity, tuning share, images, trials
    'dims': (['dims_0_8', 'dims_2_6', 'dims_4_4', 'dims_6_2', 'dims_8_0'], 'd_inv'),
    'epsilon': (['epsilon_0.0', 'epsilon_0.4', 'epsilon_0.8', 'reference', 'epsilon_1.5'], 'epsilon'),
    'delta': (['delta_0.0', 'delta_0.3', 'delta_0.63', 'reference'], 'delta'),
    'snr_hetero': (['reference', 'snr_hetero_mild', 'snr_hetero_data'], 'snr_hetero'),
    'vtun': (['reference', 'datamatch_vtun_0.5', 'datamatch_vtun_0.2', 'datamatch_vtun_0.1', 'datamatch_vtun_0.05'], 'var_tun'),
    'images': (['images_12', 'images_23', 'images_46', 'reference'], 'n_images'),
    'trials': (['trials_2', 'trials_4', 'reference'], 'n_trials'),
}
QUANT = ['n_reliable', 'd_inv', 'd_tun', 'true_d_inv', 'true_d_tun',
         'var_test_inv', 'var_test_tun', 'pc_test_inv', 'pc_test_tun',
         'cc_norm_equivariance_c', 'feve_equivariance_c', 'true_r2_equivariance_c', 'true_cc_equivariance_c',
         'gain_equivariance_c', 'cc_norm_per_step_c', 'cc_norm_invariant_c',
         'svm_err', 'svm_err_train', 'svm_conf', 'svm_err_best', 'svm_best_C', 'ori_err', 'ori_r2',
         'svm_err_C0.001', 'svm_err_C0.01', 'svm_err_C0.1', 'svm_err_C1', 'svm_err_C10', 'svm_err_C100',
         'svm_err_eqv', 'svm_err_eqv_train', 'svm_err_eqv_best', 'svm_best_C_eqv', 'ori_err_eqv', 'ori_r2_eqv',
         'svm_err_eqv_C0.001', 'svm_err_eqv_C0.01', 'svm_err_eqv_C0.1', 'svm_err_eqv_C1', 'svm_err_eqv_C10', 'svm_err_eqv_C100']

MAX_SEEDS = 10   # every panel uses seeds 0-9, so the sem bars are comparable across columns

XI_LABELS = {'reference': '0', 'control_C2p_a100': '0.1', 'control_C2p_a1': '0.9'}   # step CV sqrt(14/(15 alpha + 1))


def load_runs(root, max_seeds=MAX_SEEDS):
    """All cached simulation summaries (one row per cell and seed), seeds < max_seeds."""
    files = sorted(glob.glob(str(Path(root) / 'cache' / CACHE_DIR / '*_[0-9]*.csv')))
    files = [f for f in files if not (f.endswith('_fits.csv') or f.endswith('_vsdim.csv'))]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if max_seeds is not None and 'seed' in df.columns:
        df = df[df['seed'] < max_seeds]
    return df


def summarize(df):
    """Per cell the mean, sd and sem over seeds of every quantity, with the cell's configuration."""
    cols = [c for c in QUANT if c in df.columns]
    g = df.groupby('cell')[cols]
    mean = g.mean(); sd = g.std(); cnt = g.count(); n = cnt['d_inv'].rename('n_seeds')
    sem = sd / np.sqrt(cnt)
    mean.columns = [f'{c} mean' for c in cols]; sd.columns = [f'{c} sd' for c in cols]; sem.columns = [f'{c} sem' for c in cols]
    out = pd.concat([n, mean, sd, sem], axis=1).reset_index()
    G = grid()
    for k in ['inv_snr', 'n_neurons', 'n_images', 'n_trials', 'iid_frac', 'epsilon', 'r', 'delta', 'snr_hetero',
              'control', 'c2_alpha', 'd', 'd_inv', 'd_tun', 'var_tun']:
        out[k] = [dict(DEFAULT, **G.get(c, {})).get(k) for c in out['cell']]
    return out


def _write(fig, path):
    """write_image that survives a file locked by a viewer."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(path, scale=1, engine='kaleido')
    except PermissionError:
        print(f'WARNING: could not write {path} (file locked?), skipped', flush=True)


def _labels(axis, cells):
    if axis == 'xi':
        return [XI_LABELS.get(c, c) for c in cells]
    return [c.replace(axis + '_', '').replace('eqvfrac_', '').replace('datamatch_vtun_', '') for c in cells]


#invariant (orange) / tuned (brown) colours shared with the data figures, grey for scalars
COLOR_INV = 'rgba(247, 148, 29, 1)'
COLOR_TUN = 'rgba(139, 94, 60, 1)'
COLOR_GREY = 'rgba(140, 140, 140, 1)'
FONT_SCALE = 1.5    # every panel font is 50 % larger than plot_points_w_errorbars' defaults
#every axis uses width = TICK_PX * n + 40 (l + r = 40 px margins, x-range pinned to [-0.5, n - 0.5]),
#so the pixel gap between adjacent x-ticks is the same on every axis
TICK_PX = 0.8 * (0.8 * 60 * 10 - 40) / 10   # 35.2 px between adjacent x-ticks
X_MARGIN_PX = 40                            # l + r margins of plot_points_w_errorbars
PANEL_HEIGHT = 170
PANEL_TOP = 10


def _fmt(v):
    """tick label for a config value: int/float without trailing zeros ('250', '3.9', '1'), else str."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f'{v:g}'
    return str(v)


def plot_axis(summary, axis, results_folder):
    """
    The four rows of ED Fig 8b-c for one axis: estimated invariant/tuned dimensions
    (simulation_<axis>_dims.pdf), image-identity SVM error at the best C in the full space
    (faint diamonds) and in the estimated invariant subspace (circles; _svm_err.pdf),
    relative-orientation error in the full space and in the estimated tuned subspace
    (_ori_err.pdf), and the pooled centered CC_norm of the equivariance model
    (_cc_norm_equivariance_c.pdf).
    """
    cells, xkey = AXES[axis]
    sub = summary.set_index('cell').reindex([c for c in cells if c in summary['cell'].values]).reset_index()
    if sub.empty:
        return
    #the N axis is drawn on a true log x-scale (markers at their actual n_neurons); every other axis
    #keeps the uniform categorical spacing (xs = 0, 1, 2, ...).
    xlog = (axis == 'N')
    xs = sub[xkey].astype(float).values if xlog else np.arange(len(sub))
    #x-tick labels: the reference cell shows its actual x-value, never the word "reference"
    labels = list(_labels(axis, sub['cell']))
    for i in range(len(labels)):
        if labels[i] == 'reference':
            if xkey is not None and xkey in sub.columns and pd.notna(sub[xkey].iloc[i]):
                labels[i] = _fmt(sub[xkey].iloc[i])
            elif axis == 'eqvfrac' and 'iid_frac' in sub.columns:
                labels[i] = _fmt(1 - float(sub['iid_frac'].iloc[i]))   # equivariant fraction f = 1 - iid_frac
    #uniform x-tick spacing across axes (pinned x-range + width = TICK_PX * n + margins); the log
    #N axis uses the width of 5 cells, widened 10 %
    width = TICK_PX * (5 if xlog else len(sub)) + X_MARGIN_PX
    if xlog:
        width *= 1.1
    #kaleido (0.1.0.post1) floors the pixel width and emits a blank second PDF page whenever the
    #integer width is 2 mod 4; floor explicitly and nudge such widths by 1 px
    width = int(width)
    if width % 4 == 2:
        width += 1
    tick = round(10 * FONT_SCALE)

    #default hline style = the chance lines of the decoding figures: dotted, light grey
    ref_hline_kw = dict(line_width=1, line_color='lightgray', opacity=0.8, line_dash='dot')

    def _restyle(fig, yrange, hline=None, ytickvals=None, hline_kw=None):
        if hline is not None:
            for y in (hline if isinstance(hline, (list, tuple)) else [hline]):
                fig.add_hline(y=y, **(hline_kw or ref_hline_kw))
        fig.update_traces(marker_line_width=0, error_x_width=0, error_y_width=2)
        fig.update_layout(width=width, height=PANEL_HEIGHT, margin_t=PANEL_TOP, font=dict(size=round(11 * FONT_SCALE)))
        if xlog:
            lo, hi = float(np.min(xs)), float(np.max(xs))
            fig.update_xaxes(type='log', tickvals=list(xs), ticktext=labels,
                             title_standoff=0, tickfont=dict(size=tick),
                             range=[np.log10(lo) - 0.08, np.log10(hi) + 0.08])
        else:
            fig.update_xaxes(tickvals=xs, ticktext=labels, title_standoff=0, tickfont=dict(size=tick),
                             range=[-0.5, len(sub) - 0.5])
        fig.update_yaxes(title_standoff=0, range=yrange, tickfont=dict(size=tick),
                         tickvals=ytickvals if ytickvals is not None else None)

    def _pair(qi, qt, dx=0.12):
        """two slightly x-offset series (invariant orange, tuned brown). On the log N axis the
        offset is multiplicative (x * (1 -/+ f)) so it reads symmetric."""
        xl, xr = (xs * (1 - 0.03), xs * (1 + 0.03)) if xlog else (xs - dx, xs + dx)
        return pd.concat([
            pd.DataFrame({'x': xl, 'y': sub[f'{qi} mean'].values, 'erry': sub[f'{qi} sem'].fillna(0).values,
                          'errx': 0, 'color': COLOR_INV, 'errcolor': 'rgba(200,200,200,1.0)', 'symbol': 'circle', 'size': 8}),
            pd.DataFrame({'x': xr, 'y': sub[f'{qt} mean'].values, 'erry': sub[f'{qt} sem'].fillna(0).values,
                          'errx': 0, 'color': COLOR_TUN, 'errcolor': 'rgba(200,200,200,1.0)', 'symbol': 'circle', 'size': 8}),
        ], ignore_index=True)

    def _overlay(full, eqv, rgb='140,140,140'):
        """decoder in the full space (40 %-opaque diamonds) over the estimated subspace (opaque
        circles, drawn on top), overlaid at the same x; both markers the same colour `rgb`."""
        return pd.concat([
            pd.DataFrame({'x': xs, 'y': sub[f'{full} mean'].values, 'erry': sub[f'{full} sem'].fillna(0).values,
                          'errx': 0, 'color': f'rgba({rgb},0.4)', 'errcolor': 'rgba(200,200,200,1.0)', 'symbol': 'diamond', 'size': 10}),
            pd.DataFrame({'x': xs, 'y': sub[f'{eqv} mean'].values, 'erry': sub[f'{eqv} sem'].fillna(0).values,
                          'errx': 0, 'color': f'rgba({rgb},1)', 'errcolor': 'rgba(200,200,200,1.0)', 'symbol': 'circle', 'size': 8}),
        ], ignore_index=True)

    #estimated invariant / tuned dimension counts in one panel, with dashed lines at the reference
    #truth (d_inv = 6, d_tun = 2); y-axis fixed to 0-10 (except the `dims` axis, whose cells sweep
    #the true counts, so no fixed reference line applies there)
    if 'd_inv mean' in sub.columns and 'd_tun mean' in sub.columns:
        fig = plot_points_w_errorbars(_pair('d_inv', 'd_tun'), sz=8, height=PANEL_HEIGHT, width=width, alpha=.6)
        _restyle(fig, [0, 10], hline=None if axis == 'dims' else [DEFAULT['d_inv'], DEFAULT['d_tun']],
                 hline_kw=dict(line_width=1.2, line_dash='dash', line_color='rgba(80,80,80,1)'))
        _write(fig, results_folder / f'simulation_{axis}_dims.pdf')

    #image-identity SVM decoding error at the BEST C over SVM_C_RANGE (svm_err_best /
    #svm_err_eqv_best): full space (faint diamonds) and estimated invariant subspace (circles)
    if 'svm_err_best mean' in sub.columns and 'svm_err_eqv_best mean' in sub.columns:
        fig = plot_points_w_errorbars(_overlay('svm_err_best', 'svm_err_eqv_best'), sz=8, height=PANEL_HEIGHT, width=width, alpha=.6)
        _restyle(fig, [0, 1], hline=1 - 1 / 91)   # dotted chance level (91-way image identity)
        _write(fig, results_folder / f'simulation_{axis}_svm_err.pdf')

    #relative-orientation decoding error (deg): full space (faint diamonds) and estimated tuned
    #subspace (circles)
    if 'ori_err mean' in sub.columns and 'ori_err_eqv mean' in sub.columns:
        fig = plot_points_w_errorbars(_overlay('ori_err', 'ori_err_eqv'), sz=8, height=PANEL_HEIGHT, width=width, alpha=.6)
        _restyle(fig, [0, 95], hline=90, ytickvals=[0, 45, 90])   # dotted chance level; ymax 95 so chance-level markers are not clipped
        _write(fig, results_folder / f'simulation_{axis}_ori_err.pdf')

    #pooled centered CC_norm of the equivariance model, dotted 0-line; floor padded below 0 so a
    #marker at exactly 0 (e.g. dims_8_0) clears the axis
    q = 'cc_norm_equivariance_c'
    if f'{q} mean' in sub.columns:
        yrange = [0 if axis == 'xi' else -0.05, 1]
        plot_df = pd.DataFrame({'x': xs, 'y': sub[f'{q} mean'].values, 'erry': sub[f'{q} sem'].fillna(0).values, 'errx': 0,
                                'color': COLOR_GREY, 'errcolor': 'rgba(200,200,200,1.0)', 'symbol': 'circle', 'size': 8})
        fig = plot_points_w_errorbars(plot_df, sz=8, height=PANEL_HEIGHT, width=width, alpha=.6)
        _restyle(fig, yrange, hline=0)
        _write(fig, results_folder / f'simulation_{axis}_{q}.pdf')


def plot_calibration(summary, results_folder, datafit_csv=None):
    """
    Calibration of the equivariance model's held-out scores against the ground truth of the
    noiseless generator (SI Fig 7), one point per condition (mean over seeds), dashed identity
    line: (a) the pooled centered CC_norm against the TRUE correlation between the prediction
    and the noiseless signal (whose square is the variance explained after optimal per-neuron
    scaling); (b) CC_norm**2 against the TRUE as-is R**2, 1 - MSE / var(signal), which is a
    different quantity; (c) the centered SPE (signal power explained, Schoppe et al. 2016;
    the 'feve' of helper.pooled_scores) against the true R**2. Each for the cells of the
    low-dimensional benchmark (light grey; the snr_hetero and vtun axes excluded) and, when
    datafit_csv (default results_extended/data_fit/data_fit_sim.csv) exists, for the ten
    data-matched populations of ED Fig 10b (dark, named). Writes
    simulation_calibration_{ccnorm_vs_corr,ccnorm2_vs_r2,spe_vs_r2}.pdf.
    """
    import plotly.graph_objects as go
    sq = lambda c: np.sign(c) * c**2
    sub = summary.dropna(subset=['true_r2_equivariance_c mean', 'cc_norm_equivariance_c mean'])
    drop = sub['cell'].str.startswith(('control', 'snr_hetero', 'datamatch_vtun'))
    sub = sub[~drop]
    bench = dict(cells=sub['cell'].values,
                 r2=sub['true_r2_equivariance_c mean'].values,
                 corr=sub['true_cc_equivariance_c mean'].values,
                 ccnorm=sub['cc_norm_equivariance_c mean'].values,
                 ccnorm2=sq(sub['cc_norm_equivariance_c mean'].values),
                 spe=sub['feve_equivariance_c mean'].values)

    matched = None
    if datafit_csv is None:
        datafit_csv = Path(results_folder).resolve().parent / DATAFIT_DIR / 'data_fit_sim.csv'
    if Path(datafit_csv).exists():
        cols = ['cc_norm_equivariance_c', 'feve_equivariance_c', 'true_cc_equivariance_c', 'true_r2_equivariance_c']
        df = pd.read_csv(datafit_csv)
        if all(c in df.columns for c in cols):
            order = [a for a in TARGET_ORDER if a in set(df['area'])]
            m = df.groupby('area')[cols].mean().loc[order]
            matched = dict(cells=np.array(order), mean=m)

    def _scatter(x, y, labels, xlab, ylab, xrange, yrange, fname, overlay=None, legend_pos='top left'):
        #benchmark cells: light grey; overlay = (x, y, labels) of the data-matched populations,
        #darker and opaque, with a legend in the empty corner of each panel
        lo, hi = min(xrange[0], yrange[0]), max(xrange[1], yrange[1])
        lg = dict(x=0.02, y=0.98, xanchor='left', yanchor='top') if legend_pos == 'top left' \
            else dict(x=0.98, y=0.02, xanchor='right', yanchor='bottom')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines',
                                 line=dict(color='rgba(200,200,200,1)', width=1, dash='dash'), showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', text=labels, name='benchmark',
                                 marker=dict(size=6, color='rgba(140,140,140,0.4)')))
        if overlay is not None:
            fig.add_trace(go.Scatter(x=overlay[0], y=overlay[1], mode='markers', text=overlay[2], name='data-matched',
                                     marker=dict(size=7, color='rgba(30,30,30,0.9)')))
        fig.update_layout(height=220, width=260, template='simple_white', margin=dict(l=52, r=10, t=10, b=42),
                          font=dict(size=13.5, family='Arial'), showlegend=overlay is not None,
                          legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)', itemsizing='trace', **lg))
        fig.update_xaxes(title=xlab, title_standoff=0, range=xrange)
        fig.update_yaxes(title=ylab, title_standoff=0, range=yrange)
        _write(fig, results_folder / fname)

    ov = {}
    if matched is not None:
        m, cells = matched['mean'], matched['cells']
        ov = dict(corr=(m['true_cc_equivariance_c'].values, m['cc_norm_equivariance_c'].values, cells),
                  r2=(m['true_r2_equivariance_c'].values, m['feve_equivariance_c'].values, cells),
                  r2_cc2=(m['true_r2_equivariance_c'].values, sq(m['cc_norm_equivariance_c'].values), cells))
    spe_floor = min(-0.1, np.nanmin(bench['spe']) - 0.05)
    r2_floor = -0.05 if matched is None else min(-0.05, 1.15 * np.nanmin(ov['r2'][0]))
    #SI Fig 7a: CC_norm vs the true correlation
    _scatter(bench['corr'], bench['ccnorm'], bench['cells'], 'true corr (pred., signal)', 'centered CC_norm',
             [-0.05, 1], [-0.05, 1], 'simulation_calibration_ccnorm_vs_corr.pdf', overlay=ov.get('corr'))
    #SI Fig 7b: CC_norm**2 vs the true as-is R**2 (the data-matched populations sit above the
    #identity: CC_norm**2 > 0 while the as-is R**2 < 0)
    _scatter(bench['r2'], bench['ccnorm2'], bench['cells'], 'true R**2', 'centered CC_norm**2',
             [r2_floor, 1], [-0.05, 1], 'simulation_calibration_ccnorm2_vs_r2.pdf', overlay=ov.get('r2_cc2'))
    #SI Fig 7c: SPE vs the true as-is R**2 (x floor widened so the data-matched points, which are
    #all negative, are not clipped)
    _scatter(bench['r2'], bench['spe'], bench['cells'], 'true R**2', 'centered SPE',
             [r2_floor, 1], [spe_floor, 1], 'simulation_calibration_spe_vs_r2.pdf', overlay=ov.get('r2'),
             legend_pos='bottom right')


# ----------------------------------------------------------------------------------------
# 2. calibration simulations based on pixels: CC_norm over the (N, d) grid (ED Fig 9c-h, SI Fig 6f)
# ----------------------------------------------------------------------------------------
#grey ramp light -> dark (the colour scale of the data figures); colours must end in '1.0)'
#(plot_points_w_errorbars rebuilds the line colour as color[:-4] + '<alpha>)')
GREY = ['rgba(211, 211, 211, 1.0)', 'rgba(169, 169, 169, 1.0)', 'rgba(128, 128, 128, 1.0)',
        'rgba(105, 105, 105, 1.0)', 'rgba(0, 0, 0, 1.0)']
N_RAMP = [250, 1000, 4000, 8000]                       # the standard N of the ramps ...
N_SHADES = [GREY[0], GREY[1], GREY[3], GREY[4]]        # ... and their shades (log-interpolated in between)
D_SPARSE = [6, 20, 50, 100, 200, 400, 800]             # the d of the surfaces drawn on the x axis
ERR_ALPHA = 0.3
#shared axes of the peak-over-d panels (ED Fig 9d, f, h; SI Fig 6f)
PEAK_AXES = dict(ymax=1.05, ytick=[0, 0.5, 1.0], xlab='neurons (1)', xtick=N_RAMP, xrange=(250 / 1.4, 8000 * 1.4))
#the N at which the populations were measured: AlexNet layers up to their reliably responsive
#count (noisy and the same units noiseless, suffix '_noiseless_rel'), mouse areas pooled over the
#three mice up to their reliable count
CONV_N_LEVELS = {'conv1': [250, 1000, 4000, 5040], 'conv2': [250, 1000, 2374], 'conv3': [250, 927],
                 'conv4': [250, 539], 'conv5': [250, 303]}
AREA_N_LEVELS = {'V1': [250, 1000, 4000, 8000], 'LM': [250, 1000, 3000], 'RL': [250, 1000, 1600], 'AL': [250, 855]}
CONV_NOISELESS = '_noiseless_rel'
#the simulated populations built from the pixels' equivariant structure (ED Fig 9d right;
#simulation.run_matched_populations('pixels', ...)): (cache layer, shade, dash) -- dashed = f 1,
#solid = f 0.5, dotted = f 0.1; black = all 1362 dimensions, mid grey = R 500, light grey = R 20
PIXEL_SIMULATIONS = [('rtruth_c0', GREY[4], 'dash'),         # f = 1,   full rank
                     ('rtruth_r500_c0', GREY[3], 'dash'),    # f = 1,   R = 500
                     ('rtruth_r20_c0', GREY[1], 'dash'),     # f = 1,   R = 20
                     ('rtruth_r500_c1', GREY[3], False),     # f = 0.5, R = 500
                     ('rtruth_r20_c1', GREY[1], False),      # f = 0.5, R = 20
                     ('rtruth_r20_c2', GREY[1], 'dot')]      # f = 0.1, R = 20
#SI Fig 6f: the equivariant CNNs in the SI Fig 6 colours, conv1 light -> conv5 full colour
EQV_NET_RGB = {'eqv-c15': (227, 119, 194), 'eqv-c15r': (214, 39, 40), 'eqv-c1': (0, 0, 0)}
EQV_DEPTH_T = [0.3, 0.475, 0.65, 0.825, 1.0]
EQV_NOISELESS_LAYERS = ('conv1', 'conv5')


def _calibration_dir(project_root):
    return Path(project_root) / 'results_extended' / CALIBRATION_DIR


def ccnorm_curve(project_root, network, layer, N, d_sel=None, stat='median', suffix=''):
    """CC_norm vs d of one cached surface (cache/{network}/{layer}_ccnorm_pooled_vsdim_{N}avg
    {suffix}.csv): the stat over neurons per (subsample, d), then mean +- sem over subsamples.
    Returns a frame with x (d), y, erry, or None when the cache is missing."""
    f = Path(project_root) / 'cache' / network / f'{layer}_ccnorm_pooled_vsdim_{N}avg{suffix}.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[df['centered'] == True]  # noqa: E712
    if d_sel is not None:
        df = df[df['dim'].isin(d_sel)]
        if df.empty:
            return None
    samp = getattr(df.groupby(['sample', 'dim'])['cc_norm'], stat)().reset_index()
    g = samp.groupby('dim')['cc_norm']
    out = pd.DataFrame({'y': g.mean(), 'erry': (g.std() / np.sqrt(g.count())).fillna(0.0)})
    return out.reset_index().rename(columns={'dim': 'x'})


def ccnorm_peaks(project_root, network, layer, n_list, suffix=''):
    """Peak over d of the CC_norm curve vs N (x = N) from the cached surfaces of one population;
    the error bar is the across-subsample sem at the peak d. None when nothing is cached."""
    rows = []
    for N in n_list:
        df = ccnorm_curve(project_root, network, layer, N, suffix=suffix)
        if df is None:
            continue
        i = df['y'].idxmax()
        rows.append(dict(x=N, y=df.loc[i, 'y'], erry=df.loc[i, 'erry']))
    return pd.DataFrame(rows) if rows else None


def _rgba(s):
    return [float(v) for v in re.findall(r'[\d.]+', s)]


def shade_for_n(N, ramp=N_SHADES, anchors=N_RAMP):
    """Shade of a ramp fixed to N: log-interpolation between the anchors, so a given N maps to the
    same shade in every panel (the standard N reproduce the ramp exactly)."""
    xs = np.log(anchors); x = np.log(N)
    if x <= xs[0]:
        return ramp[0]
    if x >= xs[-1]:
        return ramp[-1]
    i = int(np.searchsorted(xs, x)) - 1
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    a, b = _rgba(ramp[i]), _rgba(ramp[i + 1])
    r, g, bl = (a[k] + t * (b[k] - a[k]) for k in range(3))
    return f'rgba({int(round(r))}, {int(round(g))}, {int(round(bl))}, 1.0)'


def plot_lines(series, path, ymax=1.05, ytick=(0, 0.5, 1.0), xlab='dimension (1)', ylab='CCnorm (1)', logx=True,
               xtick=None, zeroline=True, height=150, width=252, xrange=None):
    """One panel of connected markers with error bars per series: series = [(df[x, y, erry],
    colour, dash[, no_marker])]; dash False / True / a plotly dash name; no_marker draws a bare
    line (the noiseless ceilings). xrange fixes the (log) x axis; else it is derived from the data."""
    fig = None
    for item in series:
        df, color, dash = item[0], item[1], item[2]
        no_marker = item[3] if len(item) > 3 else False
        p = df.copy()
        p['color'] = color; p['errx'] = 0
        p['errcolor'] = color[:-4] + f'{ERR_ALPHA})'
        p.sort_values('x', inplace=True)
        f0 = plot_points_w_errorbars(p, lines=True, y_dp=([0] if zeroline else None), sz=6, height=height, width=width, alpha=1.0)
        f0.data[0].error_y.color = p['errcolor'].iloc[0]
        f0.data[0].error_x.color = p['errcolor'].iloc[0]
        if dash:
            f0.data[0].line.dash = dash if isinstance(dash, str) else 'dash'
        if no_marker:
            f0.data[0].mode = 'lines'
            f0.data[0].error_y.visible = False
            f0.data[0].error_x.visible = False
        fig = f0 if fig is None else fig.add_trace(f0.data[0])
    if fig is None:
        print(f'{Path(path).name}: nothing cached, skipped', flush=True)
        return None
    fig.update_layout(yaxis_title=ylab, xaxis_title=xlab)
    fig.update_yaxes(title_standoff=0, range=[-0.02, ymax], tickvals=list(ytick) if ytick is not None else None)
    if logx:
        xs = np.concatenate([s[0]['x'].values for s in series])
        if xtick is None:
            xtick = sorted(set(int(v) for v in xs))
        lo, hi = (xs.min() / 1.4, xs.max() * 1.4) if xrange is None else xrange
        fig.update_xaxes(title_standoff=0, type='log', range=[np.log10(lo), np.log10(hi)], tickvals=list(xtick))
    else:
        fig.update_xaxes(title_standoff=0)
    _write(fig, path)
    return fig


def plot_ccnorm_vs_d(project_root, network, layer, n_list, path, noiseless_suffix=None, d_sel=D_SPARSE, ramp=N_SHADES):
    """CC_norm vs the fitted dimension d of one population at several N (shades fixed to N, light
    -> dark), with the injected noise (solid, markers) and, with noiseless_suffix, the noiseless
    responses at each N as a dashed marker-free ceiling in the same shade."""
    series = []
    for N in n_list:
        df = ccnorm_curve(project_root, network, layer, N, d_sel=d_sel)
        if df is not None:
            series.append((df, shade_for_n(N, ramp), False))
    if noiseless_suffix:
        for N in n_list:
            df = ccnorm_curve(project_root, network, layer, N, d_sel=d_sel, suffix=noiseless_suffix)
            if df is not None:
                series.append((df, shade_for_n(N, ramp), True, True))
    return plot_lines(series, path, xtick=D_SPARSE)


def plot_ccnorm_vs_d_populations(project_root, items, path, noiseless_suffix=None, d_sel=D_SPARSE):
    """CC_norm vs d, one line per population at one N: items = [(network, layer, N, colour)]
    (noisy solid with markers; with noiseless_suffix the same units noiseless, dashed, marker-free)."""
    series = []
    for network, layer, N, color in items:
        df = ccnorm_curve(project_root, network, layer, N, d_sel=d_sel)
        if df is not None:
            series.append((df, color, False))
        if noiseless_suffix:
            nl = ccnorm_curve(project_root, network, layer, N, d_sel=d_sel, suffix=noiseless_suffix)
            if nl is not None:
                series.append((nl, color, True, True))
    return plot_lines(series, path, xtick=D_SPARSE)


def plot_ccnorm_peak_vs_n(project_root, items, path):
    """Peak over d of CC_norm vs N: items = [(network, layer, n_list, colour, dash[, no_marker,
    suffix])], shared axes with the other peak panels (PEAK_AXES)."""
    series = []
    for item in items:
        network, layer, n_list, color, dash = item[:5]
        no_marker = item[5] if len(item) > 5 else False
        suffix = item[6] if len(item) > 6 else ''
        df = ccnorm_peaks(project_root, network, layer, n_list, suffix=suffix)
        if df is not None:
            series.append((df, color, dash, no_marker))
        else:
            print(f'  {Path(path).name}: {network}/{layer}{suffix} not cached', flush=True)
    return plot_lines(series, path, **PEAK_AXES)


#---- the panels of ED Fig 9c-h
def plot_pixels_vs_d(project_root):
    """ED Fig 9c: pixels, CC_norm vs d at N = 250 / 1000 / 4000 / 8000 with the injected noise
    (solid) and noiseless (dashed) -> ccnorm_vs_d_pixels.pdf"""
    return plot_ccnorm_vs_d(project_root, 'pixels', 'pixels', N_RAMP, _calibration_dir(project_root) / 'ccnorm_vs_d_pixels.pdf',
                            noiseless_suffix='_noiseless')


def plot_pixels_peak_vs_n(project_root):
    """ED Fig 9d left: pixels, peak over d of CC_norm vs N, noisy (solid, markers) and noiseless
    (dashed) -> ccnorm_peak_vs_N_pixels.pdf"""
    items = [('pixels', 'pixels', N_RAMP, GREY[4], False),
             ('pixels', 'pixels', N_RAMP, GREY[4], True, True, '_noiseless')]
    return plot_ccnorm_peak_vs_n(project_root, items, _calibration_dir(project_root) / 'ccnorm_peak_vs_N_pixels.pdf')


def plot_pixel_simulations_peak_vs_n(project_root):
    """ED Fig 9d right: the simulated populations built from the pixels' equivariant structure at
    varying equivariant signal fraction f and dimension R (PIXEL_SIMULATIONS), peak over d of
    CC_norm vs N -> ccnorm_peak_vs_N_pixel_simulations.pdf"""
    items = [('calib_pixels', layer, N_RAMP, color, dash) for layer, color, dash in PIXEL_SIMULATIONS]
    return plot_ccnorm_peak_vs_n(project_root, items, _calibration_dir(project_root) / 'ccnorm_peak_vs_N_pixel_simulations.pdf')


def plot_conv1_vs_d(project_root):
    """ED Fig 9e left: AlexNet conv1, CC_norm vs d at N = 250 / 1000 / 5040 (its reliably
    responsive units), noisy solid and the same units noiseless dashed -> ccnorm_vs_d_conv1.pdf"""
    return plot_ccnorm_vs_d(project_root, 'alexnet', 'conv1', [250, 1000, CONV_N_REL['conv1']],
                            _calibration_dir(project_root) / 'ccnorm_vs_d_conv1.pdf', noiseless_suffix=CONV_NOISELESS)


def plot_conv_layers_vs_d(project_root):
    """ED Fig 9e right: AlexNet conv1-5 (light -> dark), CC_norm vs d at each layer's full
    reliably responsive population, noisy solid / noiseless dashed -> ccnorm_vs_d_conv_layers.pdf"""
    items = [('alexnet', lay, CONV_N_REL[lay], c) for lay, c in zip(CONVS, GREY)]
    return plot_ccnorm_vs_d_populations(project_root, items, _calibration_dir(project_root) / 'ccnorm_vs_d_conv_layers.pdf',
                                        noiseless_suffix=CONV_NOISELESS)


def plot_conv_layers_peak_vs_n(project_root):
    """ED Fig 9f: AlexNet conv1-5 (light -> dark), peak over d of CC_norm vs N up to each layer's
    reliable count, noisy solid with markers / the same units noiseless dashed without
    -> ccnorm_peak_vs_N_conv_layers.pdf"""
    items = []
    for lay, c in zip(CONVS, GREY):
        items.append(('alexnet', lay, CONV_N_LEVELS[lay], c, False))
        items.append(('alexnet', lay, CONV_N_LEVELS[lay], c, True, True, CONV_NOISELESS))
    return plot_ccnorm_peak_vs_n(project_root, items, _calibration_dir(project_root) / 'ccnorm_peak_vs_N_conv_layers.pdf')


def plot_v1_vs_d(project_root):
    """ED Fig 9g left: mouse V1 pooled across the 3 mice, CC_norm vs d at N = 250 / 1000 / 4000 /
    8000 -> ccnorm_vs_d_V1.pdf"""
    return plot_ccnorm_vs_d(project_root, 'mouse_pooled', 'V1', AREA_N_LEVELS['V1'], _calibration_dir(project_root) / 'ccnorm_vs_d_V1.pdf')


def plot_areas_vs_d(project_root):
    """ED Fig 9g right: V1 / LM / RL / AL pooled across the 3 mice (light -> dark), CC_norm vs d at
    each area's largest N -> ccnorm_vs_d_areas.pdf"""
    items = [('mouse_pooled', a, AREA_N_LEVELS[a][-1], c) for a, c in zip(AREAS, GREY[1:])]
    return plot_ccnorm_vs_d_populations(project_root, items, _calibration_dir(project_root) / 'ccnorm_vs_d_areas.pdf')


def plot_areas_peak_vs_n(project_root):
    """ED Fig 9h: V1 / LM / RL / AL pooled across the 3 mice (light -> dark), peak over d of
    CC_norm vs N -> ccnorm_peak_vs_N_areas.pdf"""
    items = [('mouse_pooled', a, AREA_N_LEVELS[a], c, False) for a, c in zip(AREAS, GREY[1:])]
    return plot_ccnorm_peak_vs_n(project_root, items, _calibration_dir(project_root) / 'ccnorm_peak_vs_N_areas.pdf')


def plot_ed9_panels(project_root):
    """All panels of ED Fig 9c-h."""
    plot_pixels_vs_d(project_root); plot_pixels_peak_vs_n(project_root); plot_pixel_simulations_peak_vs_n(project_root)
    plot_conv1_vs_d(project_root); plot_conv_layers_vs_d(project_root); plot_conv_layers_peak_vs_n(project_root)
    plot_v1_vs_d(project_root); plot_areas_vs_d(project_root); plot_areas_peak_vs_n(project_root)


def _shade(rgb, t):
    r, g, b = (round(255 - (255 - c) * t) for c in rgb)
    return f'rgba({r}, {g}, {b}, 1.0)'


def plot_eqv_net_peak_vs_n(project_root, net, results_folder=None, layers=CONVS, noiseless_layers=EQV_NOISELESS_LAYERS):
    """SI Fig 6f: one equivariant CNN (eqv-c1 / eqv-c15 / eqv-c15r), peak over d of CC_norm vs N
    for conv1-5 in the network's SI Fig 6 colour shaded light (conv1) -> dark (conv5): noisy
    solid with markers for every layer, noiseless dashed marker-free for noiseless_layers. The N
    are discovered from the cached surfaces (cache/<net>/<layer>_ccnorm_pooled_vsdim_{N}avg
    [_noiseless].csv). Shared axes with ED Fig 9f -> <results_folder>/ccnorm_peak_vs_N_<net>.pdf
    (default results_extended_anns)."""
    results_folder = Path(results_folder) if results_folder else Path(project_root) / 'results_extended_anns'
    items = []
    for layer, t in zip(layers, EQV_DEPTH_T):
        colour = _shade(EQV_NET_RGB[net], t)
        n_noisy = cached_n_levels(project_root, net, layer)
        if n_noisy:
            items.append((net, layer, n_noisy, colour, False))
        if layer in noiseless_layers:
            n_nl = cached_n_levels(project_root, net, layer, '_noiseless')
            if n_nl:
                items.append((net, layer, n_nl, colour, True, True, '_noiseless'))
        print(f'  {net} {layer}: noisy N {n_noisy}' + (f', noiseless N {n_nl}' if layer in noiseless_layers else ''), flush=True)
    return plot_ccnorm_peak_vs_n(project_root, items, results_folder / f'ccnorm_peak_vs_N_{net}.pdf')


# ----------------------------------------------------------------------------------------
# 3. fit of simulations to data (ED Fig 10)
# ----------------------------------------------------------------------------------------
#x positions: pixels, V1..AL, conv1..conv5 on one axis, the three mice jittered left of each
#visual area's pooled position (as plot_classification_errors)
AREA_TO_X = {'pixels': 0, 'V1': 1.2, 'LM': 2.2, 'RL': 3.2, 'AL': 4.2,
             **{f'{m}_{a}': 0.8 + 0.1 * i + j for i, m in enumerate(MICE) for j, a in enumerate(AREAS)},
             'conv1': 5, 'conv2': 6, 'conv3': 7, 'conv4': 8, 'conv5': 9}
COLOR_POOLED, ERRCOLOR_POOLED = 'rgba(140, 140, 140, 1)', 'rgba(169, 169, 169, 1.0)'
COLOR_MOUSE, ERRCOLOR_MOUSE = 'rgba(180, 180, 180, 1.0)', 'rgba(209, 209, 209, 1.0)'
#20 %-opacity versions of the point colours for the data overlay of ED Fig 10b
COLOR_INV_T, COLOR_TUN_T = 'rgba(247, 148, 29, 0.2)', 'rgba(139, 94, 60, 0.2)'
COLOR_PTS_T = 'rgba(140, 140, 140, 0.2)'
ERRCOLOR_DIM = 'rgba(200, 200, 200, 1.0)'


def _datafit_dir(project_root):
    return Path(project_root) / 'results_extended' / DATAFIT_DIR


def _stat_row(vals, area, color, errcolor, size):
    """Mean +- sem of vals as one plot row; None if no data; sem = 0 for a single value."""
    v = np.asarray(pd.Series(vals).dropna().values, dtype=float)
    if len(v) == 0:
        return None
    sem = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return dict(area=area, y=float(v.mean()), erry=sem, color=color, errcolor=errcolor, symbol='circle', size=size)


def _series_measured(df, p):
    """AREA_TO_X key -> per-subsample values of the measured parameter p (pooled areas and per
    mouse 'TX1xx_area')."""
    df = df.copy(); df['mouse'] = df['mouse'].fillna('')
    g = {'pixels': df.loc[df['group'] == 'pixels', p]}
    for conv in CONVS:
        g[conv] = df.loc[df['group'] == conv, p]
    for area in AREAS:
        g[area] = df[(df['family'] == 'mouse') & (df['group'] == area)][p]
        for mshort in MICE:
            g[f'{mshort}_{area}'] = df[(df['family'] == 'mouse') & (df['mouse'] == mshort) & (df['group'] == area)][p]
    return g


def calibrated_sigma(project_root):
    """area -> [sigma = sqrt(inv_snr_pop) over the calibration seeds] from the cached data-fit
    simulations (cache/<DATAFIT_CACHE_DIR>/<area>_<seed>.csv)."""
    g = {}
    for f in sorted((Path(project_root) / 'cache' / DATAFIT_CACHE_DIR).glob('*_*.csv')):
        area = f.stem.rsplit('_', 1)[0]
        try:
            v = float(pd.read_csv(f).iloc[0]['inv_snr_pop'])
        except Exception:
            continue
        if area in AREA_TO_X and np.isfinite(v) and v >= 0:
            g.setdefault(area, []).append(float(np.sqrt(v)))
    return g


def _params_panel(getter, ylabel, path):
    """One row of ED Fig 10a: pooled populations as large dots, each mouse as a small dot; getter
    maps an AREA_TO_X key to the values whose mean +- sem is plotted (missing keys skipped)."""
    rows = []
    for area in TARGET_ORDER:
        if area in getter:
            rows.append(_stat_row(getter[area], area, COLOR_POOLED, ERRCOLOR_POOLED, 8))
    for mshort in MICE:
        for area in AREAS:
            k = f'{mshort}_{area}'
            if k in getter:
                rows.append(_stat_row(getter[k], k, COLOR_MOUSE, ERRCOLOR_MOUSE, 5))
    plot_df = pd.DataFrame([r for r in rows if r is not None])
    if plot_df.empty:
        print(f'  {Path(path).name}: no data, skipped', flush=True)
        return None
    plot_df['x'] = plot_df['area'].map(AREA_TO_X)
    plot_df['errx'] = 0.0
    plot_df.sort_values('x', inplace=True)
    lo = float((plot_df['y'] - plot_df['erry']).min()); hi = float((plot_df['y'] + plot_df['erry']).max())
    pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.08
    fig = plot_points_w_errorbars(plot_df, sz=8, height=120, width=680, alpha=.6, symb='circle')
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_layout(yaxis_title=ylabel, margin=dict(l=45, r=20, b=30, t=20))
    tick_df = plot_df[~plot_df['area'].str.startswith('TX')]
    fig.update_xaxes(title_standoff=0, tickvals=tick_df['x'].values, ticktext=tick_df['area'].values)
    fig.update_yaxes(title_standoff=0, range=[lo - pad, hi + pad])
    _write(fig, path)
    return fig


def plot_datafit_params(project_root):
    """ED Fig 10a: the generator parameters of every population, top to bottom the measured radius
    spread delta / r and gain heterogeneity (per neuron subsample; the pixels and AlexNet layers
    have uniform SNR by construction and are drawn at 0), the calibrated noise sigma (per
    calibration seed of the data-fit simulations) and the equivariant signal fraction f = 1 - iid
    from the (N, d) sweep (params/datafit_swept_iid.csv) -> datafit_{delta_over_r, snr_hetero,
    sigma_calib, f_calib}.pdf. Reads results_extended/data_fit/data_fit_params.csv."""
    out = _datafit_dir(project_root)
    df = pd.read_csv(out / 'data_fit_params.csv')
    _params_panel(_series_measured(df, 'delta_over_r'), 'δ/r', out / 'datafit_delta_over_r.pdf')
    het = _series_measured(df, 'snr_hetero')
    for k in ['pixels'] + CONVS:
        het[k] = pd.Series(np.zeros(len(het[k])))
    _params_panel(het, 'SNR spread', out / 'datafit_snr_hetero.pdf')
    sigma = calibrated_sigma(project_root)
    missing = [a for a in TARGET_ORDER if a not in sigma]
    if missing:
        print(f'  sigma_calib: no data-fit simulation cached for {missing}, omitted', flush=True)
    _params_panel(sigma, 'calibrated σ', out / 'datafit_sigma_calib.pdf')
    sw = pd.read_csv(Path(project_root) / SWEPT_IID_CSV)
    _params_panel({r['area']: [1.0 - float(r['iid'])] for _, r in sw.iterrows() if r['area'] in AREA_TO_X},
                  'calibrated f', out / 'datafit_f_calib.pdf')


def data_values(project_root):
    """The measured values of every population for the overlay of ED Fig 10b (visual areas
    averaged over the three mice): the recovered d_inv / d_tun (var_dim caches), the pooled
    centered CC_norm of the equivariance model ({layer}_ccnorm_pooled_250avg_20dim.csv: median
    over neurons per subsample, mean over subsamples, as ED Fig 9b), the equivariant-subspace
    image-identity error at C = 1 ({layer}_class2x_decoder_250avg_1C_20dim.csv) and the tuned-
    subspace relative-orientation error ({layer}_ori2x_decoder_0ref_250avg_20dim.csv). Missing
    caches -> NaN."""
    cache = Path(project_root) / 'cache'
    out = {}
    for area in TARGET_ORDER:
        pairs = [(m, area) for m in MICE.values()] if area in AREAS else [TARGETS[area]]
        vals = dict(d_inv=[], d_tun=[], cc=[], svm=[], ori=[])
        for mdl, layer in pairs:
            c = cache / mdl
            vf = c / f'{layer}_var_dim_250avg_20dim.csv'
            if vf.exists():
                b = band_summary(load_var_dim(vf))
                vals['d_inv'].append(b['d_inv'].mean()); vals['d_tun'].append(b['d_tun'].mean())
            pf = c / f'{layer}_ccnorm_pooled_250avg_20dim.csv'
            if pf.exists():
                p = pd.read_csv(pf)
                p = p[(p['model'] == 'equivariance') & (p['centered'] == True) & (p['tag'] == 'data')]  # noqa: E712
                vals['cc'].append(p.groupby('sample')['cc_norm'].median().mean())
            sf = c / f'{layer}_class2x_decoder_250avg_1C_20dim.csv'
            if sf.exists():
                vals['svm'].append(1.0 - pd.read_csv(sf)['acc test'].mean())
            of = c / f'{layer}_ori2x_decoder_0ref_250avg_20dim.csv'
            if of.exists():
                vals['ori'].append(pd.read_csv(of)['res'].mean())
        out[area] = {k: (float(np.mean(v)) if v else np.nan) for k, v in vals.items()}
    return out


def plot_datafit_simulations(project_root):
    """ED Fig 10b: the structure-matched populations through the standard pipeline (opaque, mean
    +- sem over seeds) with the data (20 % opacity), top to bottom: recovered invariant (orange)
    and tuned (brown) dimensions, image-identity error in the invariant subspace (C = 1), relative-
    orientation error in the tuned subspace, pooled centered CC_norm of the equivariance model
    -> datafit_sim_{dim, svm1C, ori, cc}.pdf. Reads results_extended/data_fit/data_fit_sim.csv
    (simulation.datafit_table)."""
    out = _datafit_dir(project_root)
    df = pd.read_csv(out / 'data_fit_sim.csv')
    x = np.array([AREA_TO_X[a] for a in TARGET_ORDER])
    rv = data_values(project_root)

    def agg(col):
        if col not in df.columns:
            return pd.Series(np.nan, index=TARGET_ORDER), pd.Series(0.0, index=TARGET_ORDER)
        g = df.groupby('area')[col]
        return g.mean().reindex(TARGET_ORDER), (g.std(ddof=1) / np.sqrt(g.count())).reindex(TARGET_ORDER).fillna(0.0)

    def finish(fig, ylabel, yrange, stem, yticks=None):
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.update_layout(yaxis_title=ylabel, margin=dict(l=45, r=20, b=30, t=20))
        fig.update_xaxes(title_standoff=0, tickvals=x, ticktext=TARGET_ORDER)
        fig.update_yaxes(title_standoff=0, range=yrange, **(dict(tickvals=yticks) if yticks else {}))
        _write(fig, out / f'{stem}.pdf')

    def scalar_panel(col, key, ylabel, yrange, stem, y_dp, yticks):
        m, e = agg(col)
        plot_df = pd.DataFrame({'x': np.concatenate([x, x]),
                                'y': np.concatenate([m.values, [rv[a][key] for a in TARGET_ORDER]]), 'errx': 0.0,
                                'erry': np.concatenate([e.values, np.zeros(len(x))]),
                                'color': [COLOR_GREY] * len(x) + [COLOR_PTS_T] * len(x),
                                'errcolor': ERRCOLOR_POOLED, 'symbol': 'circle', 'size': 8}).dropna(subset=['y'])
        fig = plot_points_w_errorbars(plot_df, y_dp=y_dp, sz=8, height=170, width=680, alpha=.6)
        if yrange is None:
            lo = float(np.nanmin(plot_df['y'] - plot_df['erry']))
            yrange = [max(0, lo - 0.05), 1.03]
        finish(fig, ylabel, yrange, stem, yticks)

    #recovered dimensions: d_inv (orange, x - 0.1) and d_tun (brown, x + 0.1); data overlay transparent
    di, di_e = agg('d_inv'); dt, dt_e = agg('d_tun')
    plot_df = pd.DataFrame({'x': np.concatenate([x - 0.1, x + 0.1, x - 0.1, x + 0.1]),
                            'y': np.concatenate([di.values, dt.values, [rv[a]['d_inv'] for a in TARGET_ORDER],
                                                 [rv[a]['d_tun'] for a in TARGET_ORDER]]),
                            'errx': 0.0, 'erry': np.concatenate([di_e.values, dt_e.values, np.zeros(2 * len(x))]),
                            'color': [COLOR_INV] * len(x) + [COLOR_TUN] * len(x) + [COLOR_INV_T] * len(x) + [COLOR_TUN_T] * len(x),
                            'errcolor': ERRCOLOR_DIM, 'symbol': 'circle', 'size': 8}).dropna(subset=['y'])
    ymax = float(np.nanmax(plot_df['y'] + plot_df['erry'])) * 1.15
    fig = plot_points_w_errorbars(plot_df, sz=8, height=170, width=680, alpha=.6)
    finish(fig, 'dimension (1)', [0, max(ymax, 8)], 'datafit_sim_dim', yticks=[0, 5, 10, 15, 20])
    #pooled centered CC_norm of the equivariance model
    scalar_panel('cc_norm_equivariance_c', 'cc', 'CCnorm (1)', [-0.05, 0.3], 'datafit_sim_cc', [0], [0, 0.1, 0.2, 0.3])
    #image-identity error in the invariant subspace at C = 1 (chance 1 - 1/92 dotted)
    scalar_panel('svm_err_eqv_C1', 'svm', 'decoding error (1)', None, 'datafit_sim_svm1C', [1 - 1 / 92], [0, 0.5, 1])
    #relative-orientation error in the tuned subspace (deg; 90 = chance)
    scalar_panel('ori_err_eqv', 'ori', 'orientation error (deg)', [40, 95], 'datafit_sim_ori', [90], [45, 90])


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='figures of the low-dimensional simulations (ED Fig 8, SI Fig 7)')
    ap.add_argument('--results', default='results_extended/simulation')
    ap.add_argument('--no_plots', action='store_true')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    df = load_runs(root)
    if df.empty:
        print('no simulation runs found'); sys.exit(0)
    summary = summarize(df)
    out = root / args.results
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / 'simulation_summary.csv', index=False)
    pd.set_option('display.width', 250)
    show = ['cell', 'n_seeds', 'true_d_inv mean', 'd_inv mean', 'd_inv sd', 'true_d_tun mean', 'd_tun mean', 'd_tun sd',
            'cc_norm_equivariance_c mean', 'true_cc_equivariance_c mean', 'svm_err_best mean', 'svm_err_eqv_best mean',
            'ori_err mean', 'ori_err_eqv mean']
    print(summary[[c for c in show if c in summary.columns]].round(2).to_string(index=False))
    if not args.no_plots:
        for axis in AXES:
            plot_axis(summary, axis, out)
        plot_calibration(summary, out)
        print(f'figures written to {out}')
