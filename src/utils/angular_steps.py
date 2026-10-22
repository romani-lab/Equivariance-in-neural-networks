"""
Estimation of angular step size in tuned subspace (SI Fig 10; Supplementary methods).

The mean angular positions of the 15 rotations are read directly from the fitted tuned
planes. Per random subsample of n_neurons reliably responsive neurons (the pipeline's draws)
the equivariance model is fitted at d on the trial-averaged responses to all 92 images (per-
neuron std normalization, calculate_proj_rot_shift, find_eigendecomp). Within each tuned plane
(|eigenphase| in [0.5, 1.5) rotation steps) the projection of image i at rotation k, relative to
the fitted centre, is the complex number z_{i,k}; the angular position of rotation k is the
amplitude-weighted circular mean over images of its angle relative to rotation 0,
    S_k = sum_i z_{i,k} conj(z_{i,0}),   Phi_k = arg S_k,   R_k = |S_k| / sum_i |z_{i,k}| |z_{i,0}|
(R_k in [0, 1] = consistency of the angle across images); the plane is oriented so that the
mean step is positive and Phi_k is unwrapped along k. Planes are pooled per subsample by their
projected variance, subsamples (and mice per area) by R_k (circular mean of exp(i Phi_k)). The
angular steps phi_k = Phi_k - Phi_{k-1} (the 15th closes the loop back to 0) have the angular
coefficient of variation xi = sd / mean over the 15 steps (0 for a regular population).

Outputs:
  cache/angular_steps/{network}_{layer}.csv      one row per (subsample, plane, k)
  results_extended/angular_steps_summary.csv     per population (areas pooled over mice, per
                                                 mouse, pixels, conv1-5): Phi_k, the deviation
                                                 from the regular grid, the step CV xi
  results_extended/angular_steps_{TX108,TX109,TX111,alexnet}.pdf   SI Fig 10a-d
"""
import numpy as np
import pandas as pd
from pathlib import Path

from utils.equivariance_functions import calculate_proj_rot_shift
from utils.helper import find_eigendecomp
from utils.equivariance_validation import subsamples
from utils.plotting_functions import plot_points_w_errorbars

CACHE_DIR = 'angular_steps'
MICE = ['TX108_05_31_2023', 'TX109_05_30_2023', 'TX111_05_18_2023']          # mouse 1, 2, 3
AREAS = ['V1', 'LM', 'RL', 'AL']
CONVS = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
GREYS = ['rgba(211, 211, 211, 1.0)', 'rgba(169, 169, 169, 1.0)', 'rgba(128, 128, 128, 1.0)',
         'rgba(105, 105, 105, 1.0)', 'rgba(0, 0, 0, 1.0)']                      # ED Fig 2 ramp (pixels/conv1 light -> AL/conv5 black)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def readout(x, d, tuned=(0.5, 1.5)):
    """Angular positions Phi_k of the rotations in the fitted tuned planes of x (n, n_stim, n_rot).

    Returns a list of dicts, one per tuned plane: Phi (n_rot, rad, unwrapped, Phi[0] = 0), R
    (n_rot, consistency across images), var (projected variance of the plane), phase (eigenphase
    in units of 2 pi / n_rot), step (mean step angle, rad)."""
    n, n_stim, n_rot = x.shape
    unit = 2 * np.pi / n_rot
    proj, proj_shift, rot, _ = calculate_proj_rot_shift(x, d)
    evals, evecs = find_eigendecomp(rot)
    xc = (x - proj_shift[:, None, None]).reshape(n, -1)
    out = []
    for i in range(d // 2):
        pair = [2 * i, 2 * i + 1]
        ph = np.abs(evals[pair]).mean() / unit
        if not (tuned[0] <= ph < tuned[1]):
            continue
        P = proj @ evecs[:, pair]                                   # (n, 2) plane in neuron space
        P, _ = np.linalg.qr(P)                                      # orthonormal (defensive)
        z = (P.T @ xc).reshape(2, n_stim, n_rot)
        z = z[0] + 1j * z[1]                                        # (n_stim, n_rot)
        step = np.angle(np.sum(z[:, 1:] * np.conj(z[:, :-1])))      # mean step (rad), orients the plane
        if step < 0:
            z = np.conj(z); step = -step
        S = np.sum(z * np.conj(z[:, :1]), axis=0)                   # (n_rot,) resultant vs rotation 0
        R = np.abs(S) / np.sum(np.abs(z) * np.abs(z[:, :1]), axis=0)
        Phi = np.unwrap(np.angle(S)); Phi -= Phi[0]
        out.append(dict(var=float(np.mean(np.abs(z) ** 2)), phase=float(ph), step=float(step), Phi=Phi, R=R))
    return out


def run_row(project_root, network, layer, n_neurons, d, n_max, verbose=True):
    """The readout of one population (cache/{network}/{layer}_avg_resp_{train,test2x}.p, all 92
    images) on up to n_max disjoint subsamples of n_neurons reliable neurons, cached as
    cache/angular_steps/{network}_{layer}.csv (one row per subsample, plane and rotation k).
    Returns the frame (None when the average caches are missing)."""
    out_dir = Path(project_root) / 'cache' / CACHE_DIR
    out_f = out_dir / f'{network}_{layer}.csv'
    if out_f.exists():
        return pd.read_csv(out_f)
    cache = Path(project_root) / 'cache' / network
    tr_f, te_f = cache / f'{layer}_avg_resp_train.p', cache / f'{layer}_avg_resp_test2x.p'
    if not (tr_f.exists() and te_f.exists()):
        print(f'  {network} {layer}: no average caches (example_equiv_projs), skipping', flush=True)
        return None
    x_all = np.concatenate([pd.read_pickle(tr_f), pd.read_pickle(te_f)], axis=1)   # (n_resp, 92, 15)
    n_resp, _, n_rot = x_all.shape
    n_neurons, k_rand, k_samp = subsamples(n_resp, n_neurons, n_max)
    rows = []
    for k in range(k_rand):
        x = x_all[k_samp[k]]
        std = x.std(axis=(1, 2), keepdims=True); keep = std.squeeze() > 0
        x = x[keep] / std[keep]
        planes = readout(x, d)
        for p_i, p in enumerate(planes):
            for kk in range(n_rot):
                rows.append(dict(network=network, layer=layer, sample=k, plane=p_i, phase=p['phase'], var=p['var'],
                                 k=kk, Phi_deg=np.degrees(p['Phi'][kk]), R=p['R'][kk]))
        if verbose and planes:
            lead = max(planes, key=lambda p: p['var'])
            dev = np.degrees(np.mean(np.abs(_wrap(lead['Phi'] - 2 * np.pi * np.arange(n_rot) / n_rot))))
            print(f'  {network} {layer} subsample {k}: {len(planes)} tuned planes; leading plane |dev| {dev:.1f} deg, '
                  f'R {lead["R"][1:].mean():.2f}', flush=True)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_f, index=False)
    return df


def run_all(project_root, model_df, params_df):
    """run_row for every row of model_df with the standard analysis parameters of params_df
    (n_neurons, n_procs = d, n_max)."""
    n_neurons = int(params_df.loc['n_neurons'].value)
    d = int(params_df.loc['n_procs'].value)
    n_max = int(params_df.loc['n_max'].value)
    for _, r in model_df.iterrows():
        run_row(project_root, r['network'], r['layer'], n_neurons, d, n_max)


def pool(df, n_rot=15):
    """Pool Phi_k over planes (variance-weighted), subsamples and mice (R-weighted circular mean).
    Returns (Phi_deg (n_rot,), R_pooled (n_rot,), n_fits)."""
    w = df['var'] * df['R']
    z = w * np.exp(1j * np.radians(df['Phi_deg']))
    S = z.groupby(df['k']).sum(); W = w.groupby(df['k']).sum()
    Phi = np.unwrap(np.angle(S.reindex(range(n_rot)).values)); Phi -= Phi[0]
    R = (np.abs(S) / W).reindex(range(n_rot)).values
    return np.degrees(Phi), R, int(df[['network', 'layer', 'sample', 'plane']].drop_duplicates().shape[0])


def metrics(Phi_deg):
    """Deviation from the regular grid and the step statistics (the step CV = xi) of pooled
    angular positions (deg)."""
    n_rot = len(Phi_deg)
    reg = 360.0 * np.arange(n_rot) / n_rot
    dev = np.abs(np.degrees(_wrap(np.radians(Phi_deg - reg))))
    steps = np.diff(np.append(Phi_deg, Phi_deg[0] + 360.0))
    return dict(dev_mean=float(dev.mean()), dev_max=float(dev.max()), step_mean=float(steps.mean()),
                step_sd=float(steps.std()), step_cv=float(steps.std() / steps.mean()),
                step_min=float(steps.min()), step_max=float(steps.max()))


def _load_readouts(project_root):
    files = sorted((Path(project_root) / 'cache' / CACHE_DIR).glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'no readouts in cache/{CACHE_DIR}; run_all first')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if 'subset' in df.columns:                      # older caches also hold per-class readouts
        df = df[df['subset'] == 'all']
    return df


def summarize(project_root):
    """Per population the pooled angular positions Phi_k, the mean / max deviation from the
    regular 24-deg grid, the step CV xi (quoted in the SI), the step range and the mean
    consistency R: the mouse areas pooled over the three mice ('V1'), per mouse ('V1_TX108'),
    the pixels and conv1-5 -> results_extended/angular_steps_summary.csv; returns the frame."""
    df = _load_readouts(project_root)
    groups = {}
    mouse = df['network'].str.startswith('TX')
    for a in AREAS:
        g = df[(df['layer'] == a) & mouse]
        if len(g):
            groups[a] = g
        for m in sorted(df.loc[mouse & (df['layer'] == a), 'network'].unique()):
            groups[f'{a}_{m[:5]}'] = df[(df['layer'] == a) & (df['network'] == m)]
    for c in ['pixels'] + CONVS:
        g = df[df['layer'] == c]
        if len(g):
            groups[c] = g
    recs = []
    for name, g in groups.items():
        Phi, R, n = pool(g)
        rec = dict(population=name, n_fits=n, R_mean=float(np.nanmean(R[1:])), **metrics(Phi))
        rec.update({f'Phi_{k}': Phi[k] for k in range(len(Phi))})
        recs.append(rec)
    out = pd.DataFrame(recs)
    res = Path(project_root) / 'results_extended'
    res.mkdir(parents=True, exist_ok=True)
    out.to_csv(res / 'angular_steps_summary.csv', index=False)
    return out


def plot_si_panels(project_root, results_folder=None, height=150, width=680):
    """SI Fig 10: the angular step of each of the 15 rotation steps (the 15th closes the loop),
    mean +- sem across neuron subsamples (planes pooled within a subsample), x labelled by the
    rotation angle reached, dotted line at the regular 24 deg; one panel per mouse (pixels +
    V1, LM, RL, AL, light to dark) and one for AlexNet conv1-5
    -> results_extended/angular_steps_{TX108, TX109, TX111, alexnet}.pdf"""
    results_folder = Path(results_folder) if results_folder else Path(project_root) / 'results_extended'
    results_folder.mkdir(parents=True, exist_ok=True)
    data = _load_readouts(project_root)
    n_rot = int(data['k'].max()) + 1

    def steps_by_sample(g):
        out = []
        for s, gs in g.groupby('sample'):
            Phi, R, n = pool(gs, n_rot)
            out.append(np.diff(np.append(Phi, Phi[0] + 360.0)))
        return np.array(out)                                            # (n_samples, n_rot)

    def panel(series, out_name):
        fig = None
        for name, g, color in series:
            if g is None or not len(g):
                continue
            st = steps_by_sample(g)
            df = pd.DataFrame({'x': np.arange(1, n_rot + 1), 'y': st.mean(0),
                               'erry': st.std(0, ddof=1) / np.sqrt(st.shape[0]) if st.shape[0] > 1 else 0.0,
                               'errx': 0.0, 'color': color, 'errcolor': 'rgba(169, 169, 169, 1.0)'})
            f = plot_points_w_errorbars(df, lines=True, y_dp=[360 / n_rot], sz=8, height=height, width=width, alpha=1.0, symb='circle')
            if fig is None:
                fig = f
            else:
                fig.add_trace(f.data[0])
        if fig is None:
            return
        fig.update_layout(yaxis_title='angular step (deg)')
        fig.update_yaxes(title_standoff=0, range=[0, 60], tickvals=[0, 24, 48])
        fig.update_xaxes(title_standoff=0, range=[0.5, n_rot + 0.5], tickvals=np.arange(1, n_rot + 1),
                         ticktext=[str(round(360 / n_rot * k)) for k in range(1, n_rot + 1)], title_text='relative orientation (deg)')
        fig.write_image(results_folder / out_name, scale=1, engine='kaleido')

    pix = data[data['layer'] == 'pixels']
    for mouse in MICE:
        series = [('pixels', pix, GREYS[0])] + [(a, data[(data['network'] == mouse) & (data['layer'] == a)], GREYS[i + 1])
                                                for i, a in enumerate(AREAS)]
        panel(series, f'angular_steps_{mouse[:5]}.pdf')
    panel([(c, data[data['layer'] == c], GREYS[i]) for i, c in enumerate(CONVS)], 'angular_steps_alexnet.pdf')
