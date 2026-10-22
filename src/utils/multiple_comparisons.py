"""Multiple-comparison correction of the pairwise Wilcoxon tests written by the plotting
functions.

The family of comparisons is the one listed in the Online Methods: for the mouse
areas V1>LM, V1>RL, LM>RL, LM>AL, RL>AL; for AlexNet the adjacent layers conv1>conv2, conv2>conv3,
conv3>conv4, conv4>conv5 (compute_significance). Each *_significance*.csv is
one family per system; Holm-Bonferroni (step-down, controls the family-wise error rate
without independence assumptions) and Bonferroni-corrected p-values are added as columns
'p_holm' and 'p_bonf' per system, and a combined table across all significance files of a
results folder is written as significance_corrected.csv.

Usage:  correct_folder(project_root) from the notebooks, or
        python src/utils/multiple_comparisons.py [results] [results_extended ...]
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def holm(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def correct_file(f):
    try:
        df = pd.read_csv(f)
    except pd.errors.EmptyDataError:
        return None
    if df.empty or 'p' not in df.columns:
        return None
    df['system'] = np.where(df['area0'].astype(str).str.startswith('conv'), 'alexnet', 'mouse')
    df['p_holm'] = np.nan
    df['p_bonf'] = np.nan
    for sysname, sub in df.groupby('system'):
        df.loc[sub.index, 'p_holm'] = holm(sub['p'].values)
        df.loc[sub.index, 'p_bonf'] = np.minimum(1.0, sub['p'].values * len(sub))
    df['family_size'] = df.groupby('system')['p'].transform('count')
    df.to_csv(f, index=False)
    df['file'] = Path(f).name
    return df


def correct_folder(project_root, folders=('results', 'results_extended'), verbose=True):
    """Holm-Bonferroni and Bonferroni corrections of every *_significance*.csv in the given
    results folders (columns 'p_holm', 'p_bonf' added in place), and the combined table
    <folder>/significance_corrected.csv per folder. Returns {folder: table}."""
    root = Path(project_root)
    out = {}
    for folder in folders:
        frames = []
        for f in sorted(glob.glob(str(root / folder / '*significance*.csv'))):
            if f.endswith('significance_corrected.csv'):
                continue
            out = correct_file(f)
            if out is not None:
                frames.append(out)
        if frames:
            allp = pd.concat(frames, ignore_index=True)
            allp.to_csv(root / folder / 'significance_corrected.csv', index=False)
            out[folder] = allp
            if verbose:
                pd.set_option('display.width', 250)
                print(f'{folder}: {len(frames)} files')
                print(allp[['file', 'area0', 'area1', 'p', 'p_holm', 'p_bonf']].to_string(index=False))
    return out


if __name__ == '__main__':
    correct_folder(Path(__file__).resolve().parents[2], sys.argv[1:] or ('results', 'results_extended'))
