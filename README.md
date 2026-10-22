# Equivariant neuronal populations enable simultaneous tuning and invariance

The accompanying preprint is available on bioRxiv: [Equivariant neuronal populations enable simultaneous tuning and invariance](https://www.biorxiv.org/content/10.1101/2024.08.02.606398v2).

## Overview

This project investigates how neuronal populations may carry equivariant representations to simultaneously support tuning and invariance. We analyze experimental neuronal responses from mouse visual cortex (calcium imaging data) and the convolutional neural network AlexNet, and validate the computational method on simulated populations with known structure.

## Repository Structure

```
├── src/
│   ├── utils/                         # Analysis utilities
│   ├── more_anns/                     # AlexNet training, feature extraction, equivariant CNNs
│   ├── plot_figures.ipynb             # Main figures
│   ├── plot_extended_figures.ipynb    # Extended data and supplementary figures
│   ├── more_anns_analyses.ipynb       # AlexNet variants and CNNs with built-in equivariance
│   ├── gain_model_analyses.ipynb      # Multiplicative gain model analyses
│   └── simulations.ipynb              # Simulated populations: benchmark, calibration, fit to data
├── data/
│   ├── 05_18_2023/                    # Recording session data of TX111 (mouse 3)
│   ├── 05_30_2023/                    # Recording session data of TX109 (mouse 2)
│   ├── 05_31_2023/                    # Recording session data of TX108 (mouse 1)
│   ├── retinotopy/                    # Visual area boundaries
│   └── stimuli/                       # Visual stimuli
│       ├── stim_05-08-23.mat
├── params/
│   ├── model_params.csv               # Model and layer specifications
│   ├── fixed_params.csv               # Analysis parameters
│   ├── *_all.csv / *_gain.csv / *_anns.csv / model_params_eqv.csv / fixed_params_eqv_snr0.csv
│   │                                  # Per-notebook parameter variants
│   └── datafit_swept_iid.csv          # Equivariant signal fractions of the simulated populations
├── cache/                             # Intermediate analysis results (generated)
├── cache_gain/                        # Gain model intermediate results (generated)
├── results/                           # Main figures and statistics (generated)
├── results_extended/                  # Extended figures (generated)
├── results_extended_gain/             # Gain model figures (generated)
├── results_extended_anns/             # Extended ANN figures (generated)
└── docs/
    ├── ann_training.md                # How to train the ANN models
    └── paper_figs.md                  # Mapping of output files to paper figures
```

## Quick Start

The analysis and figure generation is contained in five notebooks (run from `src/`):

* **`plot_figures.ipynb`** — Main figures (Fig 1c-e, 3b-c, 4a-b), ED Fig 1 and 3a top, the effect sizes of Fig 3c
* **`plot_extended_figures.ipynb`** — ED Fig 2, 3a bottom, 3b, 4, 5 and 9a-b; SI Fig 3, 4, 8, 9 and 10; the shuffle nulls of Fig 3b and the multiple-comparison correction of the p-values
* **`more_anns_analyses.ipynb`** — AlexNet variants trained on different objectives and CNNs with built-in rotation equivariance (ED Fig 6d-g, SI Fig 6b-f)
* **`gain_model_analyses.ipynb`** — Multiplicative gain model fits and comparisons (SI Fig 5)
* **`simulations.ipynb`** — Low-dimensional benchmark simulations (ED Fig 8, SI Fig 7), the calibration simulations based on pixels (ED Fig 9c-h) and the fit of simulations to data (ED Fig 10), with the effect sizes of ED Fig 8b

Run `plot_figures.ipynb` first: it writes the trial-averaged response caches that the other notebooks read. `simulations.ipynb` also needs the single-trial caches of `plot_extended_figures.ipynb` (its measured-parameter cell calls the same function). Every long computation is cached under `cache/` and skipped on later runs; the first run of the heavy cells (decoders, pooled CC_norm with shuffle nulls, simulations) takes hours to days on a workstation.

### Running on a cluster

The heavy cells are loops over independent units of work — one row of `params/model_params.csv`, one simulation cell and seed, or one simulated population — each cached and skipped when present. To distribute them, run one unit per array task and map the task index to the unit, for example

```python
import sys, pandas as pd
from pathlib import Path
root = Path('.'); sys.path.insert(0, str(root / 'src'))
from utils.equivariance_validation import trial_caches_all, pooled_ccnorm_all
i = int(sys.argv[1])                      # the array index of your scheduler
model_df = pd.read_csv(root / 'params' / 'model_params.csv').iloc[i:i + 1]
params_df = pd.read_csv(root / 'params' / 'fixed_params.csv', index_col=0)
trial_caches_all(model_df, params_df, root); pooled_ccnorm_all(model_df, params_df, root, n_shuffle=10)
```

and likewise `simulation.run_cell_cached(root, cell, seed)` with `(cell, seed)` taken from the index. The worker count of the decoder process pools is read from the environment variable `N_WORKERS` (default 4). The notebooks then only read the caches.

## Main Directories

### `src/utils/`

* **`stimulus_loader.py`**: Visual stimulus handling and preprocessing
* **`data_loader.py`**: Loading and preprocessing calcium imaging data
* **`model_loader.py`**: Interface for pretrained artificial neural network models
* **`neuron_selectivity.py`**: Responsive neuron detection and reliability metrics
* **`decoder_functions.py`**: Simple decoding and cross-validation
* **`equivariance_functions.py`**: Computational method for estimating equivariance
* **`equivariance_validation.py`**: Validation of the fits on single trials — pooled, within-image-centered CC_norm of the equivariance model and its control models, shuffle nulls, and the sweep over the fitted dimension
* **`simulation.py`**: Simulated populations with known structure through the identical pipeline — the low-dimensional benchmark, the calibration simulations based on pixels (equivariant structure measured at full rank, simulated populations built from it) and the fit of simulations to data
* **`plot_simulation.py`**: Figures of the simulations (ED Fig 8-10, SI Fig 6f, SI Fig 7)
* **`angular_steps.py`**: Angular positions of the rotations read from the fitted tuned planes (SI Fig 10)
* **`plotting_functions.py`**, **`plotting_validation.py`**: Visualization and figure generation
* **`multiple_comparisons.py`**: Holm-Bonferroni and Bonferroni corrections of the significance tables
* **`helper.py`**: Mathematical utilities and statistical functions
* **`gain_model.py`**: Multiplicative gain model fitting
* **`robustness.py`**: Rotation and orientation robustness evaluation of AlexNet on ImageNet

### `src/more_anns/` — ANN training package

Training infrastructure for AlexNet variants on ImageNet, using Hydra configuration.
Supports three tasks: classification, orientation regression (siamese), and multitask (both), and three AlexNet-style backbones with built-in rotation equivariance (`equivariant_models.py`, built on `escnn`).
See [`docs/ann_training.md`](docs/ann_training.md) for detailed training instructions.

* **`models.py`**: Unified AlexNet supporting all three tasks
* **`equivariant_models.py`**: AlexNet-style backbones with rotation equivariance in every layer (C15), relaxed across layers (C15>C5>C1) or none (C1)
* **`losses.py`**: Angle representations (sin/cos) and orientation loss functions
* **`metrics.py`**: Orientation MAE and top-k classification accuracy
* **`data.py`**: ImageNet dataloader with circular masking and rotation augmentation
* **`preprocessing.py`**: Circular masks, single and dual random rotation transforms
* **`train.py`**: Hydra entry point — dispatches to task-specific trainers
* **`train_base.py`**: Base trainer (optimizer, scheduler, checkpointing, TensorBoard logging)
* **`train_classification.py`**: Classification trainer
* **`train_orientation.py`**: Siamese/absolute orientation trainer
* **`train_multitask.py`**: Joint classification + orientation trainer
* **`feature_extractor.py`**: Load a trained checkpoint and extract intermediate activations
* **`evaluate.py`**: Post-hoc evaluation of orientation checkpoints (`stats` prediction distribution, `sweep` predicted-vs-true over rotation)
* **`configs/`**: Hydra YAML configs for each task

**Training (quick reference):**

```bash
python -m more_anns.train experiment=rot-aug task=classification task.use_rotation_aug=true
python -m more_anns.train experiment=ori task=orientation task.similarity_layer=classifier.3
python -m more_anns.train experiment=multi-task task=multitask task.similarity_layer=classifier.3
```

Requires access to ImageNet via HuggingFace Datasets (`ILSVRC/imagenet-1k`). You must accept the dataset terms and authenticate with `huggingface-cli login`.

Training outputs (checkpoints, TensorBoard logs, config) are written to `cache/more_anns/{experiment}/` (e.g. `cache/more_anns/rot-aug/`). These are generated artifacts; create them by training the models or place existing checkpoints there before running analyses that load trained ANN variants.

### `data/`

Please download the data from [here](https://janelia.figshare.com/articles/dataset/Calcium_imaging_data_for_Equivariant_neuronal_populations_enable_simultaneous_tuning_and_invariance_/29955530) and place it inside the `data` folder (see repository structure above). The `data` folder contains:

* Neuronal calcium imaging data (3 mice)
* Neuronal positions and area boundaries (3 mice)
* Visual stimuli

### `params/`

`model_params.csv` specifies which models to analyze:

* `network`: Model or mouse name (incl. date of recording)
* `layer`: Layer or visual area
* `monitor`: Linear dimension of input image (for pixels and ANNs) or monitor index (for mice, 1: front)

`fixed_params.csv` contains analysis parameters:

* `stimulus`: Stimulus set identifier
* `corr_thre`: Correlation threshold for responsive neurons (0.1)
* `n_procs`: Dimensionality for equivariance analysis (20)
* `n_neurons`: Number of neurons to sample (250)
* `n_max`: Maximum number of random samples (100)
* `inv_snr`: Inverse signal-to-noise ratio (3.9)
* `noise_model`: Noise model of the artificial responses (`gaussian`; `mult_gain` in the gain variant, `noiseless` in `fixed_params_eqv_snr0.csv`)

Several notebooks read their own parameter variants alongside the base files:

* `*_all.csv` (`model_params_all.csv`, `fixed_params_all.csv`) — all reliably responsive neurons, N = 4000 (`plot_extended_figures.ipynb`, SI Fig 8)
* `*_gain.csv` (`model_params_gain.csv`, `fixed_params_gain.csv`) — the multiplicative gain model (`gain_model_analyses.ipynb`)
* `*_anns.csv` (`model_params_anns.csv`, `fixed_params_anns.csv`) and `model_params_eqv.csv` — the trained AlexNet variants and the CNNs with built-in equivariance (`more_anns_analyses.ipynb`; the latter need their checkpoints and `escnn`)
* `fixed_params_eqv_snr0.csv` — noiseless responses of the pixels and AlexNet layers (`simulations.ipynb`, `more_anns_analyses.ipynb`)
* `datafit_swept_iid.csv` — the equivariant signal fraction of every simulated population of ED Fig 10 (written by `simulations.ipynb`, read by its data-fit cells)

## Output

The analysis pipeline creates the following output directories under the project root:

* **`cache/`**: Intermediate results per model — averaged responses, split-half correlations, equivariance projections, decoder results, single-trial caches and pooled CC_norm. Organized as `cache/{model_name}/{layer_name}_*.{p,csv}`; the simulations write `cache/simulation_v3/`, `cache/simulation_effect/`, `cache/simulation_datafit_sized/`, `cache/calib_{population}/` and `cache/mouse_pooled/`, the angular steps `cache/angular_steps/`.
* **`cache_gain/`**: Intermediate results for the gain model analyses.
* **`results/`**: Main paper figures (PDF) and significance tables (CSV).
* **`results_extended/`**: Extended/supplementary figures (PDF) and tables; the simulations write `results_extended/simulation/`, `results_extended/calibration/` and `results_extended/data_fit/`.
* **`results_extended_gain/`**: Gain model figures (PDF), produced by `gain_model_analyses.ipynb`.
* **`results_extended_anns/`**: Extended ANN figures (PDF).

See [`docs/paper_figs.md`](docs/paper_figs.md) for the full mapping of output files to paper figures.

### Checkpoint path convention

Trained AlexNet checkpoints are expected at `cache/more_anns/{experiment}/best_model.pth`. The rows of `params/model_params_anns.csv` whose `network` value is `alt_alexnet_{experiment}` load them through `model_loader`: `alt_alexnet_rot-aug` loads `cache/more_anns/rot-aug/best_model.pth`, `alt_alexnet_ori` loads `cache/more_anns/ori/best_model.pth`, and `alt_alexnet_multi-task` loads `cache/more_anns/multi-task/best_model.pth`. The CNNs with built-in equivariance of `params/model_params_eqv.csv` load `cache/more_anns/{eqv-c15, eqv-c15r, eqv-c1}/best_model.pth`.

The robustness cells in `src/more_anns_analyses.ipynb` also load those same checkpoint paths directly. They will fail until the corresponding `best_model.pth` files exist.

## Installation

### Requirements

Python 3.9+ with dependencies pinned in `requirements.txt` (read dynamically by `pyproject.toml`). The code has primarily been tested on Windows 11.

```bash
pip install -e .
```

### Key Dependencies

* **Core scientific computing**: NumPy, SciPy, pandas
* **Machine learning**: scikit-learn
* **Image processing**: Pillow, scikit-image
* **Visualization**: Plotly, kaleido (pin `kaleido==0.1.0.post1`; 0.2.1 hangs on Windows)
* **Deep learning**: PyTorch, torchvision; `escnn` only for the CNNs with built-in equivariance (optional, see `requirements.txt`)
* **Configuration**: Hydra

Using a fresh virtual environment (e.g. `python -m venv` or conda) is recommended to keep the pinned versions isolated.
