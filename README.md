# Equivariant neuronal populations enable simultaneous tuning and invariance

This repository contains code for generating the figures of a manuscript in revision.

## Preprint

The accompanying preprint is available on bioRxiv: [Equivariant neuronal populations enable simultaneous tuning and invariance](https://www.biorxiv.org/content/10.1101/2024.08.02.606398v2.article-info) (doi: [10.1101/2024.08.02.606398](https://doi.org/10.1101/2024.08.02.606398)).

## Overview

This project investigates how neuronal populations may carry equivariant representations to simultaneously support tuning and invariance. We analyze experimental neuronal responses from mouse visual cortex (calcium imaging data) and the convolutional neural network AlexNet.

## Repository Structure

```
├── src/
│   ├── utils/                         # Analysis utilities
│   ├── more\_anns/                     # AlexNet training and feature extraction
│   ├── plot\_figures.ipynb             # Main figures
│   ├── plot\_extended\_figures.ipynb    # Extended data figures
│   ├── more\_anns\_analyses.ipynb       # Additional AlexNet analyses
│   └── gain\_model\_analyses.ipynb      # Multiplicative gain model analyses
├── data/
│   ├── 05\_18\_2023/                    # Recording session data of TX111 (mouse 3)
│   ├── 05\_30\_2023/                    # Recording session data of TX109 (mouse 2)
│   ├── 05\_31\_2023/                    # Recording session data of TX108 (mouse 1)
│   ├── retinotopy/                    # Visual area boundaries
│   └── stimuli/                       # Visual stimuli
│       ├── stim\_05-08-23.mat
├── params/
│   ├── model\_params.csv               # Model and layer specifications
│   ├── fixed\_params.csv               # Analysis parameters
│   └── \*\_all.csv / \*\_gain.csv / \*\_anns.csv  # Per-notebook parameter variants
├── cache/                             # Intermediate analysis results (generated)
├── cache\_gain/                        # Gain model intermediate results (generated)
├── results/                           # Main figures and statistics (generated)
├── results\_extended/                  # Extended figures (generated)
├── results\_extended\_gain/            # Gain model figures (generated)
├── results\_extended\_anns/            # Extended ANN figures (generated)
└── docs/
    ├── ann\_training.md                # How to train the ANN models
    └── paper\_figs.md                  # Mapping of output files to paper figures
```

## Quick Start

The primary analysis and figure generation is contained in four notebooks (run from `src/`):

* **`plot\_figures.ipynb`** — Main paper figures (Fig 1c-e, 3b-c, 4a-b)
* **`plot\_extended\_figures.ipynb`** — Extended data figures (distances, decoding sweeps)
* **`more\_anns\_analyses.ipynb`** — Model comparisons across trained AlexNet variants, rotation and orientation robustness
* **`gain\_model\_analyses.ipynb`** — Multiplicative gain model fits and comparisons

## Main Directories

### `src/utils/`

* **`stimulus\_loader.py`**: Visual stimulus handling and preprocessing
* **`data\_loader.py`**: Loading and preprocessing calcium imaging data
* **`model\_loader.py`**: Interface for pretrained artificial neural network models
* **`neuron\_selectivity.py`**: Responsive neuron detection and reliability metrics
* **`decoder\_functions.py`**: Simple decoding and cross-validation
* **`equivariance\_functions.py`**: Computational method for estimating equivariance
* **`plotting\_functions.py`**: Visualization and figure generation
* **`helper.py`**: Mathematical utilities and statistical functions
* **`gain\_model.py`**: Multiplicative gain model fitting
* **`robustness.py`**: Rotation and orientation robustness evaluation of AlexNet on ImageNet

### `src/more\_anns/` — ANN training package

Training infrastructure for AlexNet variants on ImageNet, using Hydra configuration.
Supports three tasks: classification, orientation regression (siamese), and multitask (both).
See [`docs/ann\_training.md`](docs/ann_training.md) for detailed training instructions.

* **`models.py`**: Unified AlexNet supporting all three tasks
* **`losses.py`**: Angle representations (sin/cos) and orientation loss functions
* **`metrics.py`**: Orientation MAE and top-k classification accuracy
* **`data.py`**: ImageNet dataloader with circular masking and rotation augmentation
* **`preprocessing.py`**: Circular masks, single and dual random rotation transforms
* **`train.py`**: Hydra entry point — dispatches to task-specific trainers
* **`train\_base.py`**: Base trainer (optimizer, scheduler, checkpointing, TensorBoard logging)
* **`train\_classification.py`**: Classification trainer
* **`train\_orientation.py`**: Siamese/absolute orientation trainer
* **`train\_multitask.py`**: Joint classification + orientation trainer
* **`feature\_extractor.py`**: Load a trained checkpoint and extract intermediate activations
* **`evaluate.py`**: Post-hoc evaluation of orientation checkpoints (`stats` prediction distribution, `sweep` predicted-vs-true over rotation)
* **`configs/`**: Hydra YAML configs for each task

**Training (quick reference):**

```bash
python -m more\_anns.train experiment=rot-aug task=classification task.use\_rotation\_aug=true
python -m more\_anns.train experiment=ori task=orientation task.similarity\_layer=classifier.3
python -m more\_anns.train experiment=multi-task task=multitask task.similarity\_layer=classifier.3
```

Requires access to ImageNet via HuggingFace Datasets (`ILSVRC/imagenet-1k`). You must accept the dataset terms and authenticate with `huggingface-cli login`.

Training outputs (checkpoints, TensorBoard logs, config) are written to `cache/more\_anns/{experiment}/` (e.g. `cache/more\_anns/rot-aug/`). These are generated artifacts; create them by training the models or place existing checkpoints there before running analyses that load trained ANN variants.

### `data/`

Please download the data from [here](https://janelia.figshare.com/articles/dataset/Calcium_imaging_data_for_Equivariant_neuronal_populations_enable_simultaneous_tuning_and_invariance_/29955530) and place it inside the `data` folder (see repository structure above). The `data` folder contains:

* Neuronal calcium imaging data (3 mice)
* Neuronal positions and area boundaries (3 mice)
* Visual stimuli

### `params/`

`model\_params.csv` specifies which models to analyze:

* `network`: Model or mouse name (incl. date of recording)
* `layer`: Layer or visual area
* `monitor`: Linear dimension of input image (for pixels and ANNs) or monitor index (for mice, 1: front)

`fixed\_params.csv` contains analysis parameters:

* `stimulus`: Stimulus set identifier
* `corr\_thre`: Correlation threshold for responsive neurons (0.1)
* `n\_procs`: Dimensionality for equivariance analysis (20)
* `n\_neurons`: Number of neurons to sample (250)
* `n\_max`: Maximum number of random samples (100)
* `inv\_snr`: Inverse signal-to-noise ratio (3.9)

Several notebooks read their own parameter variants alongside the base files:

* `\*\_all.csv` (`model\_params\_all.csv`, `fixed\_params\_all.csv`) — used by `plot\_extended\_figures.ipynb`
* `\*\_gain.csv` (`model\_params\_gain.csv`, `fixed\_params\_gain.csv`) — used by `gain\_model\_analyses.ipynb`
* `\*\_anns.csv` (`model\_params\_anns.csv`, `fixed\_params\_anns.csv`) — present but not currently loaded by any notebook

## Output

The analysis pipeline creates the following output directories under the project root:

* **`cache/`**: Intermediate results per model — averaged responses, split-half correlations, equivariance projections, decoder results. Organized as `cache/{model\_name}/{layer\_name}\_\*.{p,csv}`.
* **`cache\_gain/`**: Intermediate results for the gain model analyses.
* **`results/`**: Main paper figures (PDF) and significance tables (CSV).
* **`results\_extended/`**: Extended/supplementary figures (PDF).
* **`results\_extended\_gain/`**: Gain model figures (PDF), produced by `gain\_model\_analyses.ipynb`.
* **`results\_extended\_anns/`**: Extended ANN figures (PDF).

See [`docs/paper\_figs.md`](docs/paper_figs.md) for the full mapping of output files to paper figures.

### Checkpoint path convention

Trained AlexNet checkpoints are expected at `cache/more\_anns/{experiment}/best\_model.pth`. The default `params/model\_params.csv` in this checkout does not include the trained ANN variants; to analyze them through `model\_loader`, add rows whose `network` value is `alt\_alexnet\_{experiment}`. For example, `alt\_alexnet\_rot-aug` loads `cache/more\_anns/rot-aug/best\_model.pth`, `alt\_alexnet\_ori` loads `cache/more\_anns/ori/best\_model.pth`, and `alt\_alexnet\_multi-task` loads `cache/more\_anns/multi-task/best\_model.pth`.

The robustness cells in `src/more\_anns\_analyses.ipynb` also load those same checkpoint paths directly. They will fail until the corresponding `best\_model.pth` files exist.

## Installation

### Requirements

Python 3.9+ with dependencies pinned in `requirements.txt` (read dynamically by `pyproject.toml`). The code has primarily been tested on Windows 11.

### Setup

1. **Clone the repository:**

```bash
   git clone https://github.com/romani-lab/Equivariance-in-neural-networks.git
   cd Equivariance-in-neural-networks
   ```

2. **Install the package and its dependencies:**

```bash
   pip install -e .
   ```

   This installs the pinned dependencies from `requirements.txt` and makes the `src/` packages (`utils`, `more\_anns`) importable. To install only the dependencies without the editable package, use `pip install -r requirements.txt` instead.

3. **Download the data:**
Download the calcium imaging dataset from [figshare](https://doi.org/10.25378/janelia.29955530) and place it inside the `data/` folder (see [Repository Structure](#repository-structure) and the [`data/`](#data) section below for the expected layout).

   Using a fresh virtual environment (e.g. `python -m venv` or conda) is recommended to keep the pinned versions isolated.

   ### Key Dependencies

* **Core scientific computing**: NumPy (1.26.4), SciPy (1.12.0), pandas (2.2.1)
* **Machine learning**: scikit-learn (1.3.0)
* **Image processing**: Pillow (10.3.0), scikit-image (0.22.0)
* **Visualization**: Plotly (5.19.0), kaleido (pin `kaleido==0.1.0.post1`; 0.2.1 hangs on Windows)
* **Deep learning**: PyTorch (2.2.2), torchvision (0.17.2), [torch-intermediate-layer-getter](https://github.com/sebamenabar/Pytorch-IntermediateLayerGetter)
* **Training infrastructure**: Hydra, Datasets (HuggingFace), TensorBoard

