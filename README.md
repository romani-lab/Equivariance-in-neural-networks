# Equivariant neuronal populations enable simultaneous tuning and invariance

This repository contains partial code for analyzing equivariance in neural networks, be it biological or artificial.

## Overview

This project investigates how neuronal populations may carry equivariant representations to simultaneously support tuning and invariance. We analyze experimental calcium imaging data from mouse visual cortex and the convolutional neural network AlexNet.

## Repository Structure

```
├── src/                    # Source code and analysis notebooks
│   ├── utils/             # Core utility functions
│   ├── plot_figures.ipynb # Main notebook for generating figures
│   └── create_jobs_array.ipynb
├── data/                  # Experimental data (calcium imaging)
│   ├── 05_18_2023/       # Recording session data
│   ├── 05_30_2023/
│   ├── 05_31_2023/
│   ├── retinotopy/       # Visual area boundaries
│   └── stimuli/          # Visual stimuli used in experiments
├── params/               # Configuration files
│   ├── model_params.csv  # Model and layer specifications
│   └── fixed_params.csv  # Analysis parameters
├── cache/                # Intermediate analysis results
└── results/              # Generated figures and statistics
```

## Quick Start

The primary analysis and figure generation is contained in:
```
src/plot_figures.ipynb
```

This notebook generates the main analysis figures from the preprint, including:
- **Figure 1c-e**: Visual stimuli, reliably responsive neurons in mouse recordings, and decoding errors
- **Figure 3b-c**: Equivariance examples, subspace summaries, and subspace decoding errors
- **Figure 4a-b**: Equivariant subspace projections and Pearson correlation

## Main Directories

### `src/utils/`

- **`stimulus_loader.py`**: Visual stimulus handling and preprocessing
- **`data_loader.py`**: Loading and preprocessing calcium imaging data
- **`model_loader.py`**: Interface for artificial neural network models
- **`neuron_selectivity.py`**: Responsive neuron detection and reliability metrics
- **`decoder_functions.py`**: Linear decoding and cross-validation
- **`equivariance_functions.py`**: Mathematical framework for equivariance analysis
- **`plotting_functions.py`**: Visualization and figure generation
- **`helper.py`**: Mathematical utilities and statistical functions

### `data/`

Please download the data from [here](https://janelia.figshare.com/articles/dataset/Calcium_imaging_data_for_Equivariant_neuronal_populations_enable_simultaneous_tuning_and_invariance_/29955530) and place it inside the `data` folder (see repository structure above). The `data` folder contains the following information:
- Neuronal Calcium imaging data (3 mice)
- Neuronal positions and area boundaries (3 mice)
- Visual stimuli

### `params/`

`model_params.csv` specifies which models to analyze:
- `network`: Model or mouse name (incl. date of recording)
- `layer`: Layer or visual area
- `monitor`: Linear dimension of input image or monitor index (1: front)

`fixed_params.csv` contains analysis parameters
- `stimulus`: Stimulus set identifier
- `corr_thre`: Correlation threshold for responsive neurons (0.1)
- `n_procs`: Dimensionality for equivariance analysis (20)
- `n_neurons`: Number of neurons to sample (250)
- `n_max`: Maximum number of random samples (100)
- `inv_snr`: Inverse signal-to-noise ratio (3.9)

## Output

Results are saved to:
- **`results/`**: Generated figures (PDF and CSV)
- **`cache/`**: Intermediate analysis results for faster re-runs (pickle and CSV)

## Installation & Dependencies

### Requirements

This project requires Python 3.10+ and several scientific computing packages. All dependencies are listed in `requirements.txt` with specific versions tested for compatibility. The code has only been tested on Windows 11.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hoellerj/equivariance.git
   cd equivariance
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or if using conda:
   ```bash
   conda install --file requirements.txt
   ```

### Key Dependencies

- **Core scientific computing**: NumPy (1.26.4), SciPy (1.12.0), pandas (2.2.1)
- **Machine learning**: scikit-learn (1.3.0)
- **Image processing**: Pillow (10.3.0), scikit-image (0.22.0)
- **Visualization**: Plotly (5.19.0), kaleido (0.2.1)
- **Deep learning**: PyTorch (2.2.2), torchvision (0.17.2)
- **Neural network utilities**: [torch-intermediate-layer-getter](https://github.com/sebamenabar/Pytorch-IntermediateLayerGetter)
- **Jupyter support**: Full Jupyter ecosystem for running notebooks