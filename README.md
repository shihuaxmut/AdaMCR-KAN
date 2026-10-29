# AdaMCR-KAN
![Figure1_framework](Figure1_framework.png)

## Environment Setup

### Requirements

AdaMCR-KAN is implemented in Python and PyTorch. The recommended environment is listed below.

| Component | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.4.0+cu118 |
| TorchVision | 0.19.0+cu118 |
| TorchAudio | 2.4.0+cu118 |
| CUDA | 11.8 |
| Transformers | 4.36.2 |
| NumPy | 1.26.4 |
| scikit-learn | 1.9.0 |

### Installation

Create and activate the Conda environment:

```bash
conda create -n AdaMCR_KAN python=3.11 -y
conda activate AdaMCR_KAN
```

Install the required dependencies using pip:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
## Usage

Before running the scripts, ensure that the required dependencies are installed and the Conda environment is activated.

```bash
conda activate AdaMCR_KAN
cd code
```

### 1. MHC-I Binding Prediction

**Step 1: Train the MHC model**

Run the following command to train the MHC-I binding prediction model:

```bash
python Train_MHC.py
```

Before training, check the `DEFAULT_CONFIG_PATH` parameter in `Train_MHC.py` and ensure that it points to the correct configuration file. Verify that the paths and parameters specified in the configuration file are consistent with your dataset and experimental settings.

**Step 2: Test the MHC model**

After training, run the following command to evaluate the model:

```bash
python Test_MHC.py
```

Before testing, check the `DEFAULT_CONFIG_PATH` parameter in `Test_MHC.py` and update it if necessary. Ensure that the configuration file specifies the correct test dataset, model checkpoint, and other required parameters.

### 2. TCR–pMHC-I Recognition Prediction

**Step 1: Train the TCR model**

Run the following command to train the TCR–pMHC-I recognition model:

```bash
python Train_TCR.py
```

Before training, verify that `DEFAULT_CONFIG_PATH` in `Train_TCR.py` points to the appropriate configuration file. Check the configuration settings, including dataset paths, training parameters, and output directories.

**Step 2: Test the TCR model**

After training, evaluate the model using:

```bash
python Test_TCR.py
```

Before testing, verify the `DEFAULT_CONFIG_PATH` parameter in `Test_TCR.py` and ensure that the corresponding configuration file contains the correct test dataset paths, trained model checkpoint, and evaluation settings.

## Datasets, Model Configurations, and Results

The datasets, model configurations, and experimental results associated with AdaMCR-KAN are available on Hugging Face:

[**AdaMCR-KAN Results and Data Repository**](https://huggingface.co/datasets/thechen/AdaMCR-KAN-result/tree/main/data)

Please refer to this repository for the relevant datasets, model parameter configurations, and experimental results.

### Notes

- **Configuration paths:** Update `DEFAULT_CONFIG_PATH` in each script to match the actual location of its configuration file. Use valid paths for your local environment.
- **Configuration consistency:** Ensure that the configuration files specify the correct datasets, model checkpoints, and output directories before running each script.
- **Execution order:** Train and evaluate the MHC-I binding prediction model first, then train and evaluate the TCR–pMHC-I recognition model, following the corresponding configuration requirements.
- **Working directory:** Run the commands from the `code` directory unless the scripts explicitly support another working directory.


## Abstract
TCR–pMHC-I recognition is governed by intricate residue-wise cross-molecular interactions and sequence dependencies, and the extreme polymorphism of HLA-I molecules brings great difficulty to accurate prediction. We propose AdaMCR-KAN, a KAN-based model for TCR–pMHC-I recognition prediction. It uses a hierarchical residue interaction encoder (HRIEncoder) to extract multi-level sequence features, bidirectional cross-attention with adaptive head gating, and a novel cross-interaction gating (CIG) module to dynamically filter redundant interactions and emphasize key binding patterns. Benchmark results show AdaMCR-KAN achieves superior performance and strong generalization across diverse peptides, HLA alleles and test scenarios. Further motif analysis and saturation mutagenesis validation confirm the model learns biologically meaningful residue binding preferences.
