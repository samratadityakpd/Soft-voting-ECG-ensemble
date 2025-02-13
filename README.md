# Soft-voting-ECG-ensemble

Deep learning models (GRU, LSTM, CNN, SRU) for ECG arrhythmia classification using the MIT-BIH dataset. Soft-voting ensemble for enhanced accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Ensemble Method](#ensemble-method)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project explores the application of deep learning techniques to classify heart arrhythmias from Electrocardiogram (ECG) signals.  It implements and compares several models, including Gated Recurrent Units (GRUs), Long Short-Term Memory networks (LSTMs), 1D Convolutional Neural Networks (CNNs), and a Simplified Recurrent Unit (SRU).  A soft-voting ensemble combines the predictions of these individual models to achieve higher classification accuracy.  The project utilizes the widely used MIT-BIH Arrhythmia Dataset.

## Dataset

The [MIT-BIH Arrhythmia Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat/code) is a standard benchmark for evaluating arrhythmia classification algorithms.  It contains recordings of 48 half-hour excerpts of ambulatory ECGs obtained from 47 subjects studied by the BIH Arrhythmia Laboratory.  *(Include details about the classes, number of samples, etc. if you have space)*

## Models

This project implements and compares the following deep learning models:

*   **GRU (Gated Recurrent Unit):**  A type of recurrent neural network that is effective for sequential data.
*   **LSTM (Long Short-Term Memory):** Another type of recurrent neural network designed to address the vanishing gradient problem.
*   **1D CNN (1-Dimensional Convolutional Neural Network):**  A convolutional network adapted for one-dimensional time-series data.
*   **SRU (Simplified Recurrent Unit):**  *(Describe your SRU model and its architecture)*

## Ensemble Method

A soft-voting ensemble is used to combine the predictions of the individual models.  Soft voting averages the probabilities predicted by each model for each class, and the class with the highest average probability is selected as the final prediction.

## Results

The following table shows the classification report for the ensemble model:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 | 0.98 | 1.00 | 0.99 | 18117 |
| 1 | 0.99 | 0.57 | 0.72 | 556 |
| 2 | 0.97 | 0.94 | 0.96 | 1448 |
| 3 | 0.92 | 0.64 | 0.75 | 162 |
| 4 | 1.00 | 0.98 | 0.99 | 1608 |
| **Accuracy** |  |  | **0.98** | **21891** |
| **Macro Avg** | 0.97 | 0.82 | 0.88 | 21891 |
| **Weighted Avg** | 0.98 | 0.98 | 0.98 | 21891 |

The ensemble model achieved high accuracy across most classes, with the lowest F1-score observed for class 1 due to lower recall

## Requirements

asttokens==3.0.0
colorama==0.4.6
comm==0.2.2
contourpy==1.3.1
cycler==0.12.1
debugpy==1.8.12
decorator==5.1.1
executing==2.2.0
filelock==3.13.1
fonttools==4.55.8
fsspec==2024.6.1
ipykernel==6.29.5
ipynb-py-convert==0.4.6
ipython==8.32.0
jedi==0.19.2
Jinja2==3.1.4
joblib==1.4.2
jupyter_client==8.6.3
jupyter_core==5.7.2
kiwisolver==1.4.8
MarkupSafe==2.1.5
matplotlib==3.10.0
matplotlib-inline==0.1.7
mlxtend==0.23.4
mpmath==1.3.0
narwhals==1.24.1
nest-asyncio==1.6.0
networkx==3.3
ninja==1.11.1.3
numpy==2.2.2
packaging==24.2
pandas==2.2.3
parso==0.8.4
pillow==11.1.0
platformdirs==4.3.6
plotly==6.0.0
prompt_toolkit==3.0.50
psutil==6.1.1
pure_eval==0.2.3
Pygments==2.19.1
pyparsing==3.2.1
python-dateutil==2.9.0.post0
pytz==2025.1
pywin32==308
pyzmq==26.2.1
scikit-learn==1.6.1
scipy==1.15.1
seaborn==0.13.2
setuptools==70.2.0
six==1.17.0
sru==2.6.0
stack-data==0.6.3
sympy==1.13.1
threadpoolctl==3.5.0
torch==2.6.0+cu126
torchaudio==2.6.0+cu126
torchvision==0.21.0+cu126
tornado==6.4.2
traitlets==5.14.3
typing_extensions==4.12.2
tzdata==2025.1
wcwidth==0.2.13


You can install the requirements using:

```bash
pip install -r requirements.txt
```
## Usage

1. **Clone the repository:**

```bash
git clone [https://github.com/samratadityakpd/Soft-voting-ECG-ensemble.git](https://github.com/samratadityakpd/Soft-voting-ECG-ensemble.git)  # Replace with your actual repository URL
cd Soft-voting-ECG-ensemble
```

2. Create a virtual environment (recommended):
   
``` bash
python3 -m venv .venv        # Create the virtual environment
source .venv/bin/activate  # Activate on Linux/macOS
.venv\Scripts\activate     # Activate on Windows
```

3. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the notebook:

   ```bash
   jupyter notebook ECG_classification_ensemble.ipynb
   ```

##Contributing
Contributions are welcome!  Please feel free to open issues or submit pull requests.

Bug reports: If you encounter any bugs, please create a detailed issue describing the problem and steps to reproduce it.
Feature requests: If you have ideas for new features or improvements, please open an issue to discuss them.
Pull requests: If you'd like to contribute code, please submit a pull request with a clear description of your changes. Make sure to follow the project's coding style and conventions.

##Acknowledgements
The authors of the MIT-BIH Arrhythmia Dataset.
The developers of the open-source libraries used in this project, including:
PyTorch
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Plotly
mlxtend
and all other libraries listed in requirements.txt.
