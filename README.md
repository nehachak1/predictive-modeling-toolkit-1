# Predictive Modeling Toolkit

A machine learning toolkit implementing fundamental supervised learning algorithms from scratch using Python and NumPy.

## Features

- k-Nearest Neighbors (k-NN)
  - Classification
  - Regression
  - Euclidean and Manhattan distance metrics

- Logistic Regression
  - Binary classification
  - Gradient descent optimization

- Linear Regression
  - Ordinary Least Squares
  - Ridge (L2) Regularization

## Project Structure

```text
.
├── src/
│   ├── methods/
│   │   ├── knn.py
│   │   ├── logistic_regression.py
│   │   ├── linear_regression.py
│   │   └── dummy_methods.py
│   └── utils.py
├── data/
├── main.py
└── README.md
```

## Installation

```bash
git clone https://github.com/nehachak1/predictive-modeling-toolkit.git
cd predictive-modeling-toolkit

pip install numpy matplotlib
```

## Usage

### Classification

```bash
python main.py \
    --task classification \
    --method knn \
    --K 5
```

### Logistic Regression

```bash
python main.py \
    --task classification \
    --method logistic_regression \
    --lr 0.01 \
    --max_iters 1000
```

### Regression

```bash
python main.py \
    --task regression \
    --method linear_regression
```

### Hyperparameter Visualization

```bash
python main.py \
    --task classification \
    --method knn \
    --show_plots
```

## Evaluation Metrics

### Classification
- Accuracy
- Macro F1-Score

### Regression
- Mean Squared Error (MSE)

## Technologies

- Python
- NumPy
- Matplotlib

## Authors

- Neha Chakraborty
- Guillaume Marie Lepin
