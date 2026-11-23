# ML Algorithms from Scratch

Implementations of classic machine learning and deep learning algorithms, written from first principles. Most classical models use only NumPy (plus SciPy in a few places) and pyTorch for the deep learning models.

## Contents

### Supervised learning

| Folder | File | Implemented |
|---|---|---|
| [Linear-Regression](https://github.com/GeorgiosLymperis/ml_algorithms/tree/master/Linear-Regression) | `linear_regression.py` | `LinearRegression`, `RidgeRegression`, `LassoRegression`, `ElasticNet`, `LARS` |
| [Logistic-Regression](https://github.com/GeorgiosLymperis/ml_algorithms/tree/master/Logistic-Regression) | `logistic_regression.py` | `LogisticRegression` |
| [Naive-Bayes](https://github.com/GeorgiosLymperis/ml_algorithms/tree/master/Naive-Bayes) | `naive_bayes.py` | `GaussianNB`, `MultinomialNB`, `BernoulliNB` |
| [kNN](https://github.com/GeorgiosLymperis/ml_algorithms/tree/master/kNN) | `knn.py` | `kNNClassifier`, `kNNRegressor` |
| [SVM](https://github.com/GeorgiosLymperis/ml_algorithms/tree/master/SVM) | `svm.py` | `SVM` (linear), `KernelSVM` with `LinearKernel`, `RBFKernel`, `PolynomialKernel` |
| [LDA](https://github.com/GeorgiosLymperis/ml_algorithms/tree/master/LDA) | `lda.py` | Linear Discriminant Analysis via SVD (`lda_fit_svd`, `lda_predict`) |

### Tree-based models (`Trees/`)

| File | Implemented |
|---|---|
| `cart.py` | `DecisionTreeClassifier`, `DecisionTreeRegressor` (CART) |
| `random_forest.py` | `RandomForestClassifier`, `RandomForestRegressor` |
| `extra_tree.py` | `ExtraTreeClassifier/Regressor`, `ExtraTreesClassifier/Regressor` |
| `gradient_boosting.py` | `GradientBoostingClassifier`, `GradientBoostingRegressor` |
| `xgboost.py` | `XGBoostClassifier`, `XGBoostRegressor` (second-order boosting with squared and logistic loss) |
| `isolation_forest.py` | `IsolationTree`, `IsolationForest` (anomaly detection) |

`random_forest.py` and `gradient_boosting.py` import from `cart.py`.

### Unsupervised learning

| Folder | File | Implemented |
|---|---|---|
| [KMeans](https://github.com/GeorgiosLymperis/ml_algorithms/tree/master/KMeans) | `kmeans.py` | `MyKMeans` |
| [GMM](https://github.com/GeorgiosLymperis/ml_algorithms/tree/master/GMM) | `gmm.py` | `GaussianMixture` (EM) |
| [clustering](https://github.com/GeorgiosLymperis/ml_algorithms/tree/master/clustering) | `agglomerative.py` | `AgglomerativeClustering` |
| [clustering](https://github.com/GeorgiosLymperis/ml_algorithms/tree/master/clustering) | `dbscan.py` | `DBSCAN` |
| [PCA](https://github.com/GeorgiosLymperis/ml_algorithms/tree/master/PCA) | `pca.py` | `PCA`, `SparsePCA`, `KernelPCA` (RBF kernel) |

`clustering/dbscan.py` does not have a notebook yet.

### Neural networks

| Folder | File | Implemented |
|---|---|---|
| [neural-networks](https://github.com/GeorgiosLymperis/ml_algorithms/tree/master/neural-networks) | `neural_network.py` | Fully connected `NeuralNetwork` with `Sigmoid` and `ReLU` activations (NumPy) |
| [CNN](https://github.com/GeorgiosLymperis/ml_algorithms/tree/master/CNN) | `layers.py` | `Conv2D`, `MaxPool2D`, `BatchNorm2D`, `Dense`, `ReLU`, `Flatten`, `SoftmaxCrossEntropy`, and a `LeNet` model (NumPy) |

### Recurrent networks and attention (`RNNs/`, PyTorch)

| File | Implemented |
|---|---|
| `elman.py` | `ElmanCell` (vanilla RNN) |
| `lstm.py` | `LSTMCell`, `PeepholeLSTMCell`, `BiLSTM`, `DeepLSTM`, `DeepBiLSTM` |
| `gru.py` | `GRUCell`, `BiGRU`, `DeepGRU`, `DeepBiGRU` |
| `indrnn.py` | `IndRNNCell`, `IndRNN` |
| `seq2seq.py` | LSTM encoder–decoder `Seq2Seq` |
| `bahdanau.py` | Additive (Bahdanau) attention and `Seq2SeqBahdanau` |
| `luong.py` | Multiplicative (Luong) attention and `LuongAttentionModel` |
| `pointer.py` | `PointerNetwork` |
| `elmo.py` | `ELMo` and `ELMoEmbedder` (bidirectional language-model embeddings) |

Several of these files import from `lstm.py` or `gru.py`, so run them from inside `RNNs/`.

## Requirements

- Python 3.9+
- `numpy`, `scipy`: used by the implementations
- `torch`: used by `RNNs/` and parts of the CNN notebook
- `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`: used by the notebooks for datasets, baselines and plots

```bash
pip install numpy scipy scikit-learn matplotlib seaborn torch jupyter
```

## Usage

Run notebooks and scripts from inside the algorithm's own folder:

```bash
cd Trees
jupyter notebook random_forest.ipynb
```

Most models follow the scikit-learn `fit` / `predict` convention:

```python
import sys; sys.path.append("KMeans")
from kmeans import MyKMeans

model = MyKMeans(n_clusters=3)
model.fit(X)
labels = model.predict(X)
```

## Utilities

`cleanup.sh` removes `__pycache__/`, `.pytest_cache/` and `.vscode/` directories:

```bash
bash cleanup.sh
```
