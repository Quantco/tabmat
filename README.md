# Efficient matrix representations for working with tabular data

![CI](https://github.com/Quantco/quantcore.matrix/workflows/CI/badge.svg)

## Installation
For development, you should do an editable installation: 

```bash
# First, make sure you have conda-forge as your primary conda channel:
conda config --add channels conda-forge
# And install pre-commit
conda install -y pre-commit

git clone git@github.com:Quantco/quantcore.matrix.git
cd quantcore.matrix

# Set up our pre-commit hooks for black, mypy, isort and flake8.
pre-commit install

# Set up a conda environment with name "quantcore.matrix"
conda install mamba=0.2.12
mamba env create

# Install this package in editable mode. 
conda activate quantcore.matrix
pip install --no-use-pep517 --disable-pip-version-check -e .
```

## Use case

Data used in economics, actuarial science, and many other fields is often tabular, containing rows and columns. Further properties are also common:
- Tabular data often contains categorical data, often represented after processing as many columns of indicator values created by "one-hot encoding."
- It often contains a mix of dense columns and sparse columns, perhaps due to one-hot encoding.
- It often is very sparse.

High-performance statistical applications often require fast computation of certain operations, such as
- Computing "sandwich products" of the data, `transpose(X) @ diag(d) @ X`. A sandwich product shows up in the solution to weighted least squares, as well as in the Hessian of the likelihood in generalized linear models such as Poisson regression.
- Matrix-vector products
- Operating on one column at a time

Additionally, it is often desirable to normalize predictors for greater optimizer efficiency and numerical stability in coordinate descent and in other machine learning algorithms.

## This library and its design

We designed this library with these use cases in mind. We built this library first for estimating generalized linear models, but expect it will be useful in a variety of econometric and statistical use cases. This library was borne out of our need for speed, and its unified API is motivated by the annoyance by having to write repeated checks for which type of matrix-like object you are operating on.

Design principles:
- Speed and memory efficiency are paramount.
- You don't need to sacrifice functionality by using this library: `DenseMatrix` and `SparseMatrix` subclass `np.ndarray` and `scipy.sparse.csc_matrix` respectively, and inherit their behavior wherever it is not improved on.
- As much as possible, syntax follows Numpy syntax, and dimension-reducing operations (like `sum`) return Numpy arrays, following Numpy dimensions about the dimensions of results. The aim is to make these classes as close as possible to being drop-in replacements for numpy ndarray.  This is not always possible, however, due to the differing APIs of numpy ndarray and scipy sparse.
- Other operations, such as `toarray`, mimic Scipy sparse syntax.
- All matrix classes support matrix products, sandwich products, and `getcol`.

Individual subclasses may support significantly more operations.

## Matrix types
- `DenseMatrix` represents dense matrices, subclassing numpy nparray.  It additionally supports methods `getcol`, `toarray`, `sandwich`, `standardize`, and `unstandardize`.
- `SparseMatrix` represents column-major sparse data, subclassing `scipy.sparse.csc_matrix`. It additionally supports methods `sandwich` and `standardize`, and it's `dot` method (e.g. `@`) calls MKL's sparse dot product in the case of matrix-vector products, which is faster.
- `CategoricalMatrix` represents one-hot encoded categorical matrices. Because all the non-zeros in these matrices are ones and because each row has only one non-zero, the data can be represented and multiplied much more efficiently than a generic sparse matrix.
- `SplitMatrix` represents matrices with both dense, sparse and categorical parts, allowing for a significant speedup in matrix multiplications.
- `StandardizedMatrix` efficiently and sparsely represents a matrix that has had its column normalized to have mean zero and variance one. Even if the underlying matrix is sparse, such a normalized matrix will be dense. However, by storing the scaling and shifting factors separately, `StandardizedMatrix` retains the original matrix sparsity. 

![Wide data set](images/wide_data_sandwich.png)

## Benchmarks

[See here for detailed benchmarking.]()

## API documentation

[See here for detailed API documentation.]()
