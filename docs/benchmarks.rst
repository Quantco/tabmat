Benchmarks
==========
    
To generate the data to run all benchmarks, run: ``python src/quantcore/matrix/benchmark/generate_matrices.py``.

For more info on the benchmark CLI: ``python src/quantcore/matrix/benchmark/main.py --help``.

Performance
^^^^^^^^^^^

Dense matrix, 100k x 1k:

.. image:: _static/dense_times.png
   :width: 700

One-hot encoded categorical variable, 1M x 100k:

.. image:: _static/one_cat_times.png
   :width: 700

Sparse matrix, 1M x 1k:

.. image:: _static/sparse_times.png
   :width: 700

Two categorical matrices, 1M x 2k:

.. image:: _static/two_cat_times.png
   :width: 700

.. image:: _static/dense_cat_times.png
   :width: 700

