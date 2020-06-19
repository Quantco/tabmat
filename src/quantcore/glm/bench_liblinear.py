import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
from scipy import sparse as sps
from sklearn.linear_model import LogisticRegression

from .util import benchmark_convergence_tolerance, runtime


def build_and_fit(model_args, train_args):
    clf = LogisticRegression(**model_args)
    clf.fit(**train_args)
    return clf


def liblinear_bench(
    dat: Dict[str, Union[sps.spmatrix, np.ndarray]],
    distribution: str,
    alpha: float,
    l1_ratio: float,
    iterations: int,
    cv: bool,
    print_diagnostics: bool = True,  # ineffective here
    reg_multiplier: Optional[float] = None,
) -> Dict[str, Any]:

    result: Dict = dict()

    X = dat["X"]
    if not isinstance(X, np.ndarray) and not isinstance(X, sps.spmatrix):
        warnings.warn(
            "liblinear requires data as scipy.sparse matrix or numpy array. Skipping."
        )
        return result

    if distribution != "binomial":
        warnings.warn("liblinear only supports binomial")
        return result

    if l1_ratio == 1 and alpha > 0:
        pen = "l1"
    elif l1_ratio == 0 and alpha > 0:
        pen = "l2"
    else:
        warnings.warn(
            "liblinear only supports lasso and ridge regression with positive alpha"
        )
        return result

    if cv:
        warnings.warn("liblinear does not yet support CV")
        return result

    model_args = dict(
        penalty=pen,
        tol=benchmark_convergence_tolerance,
        C=1 / alpha if reg_multiplier is None else 1 / (alpha * reg_multiplier),
        fit_intercept="offset" in dat.keys(),
        # intercept_scaling=dat["offset"] if "offset" in dat.keys() else 1,
        solver="liblinear",
    )

    train_args = dict(
        X=X,
        y=dat["y"].astype(np.int64).copy(),
        sample_weight=dat["weights"] if "weights" in dat.keys() else None,
    )

    result["runtime"], m = runtime(build_and_fit, iterations, model_args, train_args)
    result["intercept"] = m.intercept_
    result["coef"] = np.squeeze(m.coef_)
    result["n_iter"] = m.n_iter_

    return result