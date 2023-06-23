import functools
import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Union

import numpy
import pandas
from formulaic import ModelMatrix, ModelSpec
from formulaic.materializers import FormulaMaterializer
from formulaic.materializers.base import EncodedTermStructure
from formulaic.materializers.types import FactorValues, NAAction
from interface_meta import override
from scipy import sparse

import tabmat as tm

from .categorical_matrix import CategoricalMatrix
from .dense_matrix import DenseMatrix
from .matrix_base import MatrixBase
from .sparse_matrix import SparseMatrix
from .split_matrix import SplitMatrix


class TabmatMaterializer(FormulaMaterializer):
    """Materializer for pandas input and tabmat output."""

    REGISTER_NAME = "tabmat"
    REGISTER_INPUTS = ("pandas.core.frame.DataFrame",)
    REGISTER_OUTPUTS = "tabmat"

    @override
    def _init(self):
        self.interaction_separator = self.params.get("interaction_separator", ":")
        self.categorical_format = self.params.get(
            "categorical_format", "{name}[T.{category}]"
        )
        self.intercept_name = self.params.get("intercept_name", "Intercept")
        self.dtype = self.params.get("dtype", numpy.float64)
        self.sparse_threshold = self.params.get("sparse_threshold", 0.1)
        self.cat_threshold = self.params.get("cat_threshold", 4)

        # We can override formulaic's C() function here
        self.context["C"] = _C

    @override
    def _is_categorical(self, values):
        if isinstance(values, (pandas.Series, pandas.Categorical)):
            return values.dtype == object or isinstance(
                values.dtype, pandas.CategoricalDtype
            )
        return super()._is_categorical(values)

    @override
    def _check_for_nulls(self, name, values, na_action, drop_rows):
        if na_action is NAAction.IGNORE:
            return

        if na_action is NAAction.RAISE:
            if isinstance(values, pandas.Series) and values.isnull().values.any():
                raise ValueError(f"`{name}` contains null values after evaluation.")

        elif na_action is NAAction.DROP:
            if isinstance(values, pandas.Series):
                drop_rows.update(numpy.flatnonzero(values.isnull().values))

        else:
            raise ValueError(
                f"Do not know how to interpret `na_action` = {repr(na_action)}."
            )

    @override
    def _encode_constant(self, value, metadata, encoder_state, spec, drop_rows):
        series = value * numpy.ones(self.nrows - len(drop_rows))
        return _InteractableDenseVector(series, name=self.intercept_name)

    @override
    def _encode_numerical(self, values, metadata, encoder_state, spec, drop_rows):
        if drop_rows:
            values = values.drop(index=values.index[drop_rows])
        if isinstance(values, pandas.Series):
            values = values.to_numpy().astype(self.dtype)
        if (values != 0).mean() <= self.sparse_threshold:
            return _InteractableSparseVector(
                sparse.csc_matrix(values[:, numpy.newaxis])
            )
        else:
            return _InteractableDenseVector(values)

    @override
    def _encode_categorical(
        self, values, metadata, encoder_state, spec, drop_rows, reduced_rank=False
    ):
        # We do not do any encoding here as it is handled by tabmat
        if drop_rows:
            values = values.drop(index=values.index[drop_rows])
        cat = pandas.Categorical(values._values)
        return _InteractableCategoricalVector.from_categorical(
            cat, reduced_rank=reduced_rank
        )

    @override
    def _combine_columns(self, cols, spec, drop_rows):
        # Special case no columns
        if not cols:
            values = numpy.empty((self.data.shape[0], 0), dtype=self.dtype)
            return DenseMatrix(values)

        # Otherwise, concatenate columns into SplitMatrix
        return SplitMatrix(
            [
                col[1].to_tabmat(self.dtype, self.sparse_threshold, self.cat_threshold)
                for col in cols
            ]
        )

    # Have to override this because of column names
    # (and possibly intercept later on)
    @override
    def _build_model_matrix(self, spec: ModelSpec, drop_rows):
        # Step 0: Apply any requested column/term clustering
        # This must happen before Step 1 otherwise the greedy rank reduction
        # below would result in a different outcome than if the columns had
        # always been in the generated order.
        terms = self._cluster_terms(spec.formula, cluster_by=spec.cluster_by)

        # Step 1: Determine strategy to maintain structural full-rankness of output matrix
        scoped_terms_for_terms = self._get_scoped_terms(
            terms,
            ensure_full_rank=spec.ensure_full_rank,
        )

        # Step 2: Generate the columns which will be collated into the full matrix
        cols = []
        for term, scoped_terms in scoped_terms_for_terms:
            scoped_cols = OrderedDict()
            for scoped_term in scoped_terms:
                if not scoped_term.factors:
                    scoped_cols[
                        "Intercept"
                    ] = scoped_term.scale * self._encode_constant(
                        1, None, {}, spec, drop_rows
                    )
                else:
                    scoped_cols.update(
                        self._get_columns_for_term(
                            [
                                self._encode_evaled_factor(
                                    scoped_factor.factor,
                                    spec,
                                    drop_rows,
                                    reduced_rank=scoped_factor.reduced,
                                )
                                for scoped_factor in scoped_term.factors
                            ],
                            spec=spec,
                            scale=scoped_term.scale,
                        )
                    )
            cols.append((term, scoped_terms, scoped_cols))

        # Step 3: Populate remaining model spec fields
        if spec.structure:
            cols = self._enforce_structure(cols, spec, drop_rows)
        else:
            # for term, scoped_terms, columns in spec.structure:
            # expanded_columns = list(itertools.chain(colname_dict[col] for col in columns))
            # expanded_structure.append(
            #     EncodedTermStructure(term, scoped_terms, expanded_columns)
            # )

            spec = spec.update(
                structure=[
                    EncodedTermStructure(
                        term,
                        [st.copy(without_values=True) for st in scoped_terms],
                        # This is the only line that is different from the original:
                        list(
                            itertools.chain(
                                *(mat.get_names() for mat in scoped_cols.values())
                            )
                        ),
                    )
                    for term, scoped_terms, scoped_cols in cols
                ],
            )

        # Step 4: Collate factors into one ModelMatrix
        return ModelMatrix(
            self._combine_columns(
                [
                    (name, values)
                    for term, scoped_terms, scoped_cols in cols
                    for name, values in scoped_cols.items()
                ],
                spec=spec,
                drop_rows=drop_rows,
            ),
            spec=spec,
        )

    @override
    def _get_columns_for_term(self, factors, spec, scale=1):
        """Assemble the columns for a model matrix given factors and a scale."""
        out = OrderedDict()
        for reverse_product in itertools.product(
            *(factor.items() for factor in reversed(factors))
        ):
            product = reverse_product[::-1]
            out[":".join(p[0] for p in product)] = scale * functools.reduce(
                functools.partial(_interact, separator=self.interaction_separator),
                (
                    p[1].set_name(p[0], name_format=self.categorical_format)
                    for p in product
                ),
            )
        return out


class _InteractableVector(ABC):
    """Abstract base class for interactable vectors, which are mostly thin
    wrappers over numpy arrays, scipy sparse matrices and pandas categoricals.
    """

    name: Optional[str]

    @abstractmethod
    def to_tabmat(
        self,
        dtype: numpy.dtype,
        sparse_threshold: float,
        cat_threshold: int,
    ) -> MatrixBase:
        """Convert to an actual tabmat matrix."""
        pass

    @abstractmethod
    def get_names(self) -> List[str]:
        """Return the names of the columns represented by this vector.

        Returns
        -------
        List[str]
            The names of the columns represented by this vector.
        """
        pass

    @abstractmethod
    def set_name(self, name, name_format):
        """Set the name of the vector.

        Parameters
        ----------
        name : str
            The name to set.
        name_format : str
            The format string to use to format the name. Only used for
            categoricals. Has to include the placeholders ``{name}``
            and ``{category}``

        Returns
        -------
        self
            A reference to the vector itself.
        """
        pass


class _InteractableDenseVector(_InteractableVector):
    def __init__(self, values: numpy.ndarray, name: Optional[str] = None):
        self.values = values
        self.name = name

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return _InteractableDenseVector(
                values=self.values * other,
                name=self.name,
            )

    def to_tabmat(
        self,
        dtype: numpy.dtype = numpy.float64,
        sparse_threshold: float = 0.1,
        cat_threshold: int = 4,
    ) -> DenseMatrix:
        if (self.values != 0).mean() > sparse_threshold:
            return DenseMatrix(self.values)
        else:
            # Columns can become sparser, but not denser through interactions
            return SparseMatrix(sparse.csc_matrix(self.values[:, numpy.newaxis]))

    def get_names(self) -> List[str]:
        if self.name is None:
            raise RuntimeError("Name not set")
        return [self.name]

    def set_name(self, name, name_format=None) -> "_InteractableDenseVector":
        self.name = name
        return self


class _InteractableSparseVector(_InteractableVector):
    def __init__(self, values: sparse.csc_matrix, name: Optional[str] = None):
        self.values = values
        self.name = name

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return _InteractableSparseVector(
                values=self.values * other,
                name=self.name,
            )

    def to_tabmat(
        self,
        dtype: numpy.dtype = numpy.float64,
        sparse_threshold: float = 0.1,
        cat_threshold: int = 4,
    ) -> SparseMatrix:
        return SparseMatrix(self.values)

    def get_names(self) -> List[str]:
        if self.name is None:
            raise RuntimeError("Name not set")
        return [self.name]

    def set_name(self, name, name_format=None) -> "_InteractableSparseVector":
        self.name = name
        return self


class _InteractableCategoricalVector(_InteractableVector):
    def __init__(
        self,
        codes: numpy.ndarray,
        categories: List[str],
        multipliers: numpy.ndarray,
        name: Optional[str] = None,
    ):
        # sentinel values for codes:
        # -1: missing
        # -2: drop
        self.codes = codes
        self.categories = categories
        self.multipliers = multipliers
        self.name = name

    @classmethod
    def from_categorical(
        cls, cat: pandas.Categorical, reduced_rank: bool
    ) -> "_InteractableCategoricalVector":
        """Create an interactable categorical vector from a pandas categorical."""
        categories = list(cat.categories)
        codes = cat.codes.copy().astype(numpy.int64)
        if reduced_rank:
            codes[codes == 0] = -2
            codes[codes > 0] -= 1
            categories = categories[1:]
        return cls(
            codes=codes,
            categories=categories,
            multipliers=numpy.ones(len(cat.codes)),
        )

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return _InteractableCategoricalVector(
                categories=self.categories,
                codes=self.codes,
                multipliers=self.multipliers * other,
                name=self.name,
            )

    def to_tabmat(
        self,
        dtype: numpy.dtype = numpy.float64,
        sparse_threshold: float = 0.1,
        cat_threshold: int = 4,
    ) -> Union[CategoricalMatrix, SplitMatrix]:
        codes = self.codes.copy()
        categories = self.categories.copy()
        if -2 in self.codes:
            codes[codes >= 0] += 1
            codes[codes == -2] = 0
            categories.insert(0, "__drop__")
            drop_first = True
        else:
            drop_first = False

        cat = pandas.Categorical.from_codes(
            codes=codes,
            categories=categories,
            ordered=False,
        )

        categorical_part = CategoricalMatrix(cat, drop_first=drop_first, dtype=dtype)

        if (self.codes == -2).all():
            # All values are dropped
            return DenseMatrix(numpy.empty((len(codes), 0), dtype=dtype))
        elif (self.multipliers == 1).all() and len(categories) >= cat_threshold:
            return categorical_part
        else:
            sparse_matrix = sparse.csc_matrix(
                categorical_part.tocsr().multiply(self.multipliers[:, numpy.newaxis])
            )
            return tm.from_csc(sparse_matrix, threshold=sparse_threshold)

    def get_names(self) -> List[str]:
        if self.name is None:
            raise RuntimeError("Name not set")
        return self.categories

    def set_name(
        self, name, name_format="{name}[T.{category}]"
    ) -> "_InteractableCategoricalVector":
        if self.name is None:
            # Make sure to only format the name once
            self.name = name
            self.categories = [
                name_format.format(name=name, category=cat) for cat in self.categories
            ]
        return self


def _interact(
    left: _InteractableVector, right: _InteractableVector, reverse=False, separator=":"
) -> _InteractableVector:
    """Interact two interactable vectors.

    Parameters
    ----------
    left : _InteractableVector
        The left vector.
    right : _InteractableVector
        The right vector.
    reverse : bool, optional
        Whether to reverse the order of the interaction, by default False
    separator : str, optional
        The separator to use between the names of the interacted vectors, by default ":"

    Returns
    -------
    _InteractableVector
        The interacted vector.
    """
    if isinstance(left, _InteractableDenseVector):
        if isinstance(right, _InteractableDenseVector):
            if not reverse:
                new_name = f"{left.name}{separator}{right.name}"
            else:
                new_name = f"{right.name}{separator}{left.name}"
            return _InteractableDenseVector(left.values * right.values, name=new_name)

        else:
            return _interact(right, left, reverse=True, separator=separator)

    if isinstance(left, _InteractableSparseVector):
        if isinstance(right, (_InteractableDenseVector, _InteractableSparseVector)):
            if not reverse:
                new_name = f"{left.name}{separator}{right.name}"
            else:
                new_name = f"{right.name}{separator}{left.name}"
            return _InteractableSparseVector(
                left.values.multiply(right.values),
                name=new_name,
            )

        else:
            return _interact(right, left, reverse=True, separator=separator)

    if isinstance(left, _InteractableCategoricalVector):
        if isinstance(right, (_InteractableDenseVector, _InteractableSparseVector)):
            if isinstance(right, _InteractableDenseVector):
                right_values = right.values
            else:
                right_values = right.values.todense()
            if not reverse:
                new_categories = [
                    f"{cat}{separator}{right.name}" for cat in left.categories
                ]
                new_name = f"{left.name}{separator}{right.name}"
            else:
                new_categories = [
                    f"{right.name}{separator}{cat}" for cat in left.categories
                ]
                new_name = f"{right.name}{separator}{left.name}"
            return _InteractableCategoricalVector(
                codes=left.codes,
                categories=new_categories,
                multipliers=left.multipliers * right_values,
                name=new_name,
            )

        elif isinstance(right, _InteractableCategoricalVector):
            return _interact_categoricals(left, right, separator=separator)

    raise TypeError(
        f"Cannot interact {type(left).__name__} with {type(right).__name__}"
    )


def _interact_categoricals(
    left: _InteractableCategoricalVector,
    right: _InteractableCategoricalVector,
    separator=":",
) -> _InteractableCategoricalVector:
    """Interact two categorical vectors.

    Parameters
    ----------
    left : _InteractableCategoricalVector
        The left categorical vector.
    right : _InteractableCategoricalVector
        The right categorical vector.
    separator : str, optional
        The separator to use between the names of the interacted vectors, by default ":"

    Returns
    -------
    _InteractableCategoricalVector
        The interacted categorical vector.
    """
    cardinality_left = len(left.categories)
    new_codes = right.codes * cardinality_left + left.codes

    na_mask = (left.codes == -1) | (right.codes == -1)
    drop_mask = (left.codes == -2) | (right.codes == -2)

    new_codes[drop_mask] = -2
    new_codes[na_mask] = -1

    new_categories = [
        f"{left_cat}{separator}{right_cat}"
        for right_cat, left_cat in itertools.product(right.categories, left.categories)
    ]

    return _InteractableCategoricalVector(
        codes=new_codes,
        categories=new_categories,
        multipliers=left.multipliers * right.multipliers,
        name=f"{left.name}{separator}{right.name}",
    )


def _C(
    data,
    *,
    spans_intercept: bool = True,
):
    """
    Mark data as categorical.

    A reduced-functionality version of the ``formulaic`` ``C()`` function. It does not
    support custom contrasts or the level argument, but it allows setting
    ``spans_intercept=False`` to avoid dropping categories.
    """

    def encoder(
        values,
        reduced_rank,
        drop_rows,
        encoder_state,
        model_spec,
    ):
        if drop_rows:
            values = values.drop(index=values.index[drop_rows])
        cat = pandas.Categorical(values._values)
        return _InteractableCategoricalVector.from_categorical(
            cat, reduced_rank=reduced_rank
        )

    return FactorValues(
        data,
        kind="categorical",
        spans_intercept=spans_intercept,
        encoder=encoder,
    )
