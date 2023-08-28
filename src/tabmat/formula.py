import functools
import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy
import pandas
from formulaic import ModelMatrix, ModelSpec
from formulaic.errors import FactorEncodingError
from formulaic.materializers import FormulaMaterializer
from formulaic.materializers.base import EncodedTermStructure
from formulaic.materializers.types import FactorValues, NAAction, ScopedTerm
from formulaic.parser.types import Term
from formulaic.transforms import stateful_transform
from interface_meta import override
from scipy import sparse as sps

from .categorical_matrix import CategoricalMatrix
from .constructor_util import _split_sparse_and_dense_parts
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
            "categorical_format", "{name}[{category}]"
        )
        self.intercept_name = self.params.get("intercept_name", "Intercept")
        self.dtype = self.params.get("dtype", numpy.float64)
        self.sparse_threshold = self.params.get("sparse_threshold", 0.1)
        self.cat_threshold = self.params.get("cat_threshold", 4)
        self.add_column_for_intercept = self.params.get(
            "add_column_for_intercept", True
        )
        self.cat_missing_method = self.params.get("cat_missing_method", "fail")
        self.cat_missing_name = self.params.get("cat_missing_name", "(MISSING)")

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
            return _InteractableSparseVector(sps.csc_matrix(values[:, numpy.newaxis]))
        else:
            return _InteractableDenseVector(values)

    @override
    def _encode_categorical(
        self, values, metadata, encoder_state, spec, drop_rows, reduced_rank=False
    ):
        # We do not do any encoding here as it is handled by tabmat
        if drop_rows:
            values = values.drop(index=values.index[drop_rows])
        return encode_contrasts(
            values,
            reduced_rank=reduced_rank,
            missing_method=self.cat_missing_method,
            missing_name=self.cat_missing_name,
            _metadata=metadata,
            _state=encoder_state,
            _spec=spec,
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
                col[1].to_tabmat(
                    self.dtype,
                    self.sparse_threshold,
                    self.cat_threshold,
                )
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
                    if not self.add_column_for_intercept:
                        continue
                    scoped_cols[
                        self.intercept_name
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

    # Again, need a correction to handle categoricals properly
    @override
    def _enforce_structure(
        self,
        cols: List[Tuple[Term, List[ScopedTerm], Dict[str, Any]]],
        spec,
        drop_rows: set,
    ):
        assert len(cols) == len(spec.structure)
        for i, col_spec in enumerate(cols):
            scoped_cols = col_spec[2]
            target_cols = spec.structure[i][2].copy()

            # Correction for categorical variables:
            for name, col in scoped_cols.items():
                if isinstance(col, _InteractableCategoricalVector):
                    try:
                        _replace_sequence(target_cols, col.get_names(), name)
                    except ValueError:
                        raise FactorEncodingError(
                            f"Term `{col_spec[0]}` has generated columns that are "
                            "inconsistent with the specification: generated: "
                            f"{col.get_names()}, expecting: {target_cols}."
                        )

            if len(scoped_cols) > len(target_cols):
                raise FactorEncodingError(
                    f"Term `{col_spec[0]}` has generated too many columns compared to "
                    f"specification: generated {list(scoped_cols)}, expecting "
                    f"{target_cols}."
                )
            if len(scoped_cols) < len(target_cols):
                if len(scoped_cols) == 0:
                    col = self._encode_constant(0, None, None, spec, drop_rows)
                elif len(scoped_cols) == 1:
                    col = tuple(scoped_cols.values())[0]
                else:
                    raise FactorEncodingError(
                        f"Term `{col_spec[0]}` has generated insufficient columns "
                        "compared to specification: generated {list(scoped_cols)}, "
                        f"expecting {target_cols}."
                    )
                scoped_cols = {name: col for name in target_cols}
            elif set(scoped_cols) != set(target_cols):
                raise FactorEncodingError(
                    f"Term `{col_spec[0]}` has generated columns that are inconsistent "
                    "with specification: generated {list(scoped_cols)}, expecting "
                    f"{target_cols}."
                )

            yield col_spec[0], col_spec[1], {
                col: scoped_cols[col] for col in target_cols
            }


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
    ) -> Union[SparseMatrix, DenseMatrix]:
        if (self.values != 0).mean() > sparse_threshold:
            return DenseMatrix(self.values, column_names=[self.name])
        else:
            # Columns can become sparser, but not denser through interactions
            return SparseMatrix(
                sps.csc_matrix(self.values[:, numpy.newaxis]), column_names=[self.name]
            )

    def get_names(self) -> List[str]:
        if self.name is None:
            raise RuntimeError("Name not set")
        return [self.name]

    def set_name(self, name, name_format=None) -> "_InteractableDenseVector":
        self.name = name
        return self


class _InteractableSparseVector(_InteractableVector):
    def __init__(self, values: sps.csc_matrix, name: Optional[str] = None):
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
        return SparseMatrix(self.values, column_names=[self.name])

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
        cls,
        cat: pandas.Categorical,
        reduced_rank: bool,
        missing_method: str = "fail",
        missing_name: str = "(MISSING)",
        force_convert: bool = False,
    ) -> "_InteractableCategoricalVector":
        """Create an interactable categorical vector from a pandas categorical."""
        categories = list(cat.categories)
        codes = cat.codes.copy().astype(numpy.int64)

        if reduced_rank:
            codes[codes == 0] = -2
            codes[codes > 0] -= 1
            categories = categories[1:]

        if missing_method == "fail" and -1 in codes:
            raise ValueError(
                "Categorical data can't have missing values "
                "if [cat_]missing_method='fail'."
            )

        if missing_method == "convert" and (-1 in codes or force_convert):
            codes[codes == -1] = len(categories)
            categories.append(missing_name)

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
    ) -> Union[DenseMatrix, CategoricalMatrix, SplitMatrix]:
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

        categorical_part = CategoricalMatrix(
            cat,
            drop_first=drop_first,
            dtype=dtype,
            column_name=self.name,
            column_name_format="{category}",
            cat_missing_method="zero",  # missing values are already handled
        )

        if (self.codes == -2).all():
            # All values are dropped
            return DenseMatrix(numpy.empty((len(codes), 0), dtype=dtype))
        elif (self.multipliers == 1).all() and len(categories) >= cat_threshold:
            return categorical_part
        else:
            sparse_matrix = sps.csc_matrix(
                categorical_part.tocsr().multiply(self.multipliers[:, numpy.newaxis])
            )
            (
                dense_part,
                sparse_part,
                dense_idx,
                sparse_idx,
            ) = _split_sparse_and_dense_parts(
                sparse_matrix,
                sparse_threshold,
                column_names=categorical_part.column_names,
            )
            return SplitMatrix([dense_part, sparse_part], [dense_idx, sparse_idx])

    def get_names(self) -> List[str]:
        if self.name is None:
            raise RuntimeError("Name not set")
        return self.categories

    def set_name(
        self, name, name_format="{name}[{category}]"
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
            return _interact(right, left, reverse=not reverse, separator=separator)

    if isinstance(left, _InteractableSparseVector):
        if isinstance(right, (_InteractableDenseVector, _InteractableSparseVector)):
            if not reverse:
                new_name = f"{left.name}{separator}{right.name}"
            else:
                new_name = f"{right.name}{separator}{left.name}"
            return _InteractableSparseVector(
                left.values.multiply(right.values.reshape((-1, 1))),
                name=new_name,
            )

        else:
            return _interact(right, left, reverse=not reverse, separator=separator)

    if isinstance(left, _InteractableCategoricalVector):
        if isinstance(right, (_InteractableDenseVector, _InteractableSparseVector)):
            if isinstance(right, _InteractableDenseVector):
                right_values = right.values
            else:
                right_values = right.values.toarray().squeeze()
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
            if not reverse:
                return _interact_categoricals(left, right, separator=separator)
            else:
                return _interact_categoricals(right, left, separator=separator)

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

    new_codes[na_mask] = -1
    new_codes[drop_mask] = -2

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
    levels: Optional[Iterable[str]] = None,
    missing_method: str = "fail",
    missing_name: str = "(MISSING)",
    spans_intercept: bool = True,
):
    """
    Mark data as categorical.

    A reduced-functionality version of the ``formulaic`` ``C()`` function. It does not
    support custom contrasts or the level argument, but it allows setting
    ``spans_intercept=False`` to avoid dropping categories.
    """

    def encoder(
        values: Any,
        reduced_rank: bool,
        drop_rows: List[int],
        encoder_state: Dict[str, Any],
        model_spec: ModelSpec,
    ):
        if drop_rows:
            values = values.drop(index=values.index[drop_rows])
        return encode_contrasts(
            values,
            levels=levels,
            reduced_rank=reduced_rank,
            missing_method=missing_method,
            missing_name=missing_name,
            _state=encoder_state,
            _spec=model_spec,
        )

    return FactorValues(
        data,
        kind="categorical",
        spans_intercept=spans_intercept,
        encoder=encoder,
    )


@stateful_transform
def encode_contrasts(
    data,
    *,
    levels: Optional[Iterable[str]] = None,
    missing_method: str = "fail",
    missing_name: str = "(MISSING)",
    reduced_rank: bool = False,
    _state=None,
    _spec=None,
) -> FactorValues[_InteractableCategoricalVector]:
    """
    Encode a categorical dataset into one an _InteractableCategoricalVector

    Parameters
    ----------
        data: The categorical data array/series to be encoded.
        levels: The complete set of levels (categories) posited to be present in
            the data. This can also be used to reorder the levels as needed.
        reduced_rank: Whether to reduce the rank of output encoded columns in
            order to avoid spanning the intercept.
    """
    levels = levels if levels is not None else _state.get("categories")
    force_convert = _state.get("force_convert", False)
    cat = pandas.Categorical(data._values, categories=levels)
    _state["categories"] = cat.categories
    _state["force_convert"] = missing_method == "convert" and cat.isna().any()

    return _InteractableCategoricalVector.from_categorical(
        cat,
        reduced_rank=reduced_rank,
        missing_method=missing_method,
        missing_name=missing_name,
        force_convert=force_convert,
    )


def _replace_sequence(lst: List[str], sequence: List[str], replacement: "str") -> None:
    """Replace a sequence of elements in a list with a single element.

    Raises a ValueError if the sequence is not in the list in the correct order.
    Only checks for the first possible start of the sequence.

    Parameters
    ----------
    lst : List[str]
        The list to replace elements in.
    sequence : List[str]
        The sequence of elements to replace.
    replacement : str
        The element to replace the sequence with.
    """
    try:
        start = lst.index(sequence[0])
    except ValueError:
        start = 0  # Will handle this below

    for elem in sequence:
        if lst[start] != elem:
            raise ValueError("The sequence is not in the list")
        del lst[start]

    lst.insert(start, replacement)
