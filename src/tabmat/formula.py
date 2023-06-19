import copy
import itertools
from collections import OrderedDict

import numpy
import pandas
from formulaic import ModelMatrix, ModelSpec
from formulaic.materializers import FormulaMaterializer
from formulaic.materializers.base import EncodedTermStructure
from formulaic.materializers.types import NAAction
from interface_meta import override

from .categorical_matrix import CategoricalMatrix
from .dense_matrix import DenseMatrix
from .sparse_matrix import SparseMatrix
from .split_matrix import SplitMatrix


class TabmatMaterializer(FormulaMaterializer):
    """Materializer for pandas input and tabmat output."""

    REGISTER_NAME = "tabmat"
    REGISTER_INPUTS = ("pandas.core.frame.DataFrame",)
    REGISTER_OUTPUTS = "tabmat"

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

        if isinstance(
            values, dict
        ):  # pragma: no cover; no formulaic transforms return dictionaries any more
            for key, vs in values.items():
                self._check_for_nulls(f"{name}[{key}]", vs, na_action, drop_rows)

        elif na_action is NAAction.RAISE:
            if isinstance(values, pandas.Series) and values.isnull().values.any():
                raise ValueError(f"`{name}` contains null values after evaluation.")

        elif na_action is NAAction.DROP:
            if isinstance(values, pandas.Series):
                drop_rows.update(numpy.flatnonzero(values.isnull().values))

        else:
            raise ValueError(
                f"Do not know how to interpret `na_action` = {repr(na_action)}."
            )  # pragma: no cover; this is currently impossible to reach

    @override
    def _encode_constant(self, value, metadata, encoder_state, spec, drop_rows):
        series = value * numpy.ones(self.nrows - len(drop_rows))
        return InteractableDenseMatrix(series)

    @override
    def _encode_numerical(self, values, metadata, encoder_state, spec, drop_rows):
        if drop_rows:
            values = values.drop(index=values.index[drop_rows])
        if isinstance(values, pandas.Series):
            values = values.to_numpy()
        return InteractableDenseMatrix(values)

    @override
    def _encode_categorical(
        self, values, metadata, encoder_state, spec, drop_rows, reduced_rank=False
    ):
        # We do not do any encoding here as it is handled by tabmat
        if drop_rows:
            values = values.drop(index=values.index[drop_rows])
        return InteractableCategoricalMatrix(values._values, drop_first=reduced_rank)

    @override
    def _combine_columns(self, cols, spec, drop_rows):
        # Special case no columns
        if not cols:
            values = numpy.empty((self.data.shape[0], 0))
            return SplitMatrix([DenseMatrix(values)])

        # Otherwise, concatenate columns into SplitMatrix
        return SplitMatrix([col[1].to_non_interactable() for col in cols])

    # Have to override this because of culumn names
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
                                *(
                                    mat.get_names(col)
                                    for col, mat in scoped_cols.items()
                                )
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


class InteractableDenseMatrix(DenseMatrix):
    def __mul__(self, other):
        if isinstance(other, (InteractableDenseMatrix, int, float)):
            return self.multiply(other)
        elif isinstance(
            other, (InteractableSparseMatrix, InteractableCategoricalMatrix)
        ):
            return other.__mul__(self)
        else:
            raise TypeError(f"Cannot multiply {type(self)} and {type(other)}")
        # Multiplication with sparse and categorical is handled by the other classes

    def __rmul__(self, other):
        return self.__mul__(other)

    def to_non_interactable(self):
        return DenseMatrix(self)

    def get_names(self, col):
        return [col]


class InteractableSparseMatrix(SparseMatrix):
    def __mul__(self, other):
        if isinstance(other, (InteractableDenseMatrix, InteractableSparseMatrix)):
            return self.multiply(other)
        elif isinstance(other, InteractableCategoricalMatrix):
            return other.__mul__(self)
        elif isinstance(other, (int, float)):
            return self.multiply(numpy.array(other))
        else:
            raise TypeError(f"Cannot multiply {type(self)} and {type(other)}")
        # Multiplication with categorical is handled by the categorical

    def __rmul__(self, other):
        return self.__mul__(other)

    def to_non_interactable(self):
        return SparseMatrix(self)

    def get_names(self, col):
        return [col]


class InteractableCategoricalMatrix(CategoricalMatrix):
    def __init__(self, *args, **kwargs):
        multipliers = kwargs.pop("multipliers", None)
        super().__init__(*args, **kwargs)
        if multipliers is None:
            self.multipliers = numpy.ones_like(self.cat, dtype=numpy.float_)
        else:
            self.multipliers = multipliers

    def __mul__(self, other):
        if isinstance(other, (InteractableDenseMatrix, float, int)):
            result = copy.copy(self)
            result.multipliers = result.multipliers * numpy.array(other)
            return result
        elif isinstance(other, InteractableSparseMatrix):
            result = copy.copy(self)
            result.multipliers = result.multipliers * other.todense()
            return result
        elif isinstance(other, InteractableCategoricalMatrix):
            return self._interact_categorical(other)
        else:
            raise TypeError(
                f"Can't multiply InteractableCategoricalMatrix with {type(other)}"
            )

    def __rmul__(self, other):
        if isinstance(other, InteractableCategoricalMatrix):
            other._interact_categorical(self)  # order matters
        else:
            return self.__mul__(other)

    def to_non_interactable(self):
        if numpy.all(self.multipliers == 1):
            return CategoricalMatrix(
                self.cat,
                drop_first=self.drop_first,
                dtype=self.dtype,
            )
        else:
            return SparseMatrix(
                self.tocsr().multiply(self.multipliers[:, numpy.newaxis])
            )

    def _interact_categorical(self, other):
        cardinality_right = len(other.cat.categories)

        new_codes = self.cat.codes * cardinality_right + other.cat.codes

        if other.drop_first:
            new_codes[new_codes % cardinality_right == 0] = 0
            new_codes -= new_codes // cardinality_right
            left_shift = cardinality_right - 1
            right_slice = slice(1, None)
        else:
            left_shift = cardinality_right
            right_slice = slice(None)

        if self.drop_first:
            new_codes -= left_shift
            new_codes[new_codes < 0] = 0
            left_slice = slice(1, None)
        else:
            left_slice = slice(None)

        new_categories = [
            f"{left_cat}:{right_cat}"
            for left_cat, right_cat in itertools.product(
                self.cat.categories[left_slice], other.cat.categories[right_slice]
            )
        ]

        new_drop_first = self.drop_first or other.drop_first
        if new_drop_first:
            new_categories = ["__drop__"] + new_categories

        cat = pandas.Categorical.from_codes(
            categories=new_categories,
            codes=new_codes,
            ordered=self.cat.ordered and other.cat.ordered,
        )

        return InteractableCategoricalMatrix(
            cat,
            multipliers=self.multipliers * other.multipliers,
            drop_first=new_drop_first,
        )

    def get_names(self, col):
        if self.drop_first:
            categories = self.cat.categories[1:]
        else:
            categories = self.cat.categories
        return [f"{col}[{cat}]" for cat in categories]
