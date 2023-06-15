import functools
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
from .matrix_base import MatrixBase
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
        return DenseMatrix(series)

    @override
    def _encode_numerical(self, values, metadata, encoder_state, spec, drop_rows):
        if drop_rows:
            values = values.drop(index=values.index[drop_rows])
        if isinstance(values, pandas.Series):
            values = values.to_numpy()
        return DenseMatrix(values)

    @override
    def _encode_categorical(
        self, values, metadata, encoder_state, spec, drop_rows, reduced_rank=False
    ):
        # We do not do any encoding here as it is handled by tabmat
        if drop_rows:
            values = values.drop(index=values.index[drop_rows])
        return CategoricalMatrix(values._values, drop_first=reduced_rank)

    @override
    def _get_columns_for_term(self, factors, spec, scale=1):
        out = OrderedDict()

        names = [
            ":".join(reversed(product))
            for product in itertools.product(*reversed(factors))
        ]

        for i, reversed_product in enumerate(
            itertools.product(*(factor.items() for factor in reversed(factors)))
        ):
            # TODO: implement this
            out[names[i]] = functools.reduce(
                _interact_columns,
                (p[1] for p in reversed(reversed_product)),
            )
            if scale != 1:
                # TODO: do we need this? Maybe raise?
                out[names[i]] = scale * out[names[i]]
        return out

    @override
    def _combine_columns(self, cols, spec, drop_rows):
        # Special case no columns
        if not cols:
            values = numpy.empty((self.data.shape[0], 0))
            return SplitMatrix([DenseMatrix(values)])

        # Otherwise, concatenate columns into SplitMatrix
        return SplitMatrix([col[1] for col in cols])

    # Have to override _build_model_matrix, too, because of tabmat/glum's way
    # of handling intercepts and categorical variables.
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
            spec = spec.update(
                structure=[
                    EncodedTermStructure(
                        term,
                        [st.copy(without_values=True) for st in scoped_terms],
                        list(scoped_cols),
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


# There should be a better palce for this:
def _interact_columns(
    left: MatrixBase, right: MatrixBase, dense_threshold: float = 0.1
) -> MatrixBase:
    if isinstance(left, DenseMatrix) and isinstance(right, DenseMatrix):
        return left.multiply(right)

    if isinstance(left, SparseMatrix) and not isinstance(right, CategoricalMatrix):
        return left.multiply(right)

    if isinstance(right, SparseMatrix) and not isinstance(left, CategoricalMatrix):
        return right.multiply(left)

    if isinstance(left, CategoricalMatrix) and not isinstance(right, CategoricalMatrix):
        if len(right.shape):
            right = right.reshape(-1, 1)  # type: ignore
        return SparseMatrix(left.tocsr().multiply(right))
        # TODO: we could do better by making it dense above a threshold

    if isinstance(right, CategoricalMatrix) and not isinstance(left, CategoricalMatrix):
        if len(left.shape):
            left = left.reshape(-1, 1)  # type: ignore
        return SparseMatrix(right.tocsr().multiply(left))

    if isinstance(left, CategoricalMatrix) and isinstance(right, CategoricalMatrix):
        return _interact_categorical_categorical(left, right)

    # Should be unreachable
    raise RuntimeError(
        f"_interact_columns not implemented for {type(left)} and {type(right)}"
    )


def _interact_categorical_categorical(
    left: CategoricalMatrix, right: CategoricalMatrix
) -> CategoricalMatrix:
    card_right = len(right.cat.categories)

    new_codes = left.cat.codes * card_right + right.cat.codes

    if right.drop_first:
        new_codes[new_codes % card_right == 0] = 0
        new_codes -= new_codes // card_right
        left_shift = card_right - 1
        right_slice = slice(1, None)
    else:
        left_shift = card_right
        right_slice = slice(None)

    if left.drop_first:
        new_codes -= left_shift
        new_codes[new_codes < 0] = 0
        left_slice = slice(1, None)
    else:
        left_slice = slice(None)

    new_categories = [
        f"{left_cat}__{right_cat}"
        for left_cat, right_cat in itertools.product(
            left.cat.categories[left_slice], right.cat.categories[right_slice]
        )
    ]

    new_drop_first = left.drop_first or right.drop_first
    if new_drop_first:
        new_categories = ["__drop__"] + new_categories

    new_col = pandas.Categorical.from_codes(
        new_codes,
        new_categories,
        ordered=left.cat.ordered and right.cat.ordered,
    )

    return CategoricalMatrix(new_col, drop_first=new_drop_first)
