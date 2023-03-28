#include <vector>


<%def name="transpose_matvec(dropfirst)">
template <typename Int, typename F>
void _transpose_matvec_${dropfirst}(
    Int n_rows,
    Int* indices,
    F* other,
    F* res,
    Int res_size
) {
    #pragma omp parallel
    {
        std::vector<F> restemp(res_size, 0.0);
        #pragma omp for
        for (Py_ssize_t i = 0; i < n_rows; i++) {
            % if dropfirst == 'all_rows_drop_first':
                Py_ssize_t col_idx = indices[i] - 1;
                if (col_idx != -1) {
                    restemp[col_idx] += other[i];
                }
            % else:
                restemp[indices[i]] += other[i];
            % endif
        }
        for (Py_ssize_t i = 0; i < res_size; i++) {
            # pragma omp atomic
            res[i] += restemp[i];
        }
    }
}
</%def>


template <typename Int, typename F>
void _sandwich_cat_cat(
    F* d,
    const Int* i_indices,
    const Int* j_indices,
    Int* rows,
    Int len_rows,
    F* res,
    Int res_n_col,
    Int res_size,
    bool i_drop_first,
    bool j_drop_first
)
{
    #pragma omp parallel
    {
        std::vector<F> restemp(res_size, 0.0);
        # pragma omp for
        for (Py_ssize_t k_idx = 0; k_idx < len_rows; k_idx++) {
            Int k = rows[k_idx];
            Int i = i_indices[k] - i_drop_first;
            if (i == -1) {
                continue;
            }
            Int j = j_indices[k] - j_drop_first;
            if (j == -1) {
                continue;
            }
            restemp[(Py_ssize_t) i * res_n_col + j] += d[k];
        }
        for (Py_ssize_t i = 0; i < res_size; i++) {
            # pragma omp atomic
            res[i] += restemp[i];
        }
    }
}


<%def name="sandwich_cat_dense_tmpl(order)">
template <typename Int, typename F>
void _sandwich_cat_dense${order}(
    F* d,
    const Int* indices,
    Int* rows,
    Int len_rows,
    Int* j_cols,
    Int len_j_cols,
    F* res,
    Int res_size,
    F* mat_j,
    Int mat_j_nrow,
    Int mat_j_ncol
    )
{
    #pragma omp parallel
    {
        std::vector<F> restemp(res_size, 0.0);
        #pragma omp for
        for (Py_ssize_t k_idx = 0; k_idx < len_rows; k_idx++) {
            Py_ssize_t k = rows[k_idx];
            Py_ssize_t i = indices[k];
            // MAYBE TODO: explore whether the column restriction slows things down a
            // lot, particularly if not restricting the columns allows using SIMD
            // instructions
            // MAYBE TODO: explore whether swapping the loop order for F-ordered mat_j
            // is useful.
            for (Py_ssize_t j_idx = 0; j_idx < len_j_cols; j_idx++) {
                Py_ssize_t j = j_cols[j_idx];
                % if order == 'C':
                    restemp[i * len_j_cols + j_idx] += d[k] * mat_j[k * mat_j_ncol + j];
                % else:
                    restemp[i * len_j_cols + j_idx] += d[k] * mat_j[j * mat_j_nrow + k];
                % endif
            }
        }
        for (Py_ssize_t i = 0; i < res_size; i++) {
            #pragma omp atomic
            res[i] += restemp[i];
        }
    }
}
</%def>

${sandwich_cat_dense_tmpl('C')}
${sandwich_cat_dense_tmpl('F')}
${transpose_matvec('all_rows')}
${transpose_matvec('all_rows_drop_first')}
