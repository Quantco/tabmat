#pragma once

// Memory layout of a dense two-dimensional array, plus a uniform element
// accessor, so that each kernel is written once and instantiated for both
// layouts instead of being duplicated per layout.
enum class Order { C, F };

// Element (row, col) of an n_rows x n_cols array stored in the given order.
template <Order order, typename F>
inline F load_X(const F* X, Py_ssize_t row, Py_ssize_t col,
                Py_ssize_t n_rows, Py_ssize_t n_cols) {
    if constexpr (order == Order::C) {
        return X[row * n_cols + col];
    } else {
        return X[col * n_rows + row];
    }
}
