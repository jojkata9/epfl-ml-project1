"""Methods required for Project 1. Each returns (w, loss)."""

import numpy as np


def _mse(y, tx, w):
    error = y - tx @ w
    return np.mean(error**2) / 2


def _sigmoid(z):
    # This form also works for very large scores.
    return np.exp(-np.logaddexp(0, -z))


def _logistic_loss(y, tx, w):
    scores = tx @ w
    return np.mean(np.logaddexp(0, scores) - y * scores)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with gradient descent."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        error = tx @ w - y
        gradient = tx.T @ error / len(y)
        w -= gamma * gradient
    return w, _mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression, sampling one row per update."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        i = np.random.randint(len(y))
        error = tx[i] @ w - y[i]
        w -= gamma * tx[i] * error
    return w, _mse(y, tx, w)


def least_squares(y, tx):
    """Least squares using the normal equations."""
    a = tx.T @ tx
    b = tx.T @ y
    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(a) @ b
    return w, _mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Fit ridge regression; return the MSE without the penalty."""
    a = tx.T @ tx + 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    b = tx.T @ y
    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(a) @ b
    return w, _mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression with gradient descent. Labels are 0 or 1."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        predictions = _sigmoid(tx @ w)
        gradient = tx.T @ (predictions - y) / len(y)
        w -= gamma * gradient
    return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic regression with the penalty lambda_ * sum(w**2)."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        predictions = _sigmoid(tx @ w)
        gradient = tx.T @ (predictions - y) / len(y)
        gradient += 2 * lambda_ * w
        w -= gamma * gradient
    return w, _logistic_loss(y, tx, w)
