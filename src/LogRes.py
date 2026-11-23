from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

class LogRes:
    def __init__(
        self,
        learn_rate: float = 0.01,
        iterations: int = 100,
        penalty: {'l1', 'l2', 'elastic', None} = None,
        C: float = 1.0,
        l1_ratio: float = 0.01,
        tolerance: float = 1e-4,
        fit_intercept: bool = True,
        shuffle: bool = True,
        verbose: bool = True,
        random_seed: int | None = None
    ):
        self.learn_rate = learn_rate
        self.iterations = iterations
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_seed = random_seed
        # set later
        self.weights: np.ndarray | None = None
        self.bias: float | None = None
        self.classes: np.ndarray | None = None
        self.log_likelihood_records: list[float] = []
        self.param_history: list[tuple[float, float]] = []
        self.iter_result: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogRes:
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("This implementation only supports binary classification.")
        y_encoded = (y == self.classes[1]).astype(float)
        self.weights = self._initialize_weights(n_features)
        self.bias = 0.0
        self.log_likelihood_records = []
        self.param_history = []
        self.iter_result = 0
        self._rng = np.random.default_rng(seed = self.random_seed)
        for i in range(self.iterations):
            if self.shuffle:
                X, y_encoded = self._shuffle_data(X, y_encoded)
            for idx in range(n_samples):
                x_i = X[idx]
                y_i = y_encoded[idx]
                grad_weights, grad_bias = self._compute_sample_gradient(x_i, y_i)
                penalty_grad = self._penalty_gradient(self.weights) / n_samples
                self.weights += self.learn_rate * (grad_weights - penalty_grad)
                if self.fit_intercept:
                    self.bias += self.learn_rate * grad_bias
            log_likelihood = self._compute_log_likelihood(X, y_encoded)
            self.log_likelihood_records.append(log_likelihood)
            self.param_history.append((self.bias, self.weights[0]))
            if self.verbose:
                print(f"Iteration {i + 1}/{self.iterations}, Log-Likelihood: {log_likelihood:.4f}")
            self.iter_result = i + 1
            if i > 0 and np.abs(log_likelihood - self.log_likelihood_records[-2]) < self.tolerance:
                if self.verbose:
                    print(f"Convergence reached at iteration {i + 1}.")
                break
        return self

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet.")
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_prob(X)
        result = (probs > 0.5).astype(int)
        return self.classes[result.astype(int)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def _initialize_weights(self, n_features: int) -> np.ndarray:
        if self.random_seed is not None:
            rng = np.random.default_rng(seed = self.random_seed)
            return rng.random(n_features) * 0.01
        return np.zeros(n_features)

    def _shuffle_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = self._rng.permutation(X.shape[0])
        return X[idx], y[idx]

    def _sigmoid(self, z: float) -> float:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _penalty_term(self, weights: np.ndarray) -> float:
        if self.penalty == 'l1':
            return (1 / self.C) * np.sum(np.abs(weights))
        if self.penalty == 'l2':
            return (1 / self.C) * 0.5 * np.sum(weights ** 2)
        if self.penalty == 'elastic':
            return (1 / self.C) * (self.l1_ratio * np.sum(np.abs(weights)) + (1 - self.l1_ratio) * 0.5 * np.sum(weights ** 2))
        return 0.0

    def _penalty_gradient(self, weights: np.ndarray) -> np.ndarray:
        if self.penalty == 'l1':
            return (1 / self.C) * np.sign(weights)
        if self.penalty == 'l2':
            return (1 / self.C) * weights
        if self.penalty == 'elastic':
            return (1 / self.C) * (self.l1_ratio * np.sign(weights) + (1 - self.l1_ratio) * weights)
        return np.zeros_like(weights)

    def _compute_log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = np.clip(self.predict_prob(X), 1e-12, 1 - 1e-12)
        loglike = np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        loglike -= self._penalty_term(self.weights)
        return loglike

    def _compute_sample_gradient(self, x_i: np.ndarray, y_i: float) -> tuple[np.ndarray, float]:
        prediction = self._sigmoid(np.dot(x_i, self.weights) + self.bias)
        bias = y_i - prediction
        gradient_weights = bias * x_i
        return gradient_weights, bias