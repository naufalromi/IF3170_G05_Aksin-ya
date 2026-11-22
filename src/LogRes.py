import numpy as np

class LogRes:
    def __init__(
        self,
        learn_rate: float = 0.0,
        iterations: int = 0,
        penalty: {'l1', 'l2', 'elastic', None} = None,
        C: float = 0.0,
        l1_ratio: float = 0.0,
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
        self.iter_result: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogRes:
        return NotImplemented

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        return NotImplemented

    def predict(self, X: np.ndarray) -> np.ndarray:
        return NotImplemented

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return NotImplemented

    def _initialize_weights(self, n_features: int) -> np.ndarray:
        if self.random_seed is not None:
            rng = np.random.default_rng(seed = self.random_seed)
            return rng.random(n_features)
        return np.zeros(n_features)

    def _shuffle_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed = self.random_seed)
        idx = rng.permutation(X.shape[0])
        return X[idx], y[idx]

    def _sigmoid(self, z: float) -> float:
        return 1 / (1 + np.exp(-z))

    def _penalty_term(self, weights: np.ndarray) -> float:
        if self.penalty == 'l1':
            return (1 / self.C) * np.sum(np.abs(weights))
        if self.penalty == 'l2':
            return (1 / self.C) * np.sum(weights ** 2)
        if self.penalty == 'elastic':
            return (1 / self.C) * (self.l1_ratio * np.sum(np.abs(weights)) + (1 - self.l1_ratio) * np.sum(weights ** 2))
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
        predictions = predict_prob(X)
        loglike = np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        loglike -= self._penalty_term(self.weights)
        return loglike

    def _compute_sample_gradient(self, x_i: np.ndarray, y_i: float) -> tuple[np.ndarray, float]:
        prediction = self._sigmoid(np.dot(x_i, self.weights) + self.bias)
        bias = y_i - prediction
        gradient_weights = bias * x_i
        return gradient_weights, bias