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
        random_seed: int | None = None,
        record_every: int | None = None
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
        self.record_every = record_every
        # set later
        self.weights: np.ndarray | None = None
        self.bias: float | None = None
        self.classes: np.ndarray | None = None
        self.log_likelihood_records: list[float] = []
        self.bias_history: list[float] = []
        self.weight_history: list[np.ndarray] = []
        self.history_epoch: list[int] = []
        self.iter_result: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogRes:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("This implementation only supports binary classification.")
        y_encoded = (y == self.classes[1]).astype(float)
        self.weights = self._initialize_weights(n_features)
        self.bias = 0.0
        self.log_likelihood_records = []
        self.bias_history = []
        self.weight_history = []
        self.history_epoch = []
        self.iter_result = 0
        self._rng = np.random.default_rng(seed = self.random_seed)
        self._record_params(0)
        step = 0
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
                step += 1
                if self.record_every and step % self.record_every == 0 and idx != n_samples - 1:
                    self._record_params(i)
            self._record_params(i)
            log_likelihood = self._compute_log_likelihood(X, y_encoded)
            self.log_likelihood_records.append(log_likelihood)
            if self.verbose:
                print(f"Iteration {i + 1}/{self.iterations}, Log-Likelihood: {log_likelihood:.4f}")
            self.iter_result = i + 1
            if i > 0 and np.abs(log_likelihood - self.log_likelihood_records[-2]) < self.tolerance:
                if self.verbose:
                    print(f"Convergence reached at iteration {i + 1}.")
                break
        self._n_train = n_samples
        return self

    def loss_slice(self, X: np.ndarray, y: np.ndarray, theta0_grid: np.ndarray, theta1_grid: np.ndarray, feature: int) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        y_encoded = (np.asarray(y) == self.classes[1]).astype(float)
        n = X.shape[0]
        theta0_grid = np.asarray(theta0_grid, dtype=float)
        x_f = X[:, feature]
        c = np.dot(X, self.weights) - self.weights[feature] * x_f
        losses = np.empty((len(theta1_grid), len(theta0_grid)))
        w = self.weights.copy()
        for j, t1 in enumerate(theta1_grid):
            z = c[None, :] + theta0_grid[:, None] + t1 * x_f[None, :]
            nll = np.mean(np.logaddexp(0.0, z) - y_encoded[None, :] * z, axis=1)
            w[feature] = t1
            losses[j] = nll + self._penalty_term(w) / n
        return losses

    def animate_training(self,
        X: np.ndarray,
        y: np.ndarray,
        feature: int,
        feature_name: str | None = None,
        grid_size: int = 80,
        margin: float = 0.3,
        max_frames: int = 200,
        frame_spacing: str = "log",
        interval: int = 40,
        levels: int = 30,
        n_sub: int | None = None,
        figsize: tuple[float, float] = (13, 5)
    ) -> FuncAnimation:
        if not self.weight_history:
            raise ValueError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if n_sub is not None and n_sub < X.shape[0]:
            idx = np.random.default_rng(self.random_seed).choice(X.shape[0], n_sub, replace=False)
            X, y = X[idx], y[idx]
        name = feature_name or f"feature {feature}"

        trail = np.column_stack([self.bias_history, np.asarray(self.weight_history)[:, feature]])
        epochs = np.asarray(self.history_epoch)
        n_points = len(trail)
        if frame_spacing == "log":
            frames = np.geomspace(1, n_points, min(max_frames, n_points)).round().astype(int) - 1
        else:
            frames = np.linspace(0, n_points - 1, min(max_frames, n_points)).round().astype(int)
        frames = np.unique(np.concatenate([[0], frames]))

        lo, hi = trail.min(axis=0), trail.max(axis=0)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        lo, hi = lo - margin * span, hi + margin * span
        theta0 = np.linspace(lo[0], hi[0], grid_size)
        theta1 = np.linspace(lo[1], hi[1], grid_size)
        losses = self.loss_slice(X, y, theta0, theta1, feature)
        B, W = np.meshgrid(theta0, theta1)

        fig, (ax, ax_loss) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.3, 1]})

        contour_levels = np.geomspace(losses.min(), losses.max(), levels)
        filled = ax.contourf(B, W, losses, levels=contour_levels, cmap=cm.viridis)
        ax.contour(B, W, losses, levels=contour_levels, colors="white", linewidths=0.4, alpha=0.5)
        fig.colorbar(filled, ax=ax, label="mean penalized loss", format="%.3f")
        ax.plot(self.bias, self.weights[feature], marker="*", markersize=16, color="red",
                linestyle="none", zorder=5, label="final weights (slice point)")
        (path_line,) = ax.plot([], [], color="orange", linewidth=1.5, label="SGD trail")
        (head,) = ax.plot([], [], marker="o", markersize=7, color="orange", markeredgecolor="black")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel(r"$\theta_0$ (bias)")
        ax.set_ylabel(rf"$\theta_1$ (w: {name})")
        ax.legend(loc="upper right", fontsize=8)
        title = ax.set_title("")

        epoch_loss = -np.asarray(self.log_likelihood_records) / self._n_train
        ax_loss.plot(np.arange(1, len(epoch_loss) + 1), epoch_loss, color="tab:blue")
        progress = ax_loss.axvline(0, color="orange", linestyle="--")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("mean penalized loss (full model)")
        ax_loss.set_title("Training loss")
        fig.tight_layout()

        def update(frame: int):
            k = frames[frame]
            path_line.set_data(trail[:k + 1, 0], trail[:k + 1, 1])
            head.set_data([trail[k, 0]], [trail[k, 1]])
            epoch = 0 if k == 0 else epochs[k] + 1
            progress.set_xdata([epoch, epoch])
            title.set_text(r"Loss slice over ($\theta_0$, $\theta_1$), epoch " + str(epoch))
            return path_line, head, progress, title

        anim = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
        plt.close(fig)
        return anim

    def _record_params(self, epoch: int) -> None:
        self.bias_history.append(self.bias)
        self.weight_history.append(self.weights.copy())
        self.history_epoch.append(epoch)

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