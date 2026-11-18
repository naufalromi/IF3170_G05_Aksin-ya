import numpy as np

class NaiveBayes:
    def __init__(self, numeric_cols, categorical_cols, alpha=1.0, var_smoothing=1e-9):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.alpha = alpha
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        numeric_cols = self.numeric_cols
        categorical_cols = self.categorical_cols
        alpha = self.alpha
        var_smoothing = self.var_smoothing
        y = np.asarray(y)
        numeric_cols = list(numeric_cols)
        categorical_cols = list(categorical_cols)
        columns = numeric_cols + categorical_cols

        if y.ndim != 1 or len(X) == 0 or len(X) != len(y):
            raise ValueError("X and y must have the same nonzero number of rows; y must be one-dimensional.")
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Feature lists must not be empty or contain duplicate columns.")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive.")
        if not np.isfinite(var_smoothing) or var_smoothing < 0:
            raise ValueError("var_smoothing must be finite and nonnegative.")
        if X[columns].isna().any().any():
            raise ValueError("Handle missing values before calling fit.")

        classes, class_counts = np.unique(y, return_counts=True)
        numeric = X[numeric_cols].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Numeric features must contain finite numbers.")

        epsilon = np.maximum(var_smoothing * numeric.var(axis=0), 1e-12)
        means = []
        variances = []
        for label in classes:
            class_data = numeric[y == label]
            means.append(class_data.mean(axis=0))
            variances.append(class_data.var(axis=0) + epsilon)

        category_log_probs = {}
        unknown_log_probs = {}
        for col in categorical_cols:
            values = X[col].to_numpy()
            categories = X[col].unique()
            category_log_probs[col] = []
            unknown_log_probs[col] = []

            for label, count in zip(classes, class_counts):
                class_values = values[y == label]
                denominator = count + alpha * (len(categories) + 1)
                probs = {}
                for category in categories:
                    frequency = np.count_nonzero(class_values == category)
                    probs[category] = np.log((frequency + alpha) / denominator)
                category_log_probs[col].append(probs)
                unknown_log_probs[col].append(np.log(alpha / denominator))

        self.classes_ = classes
        self.numeric_cols_ = numeric_cols
        self.categorical_cols_ = categorical_cols
        self.class_log_prior_ = np.log(class_counts / len(y))
        self.numeric_mean_ = np.asarray(means)
        self.numeric_variance_ = np.asarray(variances)
        self.category_log_probs_ = category_log_probs
        self.unknown_log_probs_ = unknown_log_probs
        return self

    def _joint_log_scores(self, X):
        if not hasattr(self, "classes_"):
            raise ValueError("Call fit before prediction.")
        columns = self.numeric_cols_ + self.categorical_cols_
        if X[columns].isna().any().any():
            raise ValueError("Handle missing values before prediction.")
        numeric = X[self.numeric_cols_].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Numeric features must contain finite numbers.")

        scores = np.tile(self.class_log_prior_, (len(X), 1))
        for i in range(len(self.classes_)):
            mean = self.numeric_mean_[i]
            variance = self.numeric_variance_[i]
            gaussian = -0.5 * (np.log(2 * np.pi * variance) + (numeric - mean) ** 2 / variance)
            scores[:, i] += gaussian.sum(axis=1)

            for col in self.categorical_cols_:
                probs = self.category_log_probs_[col][i]
                unknown = self.unknown_log_probs_[col][i]
                scores[:, i] += np.array([probs.get(value, unknown) for value in X[col]], dtype=float)

        return scores

    def predict_log_proba(self, X):
        scores = self._joint_log_scores(X)
        scores -= scores.max(axis=1, keepdims=True)
        return scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        scores = self._joint_log_scores(X)
        return self.classes_[scores.argmax(axis=1)]
