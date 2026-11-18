import pickle

import numpy as np

def fit(X, y, numeric_cols, categorical_cols, alpha=1.0, var_smoothing=1e-9):
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

    return {
        "classes": classes,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "class_log_prior": np.log(class_counts / len(y)),
        "numeric_mean": np.asarray(means),
        "numeric_variance": np.asarray(variances),
        "category_log_probs": category_log_probs,
        "unknown_log_probs": unknown_log_probs,
    }

def _joint_log_scores(model, X):
    columns = model["numeric_cols"] + model["categorical_cols"]
    if X[columns].isna().any().any():
        raise ValueError("Handle missing values before prediction.")
    numeric = X[model["numeric_cols"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Numeric features must contain finite numbers.")

    scores = np.tile(model["class_log_prior"], (len(X), 1))
    for i in range(len(model["classes"])):
        mean = model["numeric_mean"][i]
        variance = model["numeric_variance"][i]
        gaussian = -0.5 * (np.log(2 * np.pi * variance) + (numeric - mean) ** 2 / variance)
        scores[:, i] += gaussian.sum(axis=1)

        for col in model["categorical_cols"]:
            probs = model["category_log_probs"][col][i]
            unknown = model["unknown_log_probs"][col][i]
            scores[:, i] += np.array([probs.get(value, unknown) for value in X[col]], dtype=float)
            
    return scores

def predict_log_proba(model, X):
    scores = _joint_log_scores(model, X)
    scores -= scores.max(axis=1, keepdims=True)
    return scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))

def predict_proba(model, X):
    return np.exp(predict_log_proba(model, X))

def predict(model, X):
    scores = _joint_log_scores(model, X)
    return model["classes"][scores.argmax(axis=1)]

def save_model(model, path):
    with open(path, "wb") as file:
        pickle.dump(model, file)

def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)
