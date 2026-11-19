import numpy as np


class KNN:
    def __init__(self, k: int = 3) -> None:
        self.k: int = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train: np.ndarray = np.asarray(X, dtype=float)
        self.y_train: np.ndarray = np.asarray(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self.predict_one(x) for x in X])

    def euclidean_distance(self, x: np.ndarray, x_train: np.ndarray) -> float:
        return float(np.sqrt(np.sum((x - x_train) ** 2)))

    def find_neighbors(self, x: np.ndarray) -> np.ndarray:
        distances = np.zeros(len(self.X_train), dtype=float)
        for i, x_train in enumerate(self.X_train):
            distances[i] = self.euclidean_distance(x, x_train)

        neighbor_indices = np.argsort(distances)[:self.k]
        return self.y_train[neighbor_indices]

    def vote(self, neighbor_labels: np.ndarray) -> int:
        counts = {}
        for label in neighbor_labels:
            counts[label] = counts.get(label, 0) + 1
        return max(counts, key=counts.get)

    def predict_one(self, x: np.ndarray) -> int:
        neighbor_labels = self.find_neighbors(x)
        return self.vote(neighbor_labels)
