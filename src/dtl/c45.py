import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .algorithms import build_decision_tree

TARGET_COLUMN = "TARGET_CINTA"


class DecisionTreeC45:
    def __init__(self, prune: bool = True, confidence_factor: float = 0.25) -> None:
        self.prune: bool = prune
        self.confidence_factor: float = confidence_factor

    def _to_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        X = np.asarray(X)
        return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

    def fit(self, X, y) -> DecisionTreeC45:
        data = self._to_frame(X).copy()
        y = np.asarray(y)

        if len(data) == 0 or len(data) != len(y):
            raise ValueError("X dan y harus punya jumlah baris yang sama dan tidak kosong.")
        if TARGET_COLUMN in data.columns:
            raise ValueError(f"Nama kolom '{TARGET_COLUMN}' dipakai untuk kelas.")

        # kolom kelas harus kolom terakhir
        data[TARGET_COLUMN] = y

        self.classes_: np.ndarray = np.unique(y)
        self.feature_names_: list[str] = list(data.columns[:-1])

        # metadata adalah nilai-nilai yang mungkin untuk atribut kategorikal dan kelas
        metadata = {TARGET_COLUMN: list(self.classes_)}
        for column in self.feature_names_:
            if not is_numeric_dtype(data[column]):
                metadata[column] = list(data[column].dropna().unique())

        self.tree_ = build_decision_tree(
            metadata, data, prune=self.prune, confidence_factor=self.confidence_factor
        )
        self.tree_.classes = list(self.classes_)
        return self

    def _check_fitted(self, X) -> pd.DataFrame:
        if not hasattr(self, "tree_"):
            raise ValueError("Panggil fit sebelum predict.")
        data = self._to_frame(X)
        missing_columns = set(self.feature_names_) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Kolom tidak ditemukan: {sorted(missing_columns)}")
        return data

    def predict(self, X) -> np.ndarray:
        data = self._check_fitted(X)
        return np.array([self.tree_.predict(row) for _, row in data.iterrows()])

    def predict_proba(self, X) -> np.ndarray:
        data = self._check_fitted(X)
        proba = np.array([self.tree_.probabilities(row) for _, row in data.iterrows()], dtype=float)
        return proba / proba.sum(axis=1, keepdims=True)

    predict_probability = predict_proba
