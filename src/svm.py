from __future__ import annotations
import cvxpy as cp
import numpy as np


class svm:
    def __init__(
        self,
        kernel: str = "linear",
        degree: int = 3,
        gamma: float = 0.1,
        coef0: float = 0.0,
        tol: float = 1e-5,
    ) -> None:
        self.kernel: str = kernel
        self.degree: int = degree
        self.gamma: float = gamma
        self.coef0: float = coef0
        self.tol: float = tol

    def compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return X1 @ X2.T
        if self.kernel == "poly":
            return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        if self.kernel == "sigmoid":
            return np.tanh(self.gamma * (X1 @ X2.T) + self.coef0)
        sq: np.ndarray = np.sum(X1 ** 2, axis=1, keepdims=True) + np.sum(X2 ** 2, axis=1) - 2 * (X1 @ X2.T)
        return np.exp(-self.gamma * np.maximum(sq, 0.0))

    def fit(self, X: np.ndarray, y: np.ndarray) -> svm:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes: np.ndarray = np.unique(y)
        y_bin: np.ndarray = np.where(y == self.classes[1], 1.0, -1.0)
        n: int = X.shape[0]

        K: np.ndarray = self.compute_kernel(X, X)
        P: np.ndarray = np.outer(y_bin, y_bin) * K #yiyjK(Xi,Xj)
        P = 0.5 * (P + P.T) + np.eye(n) * 1e-8
        q: np.ndarray = -np.ones(n, dtype=float)

        alpha_var = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(alpha_var, P) + q @ alpha_var), #sum(aPa)-sum(a) alias sum(aiyiyjK(xi,xj)aj)-sum(a)
            [alpha_var >= 0.0, y_bin @ alpha_var == 0.0], #constraint
        )
        prob.solve()

        alpha: np.ndarray = np.asarray(alpha_var.value, dtype=float)
        sv: np.ndarray = alpha > self.tol

        self.coef: np.ndarray = alpha[sv] * y_bin[sv] #ai*yi
        self.support_vectors: np.ndarray = X[sv]
        self.b: float = float(y_bin[sv][0] - self.coef @ K[sv, np.argmax(sv)]) if np.any(sv) else 0.0
        return self

    def discriminant_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.coef @ self.compute_kernel(self.support_vectors, X) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.discriminant_function(X) >= 0, self.classes[1], self.classes[0])
