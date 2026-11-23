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
        K: np.ndarray = X1 @ X2.T
        if self.kernel == "linear":
            return K
        if self.kernel == "poly":
            K *= self.gamma
            K += self.coef0
            K **= self.degree
            return K
        if self.kernel == "sigmoid":
            K *= self.gamma
            K += self.coef0
            np.tanh(K, out=K)
            return K
        K *= -2.0
        K += np.sum(X1 ** 2, axis=1, keepdims=True)
        K += np.sum(X2 ** 2, axis=1)
        np.maximum(K, 0.0, out=K)
        K *= -self.gamma
        np.exp(K, out=K)
        return K

    def fit(self, X: np.ndarray, y: np.ndarray) -> svm:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes: np.ndarray = np.unique(y)
        y_bin: np.ndarray = np.where(y == self.classes[1], 1.0, -1.0)
        n: int = X.shape[0]
        q: np.ndarray = -np.ones(n, dtype=float)
        alpha_var = cp.Variable(n)

        if self.kernel == "linear":
            Z: np.ndarray = X * y_bin[:, None]
            w_var = cp.Variable(X.shape[1])
            prob = cp.Problem(
                cp.Minimize(0.5 * cp.sum_squares(w_var) + 0.5e-8 * cp.sum_squares(alpha_var) + q @ alpha_var),
                [w_var == Z.T @ alpha_var, alpha_var >= 0.0, y_bin @ alpha_var == 0.0], #constraint
            )
        else:
            P: np.ndarray = self.compute_kernel(X, X)
            P *= y_bin[:, None]
            P *= y_bin[None, :] #yiyjK(Xi,Xj)
            P.flat[:: n + 1] += 1e-8
            prob = cp.Problem(
                cp.Minimize(0.5 * cp.quad_form(alpha_var, P, assume_PSD=True) + q @ alpha_var), #sum(aPa)-sum(a) alias sum(aiyiyjK(xi,xj)aj)-sum(a)
                [alpha_var >= 0.0, y_bin @ alpha_var == 0.0], #constraint
            )
        prob.solve()

        alpha: np.ndarray = np.asarray(alpha_var.value, dtype=float)
        sv: np.ndarray = alpha > self.tol

        self.coef: np.ndarray = alpha[sv] * y_bin[sv] #ai*yi
        self.support_vectors: np.ndarray = X[sv]
        self.b: float = (
            float(y_bin[sv][0] - self.coef @ self.compute_kernel(self.support_vectors, self.support_vectors[:1]).ravel())
            if np.any(sv)
            else 0.0
        )
        return self

    def discriminant_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.kernel == "linear":
            return X @ (self.coef @ self.support_vectors) + self.b
        return self.coef @ self.compute_kernel(self.support_vectors, X) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.discriminant_function(X) >= 0, self.classes[1], self.classes[0])
