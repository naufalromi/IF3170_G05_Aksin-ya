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

    def approximate_kernel(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n: int = X.shape[0]
        m: int = min(n, 200)
        idx: np.ndarray = np.random.default_rng(42).choice(n, size=m, replace=False)
        X_m: np.ndarray = X[idx]
        eigvals, eigvecs = np.linalg.eigh(self.compute_kernel(X_m, X_m)) #K_mm = V * Lambda * V^T
        mask: np.ndarray = eigvals > 1e-8
        basis: np.ndarray = eigvecs[:, mask] / np.sqrt(eigvals[mask]) #K_mm^(-1/2)
        Phi: np.ndarray = self.compute_kernel(X, X_m) @ basis #phi(X) = K_nm * K_mm^(-1/2), matriks fitur di ruang kernel (K ≈ Phi * Phi^T)
        return Phi, basis, X_m

    def fit(self, X: np.ndarray, y: np.ndarray) -> svm:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes: np.ndarray = np.unique(y)
        y_bin: np.ndarray = np.where(y == self.classes[1], 1.0, -1.0)
        n: int = X.shape[0]
        q: np.ndarray = -np.ones(n, dtype=float)
        alpha_var = cp.Variable(n)

        if self.kernel == "linear":
            Phi: np.ndarray = X #phi(X) = X, pemetaan fitur ruang kernel (<phi(xi), phi(xj)> = K(xi, xj))
        else:
            Phi, basis, X_m = self.approximate_kernel(X) #phi(X) hasil aproksimasi Nystrom

        Z: np.ndarray = Phi * y_bin[:, None] #yi * phi(Xi)
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(Z.T @ alpha_var) + 0.5 * cp.sum_squares(alpha_var) + q @ alpha_var), #sum(aPa)-sum(a) alias sum(aiyiyjK(xi,xj)aj)-sum(a)
            [alpha_var >= 0.0, y_bin @ alpha_var == 0.0], #constraint
        )
        prob.solve(solver=cp.OSQP)

        alpha: np.ndarray = np.asarray(alpha_var.value, dtype=float)
        sv: np.ndarray = alpha > self.tol

        if self.kernel == "linear":
            self.coef: np.ndarray = alpha[sv] * y_bin[sv] #ai*yi
            self.support_vectors: np.ndarray = X[sv]
        else:
            self.coef = ((alpha[sv] * y_bin[sv]) @ Phi[sv]) @ basis.T #ai*yi*phi(Xi)*K_mm^(-1/2)^T
            self.support_vectors = X_m
        self.b: float = (
            float(y_bin[sv][0] * (1.0 - alpha[sv][0]) - self.coef @ self.compute_kernel(self.support_vectors, X[sv][:1]).ravel()) #yk(1-ak) - sum(ai*yi*K(xi,xk))
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
