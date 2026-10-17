import numpy as np
from scipy.linalg import solve_triangular

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def loss(self, y, y_pred):
        return 0.5 * np.mean((y - y_pred) ** 2)

    def fit(self, X, y, tol=1e-4, max_iter=1000, lr=1e-3, verbose=0):
        beta = np.zeros(X.shape[1] + 1)
        X = np.column_stack((np.ones(X.shape[0]), X))
        n = X.shape[0]
        for i in range(max_iter):
            gradient = -2 * np.dot(X.T, y - np.dot(X, beta))/n # [features + 1]
            beta = beta - lr * gradient
            if np.linalg.norm(gradient) < tol:
                break

            if verbose and i%100 == 0:
                print(f"{i} Iteration | Loss: {self.loss(y, np.dot(X, beta))}")

        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
    
    def ols(self, X, y):
        X = np.column_stack((np.ones(X.shape[0]), X))
        gram = X.T @ X
        L = np.linalg.cholesky(gram)
        z = solve_triangular(L, X.T @ y, lower=True)
        beta = solve_triangular(L.T, z, lower=False)
        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

        return self
    
class RidgeRegression:
    def __init__(self, lambda_: float = 1.0):
        self.coef_ = None
        self.intercept_ = None
        self.lambda_ = lambda_

    def loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def fit(self, X, y, tol=1e-4, max_iter=1000, lr=1e-3, verbose=0):
        beta = np.zeros(X.shape[1] + 1)
        X = np.column_stack((np.ones(X.shape[0]), X))
        n = X.shape[0]
        for i in range(max_iter):
            gradient = -2 * np.dot(X.T, y - np.dot(X, beta))/n # [features + 1]
            gradient[1:] += 2 * self.lambda_ * beta[1:] / n  # L2 regularization
            beta = beta - lr * gradient
            if np.linalg.norm(gradient) < tol:
                break

            if verbose and i%100 == 0:
                print(f"{i} Iteration | Loss: {self.loss(y, np.dot(X, beta))}")

        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
    
    def ols(self, X, y):
        X = np.column_stack((np.ones(X.shape[0]), X))
        penalty = self.lambda_ * np.eye(X.shape[1])
        penalty[0, 0] = 0  # do not regularize the intercept
        beta = np.linalg.inv(X.T @ X + penalty) @ X.T @ y
        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

        return self


class LassoRegression:
    def __init__(self, lambda_: float = 1.0):
        self.coef_ = None
        self.intercept_ = None
        self.lambda_ = lambda_

    def loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def fit(self, X, y, tol=1e-4, max_iter=1000, lr=1e-3, verbose=0):
        beta = np.zeros(X.shape[1] + 1)
        X = np.column_stack((np.ones(X.shape[0]), X))
        n = X.shape[0]
        for i in range(max_iter):
            gradient = -2 * np.dot(X.T, y - np.dot(X, beta))/n # [features + 1]
            beta_temp = beta - lr * gradient
            threshold = lr * self.lambda_ / n
            beta[1:] = np.sign(beta_temp[1:]) * np.maximum(np.abs(beta_temp[1:]) - threshold, 0)
            beta[0] = beta_temp[0]
            if np.linalg.norm(gradient) < tol:
                break

            if verbose and i%100 == 0:
                print(f"{i} Iteration | Loss: {self.loss(y, np.dot(X, beta))}")

        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_
    

class ElasticNet:
    def __init__(self, l1: float = 1.0, l2: float = 1.0):
        self.coef_ = None
        self.intercept_ = None
        self.l1 = l1
        self.l2 = l2

    def loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)
    
    def fit(self, X, y, tol=1e-4, max_iter=1000, lr=1e-3, verbose=0):
        beta = np.zeros(X.shape[1] + 1)
        X = np.column_stack((np.ones(X.shape[0]), X))
        n = X.shape[0]
        for i in range(max_iter):
            gradient = -2 * np.dot(X.T, y - np.dot(X, beta))/n # [features + 1]
            gradient[1:] += 2 * self.l2 * beta[1:] / n  # L2 regularization
            beta_temp = beta - lr * gradient
            threshold = lr * self.l1 / n
            beta[1:] = np.sign(beta_temp[1:]) * np.maximum(np.abs(beta_temp[1:]) - threshold, 0)
            beta[0] = beta_temp[0]
            if np.linalg.norm(gradient) < tol:
                break

            if verbose and i%100 == 0:
                print(f"{i} Iteration | Loss: {self.loss(y, np.dot(X, beta))}")

        self.coef_ = beta[1:]
        self.intercept_ = beta[0]

        return self

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

class LARS:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def fit(self, X, y, max_iter=None):
        # X assumed centered and standarized
        # y assumed centered
        n, p = X.shape
        max_iter = max_iter or p

        beta = np.zeros(p)
        mu = np.zeros(n)  # current fitted vector

        active = []  # active predictors
        signs = np.ones(p)
        coef_path = [beta.copy()]

        for step in range(max_iter):
            residual = y - mu
            corr = X.T @ residual
            corr_masked = corr.copy()
            corr_masked[active] = 0.0
            best_idx = np.argmax(np.abs(corr_masked))
            max_corr = corr[best_idx]


            if abs(max_corr) < 1e-4:
                break

            for i in range(p):
                if i not in active and np.isclose(abs(corr[i]), max_corr):
                    active.append(i)
                    signs[i] = np.sign(corr[i])

            C = abs(max_corr)

            X_A = X[:, active] * signs[active]
            G_A = X_A.T @ X_A
            ones = np.ones(len(active))
            G_A_inv_ones = np.linalg.solve(G_A, ones)
            A_A = 1 / np.sqrt(ones.T @ G_A_inv_ones)
            w_A = A_A * G_A_inv_ones
            u_A = X_A @ w_A

            a = X.T @ u_A
            candidates = []
            for j in range(p):
                if j not in active:
                    denom1 = A_A - a[j]
                    denom2 = A_A + a[j]
                    if abs(denom1) > 1e-12:
                        g1 = (C - corr[j]) / denom1
                        if g1 > 0:
                            candidates.append(g1)
                    if abs(denom2) > 1e-12:
                        g2 = (C + corr[j]) / denom2
                        if g2 > 0:
                            candidates.append(g2)

            if len(active) == p:
                gamma = C / A_A
            else:
                gamma = min(candidates)

            mu += gamma * u_A
            for idx, j in enumerate(active):
                beta[j] += gamma * w_A[idx] * signs[j]  # undo the sign flip applied to X_A
            coef_path.append(beta.copy())

        self.coef_ = beta
        self.coef_path_ = coef_path

        return self                    

    def predict(self, X):
        return np.dot(X, self.coef_)