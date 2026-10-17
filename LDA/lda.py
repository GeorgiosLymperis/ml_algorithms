import numpy as np

def lda_fit_svd(X, y, classes):
    N, p = X.shape
    K = len(classes)

    # Class means
    mu = {k: X[y == k].mean(axis=0) for k in classes}

    # Within-class centered data
    X_centered = np.vstack([X[y == k] - mu[k] for k in classes])

    # SVD απευθείας πάνω στο centered data matrix
    _, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    d = s ** 2 / (N - K)  # eigenvalues του Sigma_pooled

    # Truncation για σταθερότητα (αγνόησε near-zero eigenvalues)
    tol = 1e-8 * d[0]
    valid = d > tol
    d, Vt = d[valid], Vt[valid]

    # Sphering transform
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    sphere = D_inv_sqrt @ Vt  # shape: (rank, p)

    discriminants = {}
    for k in classes:
        mu_star = sphere @ mu[k]
        log_prior = np.log((y == k).mean())
        discriminants[k] = {'mu_star': mu_star, 'log_prior': log_prior}

    return sphere, discriminants

def lda_predict(x, sphere, discriminants):
    x_star = sphere @ x
    scores = {
        k: -0.5 * np.sum((x_star - d['mu_star']) ** 2) + d['log_prior']
        for k, d in discriminants.items()
    }
    return max(scores, key=scores.get)