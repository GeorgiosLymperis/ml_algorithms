import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None  # shape (n,), -1 for noise
        self.core_sample_indices_ = None 

    def _pairwise_distances(self, X: np.array) -> np.array:
        norms = np.sum(X**2, axis=1)
        dists = norms[:, None] + norms[None, :] - 2 * X @ X.T
        return np.sqrt(np.clip(dists, a_min=0, a_max=None))

    def fit(self, X):
        n = X.shape[0]

        dist = self._pairwise_distances(X)
        neighbors = [np.where(dist[i] <= self.eps)[0] for i in range(n)]

        is_core = np.array([len(neighbors[i]) >= self.min_samples for i in range(n)])

        labels = np.full(n, -1)  # all is noise initially
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            if not is_core[i]:
                # leave as noise for now; may still become a border point later
                continue

            labels[i] = cluster_id
            seeds  = list(neighbors[i])

            while seeds:
                q = seeds.pop()
                if not visited[q]:
                    visited[q] = True
                    if is_core[q]:
                        seeds.extend(neighbors[q])  # core point expanded

                if labels[q] == -1:
                    labels[q] = cluster_id  # claim border point

            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.where(is_core)[0]
        return self
