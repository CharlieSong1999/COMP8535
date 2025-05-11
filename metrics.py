import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors



# Assume you have:
# angles: shape (n_samples,) → scalar angle for each image
# embeddings: shape (n_samples, 2) → LLE/t-SNE/PCA embedding

# Compute pairwise distances
# gt_dists = squareform(pdist(angles[:, None], metric='euclidean'))  # ground-truth distance matrix
# embed_dists = squareform(pdist(embeddings, metric='euclidean'))    # embedded distance matrix

# # Flatten upper triangle for correlation
# gt_vec = gt_dists[np.triu_indices_from(gt_dists, k=1)]
# embed_vec = embed_dists[np.triu_indices_from(embed_dists, k=1)]

# # Correlation (Spearman or Pearson)
# spearman_corr, _ = spearmanr(gt_vec, embed_vec)
# pearson_corr, _ = pearsonr(gt_vec, embed_vec)

# print(f"Spearman: {spearman_corr:.3f}, Pearson: {pearson_corr:.3f}")



# # Original feature vectors: here use scalar angles
# X = angles[:, None]
# X_embed = embeddings

# # Trustworthiness (sklearn only supports this, not continuity)
# trust = trustworthiness(X, X_embed, n_neighbors=5)
# print(f"Trustworthiness: {trust:.3f}")


def continuity(X, X_embed, n_neighbors=5):
    nn_orig = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    nn_embed = NearestNeighbors(n_neighbors=n_neighbors).fit(X_embed)

    _, orig_neighbors = nn_orig.kneighbors(X)
    _, embed_neighbors = nn_embed.kneighbors(X_embed)

    n = X.shape[0]
    ranks = np.zeros(n)
    for i in range(n):
        orig_set = set(orig_neighbors[i])
        embed_set = set(embed_neighbors[i])
        missing = orig_set - embed_set
        ranks[i] = len(missing)
    
    continuity_score = 1 - np.sum(ranks) / (n * n_neighbors)
    return continuity_score

# cont = continuity(X, X_embed, n_neighbors=5)
# print(f"Continuity: {cont:.3f}")

class Metric_spearman_correlation:
    def __init__(self, data, embedding, n_neighbors=5):
        self.data = data
        self.embedding = embedding
        self.n_neighbors = n_neighbors
        self.gt_dists = squareform(pdist(data, metric='euclidean'))
        self.embed_dists = squareform(pdist(embedding, metric='euclidean'))
        self.gt_vec = self.gt_dists[np.triu_indices_from(self.gt_dists, k=1)]
        self.embed_vec = self.embed_dists[np.triu_indices_from(self.embed_dists, k=1)]

    def compute(self):
        return spearmanr(self.gt_vec, self.embed_vec)[0]
    
class Metric_pearson_correlation:
    def __init__(self, data, embedding, n_neighbors=5):
        self.data = data
        self.embedding = embedding
        self.n_neighbors = n_neighbors
        self.gt_dists = squareform(pdist(data, metric='euclidean'))
        self.embed_dists = squareform(pdist(embedding, metric='euclidean'))
        self.gt_vec = self.gt_dists[np.triu_indices_from(self.gt_dists, k=1)]
        self.embed_vec = self.embed_dists[np.triu_indices_from(self.embed_dists, k=1)]

    def compute(self):
        return pearsonr(self.gt_vec, self.embed_vec)[0]
    
class Metric_trustworthiness:
    def __init__(self, data, embedding, n_neighbors=5):
        self.data = data
        self.embedding = embedding
        self.n_neighbors = n_neighbors

    def compute(self):
        return trustworthiness(self.data, self.embedding, n_neighbors=self.n_neighbors)
    
class Metric_continuity:
    def __init__(self, data, embedding, n_neighbors=5):
        self.data = data
        self.embedding = embedding
        self.n_neighbors = n_neighbors

    def compute(self):
        return continuity(self.data, self.embedding, n_neighbors=self.n_neighbors)


class MetricsFactory:
    @staticmethod
    def create_metrics(metric_name, **kwargs):
        if metric_name == "trustworthiness":
            return Metric_trustworthiness(**kwargs)
        elif metric_name == "continuity":
            return Metric_continuity(**kwargs)
        elif metric_name == "spearman_correlation":
            return Metric_spearman_correlation(**kwargs)
        elif metric_name == "pearson_correlation":
            return Metric_pearson_correlation(**kwargs)
        else:
            raise ValueError(f"Metric {metric_name} is not supported.")