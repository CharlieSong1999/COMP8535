from sklearn.manifold import LocallyLinearEmbedding, Isomap, SpectralEmbedding, TSNE, MDS
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler


def get_scaler(scaler:str):
    if scaler == 'StandardScaler':
        return StandardScaler()
    elif scaler == 'None':
        return None
    else:
        raise ValueError(f"Scaler {scaler} is not supported. Please use 'StandardScaler'.")
    
def get_embedding(embedding:str, n_neighbors:int, n_components:int):
    if embedding == 'LLE':
        return LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
    elif embedding == 'Isomap':
        return Isomap(n_neighbors=n_neighbors, n_components=n_components)
    elif embedding == 'SpectralEmbedding':
        return SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    elif embedding == 'TSNE':
        return TSNE(n_components=n_components)
    elif embedding == 'MDS':
        return MDS(n_components=n_components)
    elif embedding == 'PCA':
        return PCA(n_components=n_components)
    elif embedding == 'KernelPCA':
        return KernelPCA(n_components=n_components, kernel='rbf', gamma=0.01)
    else:
        raise ValueError(f"Embedding {embedding} is not supported. Please use 'LocallyLinearEmbedding'.")


class DimensionalityReducer:

    def __init__(self, n_components, scaler:str, embedding:str, **kwargs):
        self.n_components = n_components
        self.scaler = get_scaler(scaler)
        self.embedding = get_embedding(embedding, n_components=n_components, **kwargs)

    def fit_transform(self, X):
        # Scale the data
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        # Apply LLE
        X_transformed = self.embedding.fit_transform(X_scaled)
        return X_transformed

    def inverse_transform(self, X_transformed):
        # Inverse transform the scaled data back to original space
        X_scaled = self.embedding.inverse_transform(X_transformed)
        # Inverse scale the data back to original space
        if self.scaler is not None:
            X_original = self.scaler.inverse_transform(X_scaled)
        else:
            X_original = X_scaled
        return X_original