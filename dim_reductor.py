from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler


def get_scaler(scaler:str):
    if scaler == 'StandardScaler':
        return StandardScaler()
    elif scaler == 'None':
        return None
    else:
        raise ValueError(f"Scaler {scaler} is not supported. Please use 'StandardScaler'.")
    
def get_embedding(embedding:str, n_neighbors:int, n_components:int):
    if embedding == 'LocallyLinearEmbedding':
        return LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)
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