from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names) 

pca = PCA(n_components=2)
pca.fit(df)

PCA(copy=True, n_components=2, whiten=False)

T = pca.transform(df)

print(df.shape)

print(T.shape)
