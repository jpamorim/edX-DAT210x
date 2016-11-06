import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper
from sklearn.decomposition import PCA

matplotlib.style.use('ggplot')

scaleFeatures = False

# Load dataset and drop NaNs
df = pd.read_csv('Datasets/kidney_disease.csv', na_values='NaN')
df = df.dropna()

# Create some color coded labels
labels = ['red' if i=='ckd' else 'green' for i in df.classification]

# select only the columns bgr, wc, rc
df = df[['bgr','wc','rc']]

# Correct dtypes
df.wc = pd.to_numeric(df.wc, errors='coerce')
df.rc = pd.to_numeric(df.rc, errors='coerce')

# Print variance and describe data
print df.var()
print df.describe()

if scaleFeatures: df = helper.scaleFeatures(df)

# PCA 2 components
pca = PCA(n_components=2)
pca.fit(df)
PCA(copy=True, n_components=2, whiten=False)
T = pca.transform(df)

# Plot the transformed data as a scatter plot
ax = helper.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()