from sklearn import manifold
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper

matplotlib.style.use('ggplot')

# Load dataset and drop NaNs
df = pd.read_csv('Datasets/kidney_disease.csv', na_values='NaN')
df = df.dropna()

# Create some color coded labels
labels = ['red' if i=='ckd' else 'green' for i in df.classification]

# drop all nominal features
df = df.drop(df[['id', 'classification']], 1)

# Correct dtypes
df.pcv = pd.to_numeric(df.pcv, errors='coerce')
df.wc = pd.to_numeric(df.wc, errors='coerce')
df.rc = pd.to_numeric(df.rc, errors='coerce')
df = pd.get_dummies(df, columns=['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'])

df = helper.scaleFeatures(df)

iso = manifold.Isomap(n_neighbors=16, n_components=2)
iso.fit(df)
manifold.Isomap(eigen_solver='auto', max_iter=None, n_components=2, n_neighbors=16, 
       neighbors_algorithm='auto', path_method='auto', tol=0)
T = iso.transform(df)

# Plot the transformed data as a scatter plot
ax = helper.drawVectors(T, pca.components_, df.columns.values, plt, False)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()