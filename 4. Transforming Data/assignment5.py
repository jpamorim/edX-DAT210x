import pandas as pd
import glob
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import misc
from sklearn import manifold
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

samples = []
colors = []

# Load all images in the datesets folder to a list
for filename in glob.glob('Datasets/ALOI/32/*.png'):
    img = misc.imread(filename)
    samples.append(img.reshape(-1))
    colors.append('b')
    
for filename in glob.glob('Datasets/ALOI/32i/*.png'):
    img = misc.imread(filename)
    samples.append(img.reshape(-1))
    colors.append('r')

# Load list to a DataFrame
df = pd.DataFrame.from_records(samples, coerce_float=True)

# Isomap 3 components
iso = manifold.Isomap(n_neighbors=6, n_components=3)
iso.fit(df)
manifold.Isomap(eigen_solver='auto', max_iter=None, n_components=3, n_neighbors=6, 
       neighbors_algorithm='auto', path_method='auto', tol=0)
T = iso.transform(df)

# Plot first 2 components in a 2D Scatter Plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('2D Scatter Isomap')
ax.set_xlabel('Component 0')
ax.set_ylabel('Component 1')
ax.scatter(T[:,0],T[:,1], marker='.', c=colors, alpha=0.7)

# Plot all 3 components in a 3D Scatter Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Scatter Isomap')
ax.set_xlabel('Component 0')
ax.set_ylabel('Component 1')
ax.set_zlabel('Component 2')
ax.scatter(T[:,0],T[:,1], T[:,2], marker='.', c=colors, alpha=0.7)

plt.show()