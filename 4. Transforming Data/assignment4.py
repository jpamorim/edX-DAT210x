import pandas as pd
import numpy as np
import scipy.io
import random, math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn import manifold

matplotlib.style.use('ggplot')

def Plot2D(T, title, x, y, num_to_plot=40):
  # This method picks a bunch of random samples (images in your case)
  # to plot onto the chart:
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title(title)
  ax.set_xlabel('Component: {0}'.format(x))
  ax.set_ylabel('Component: {0}'.format(y))
  x_size = (max(T[:,x]) - min(T[:,x])) * 0.08
  y_size = (max(T[:,y]) - min(T[:,y])) * 0.08
  for i in range(num_to_plot):
    img_num = int(random.random() * num_images)
    x0, y0 = T[img_num,x]-x_size/2., T[img_num,y]-y_size/2.
    x1, y1 = T[img_num,x]+x_size/2., T[img_num,y]+y_size/2.
    img = df.iloc[img_num,:].reshape(num_pixels, num_pixels)
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

  # It also plots the full scatter:
  ax.scatter(T[:,x],T[:,y], marker='.',alpha=0.7)
  
def Plot3D(T, title, x, y, z):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_title(title)
  ax.set_xlabel('Component: {0}'.format(x))
  ax.set_ylabel('Component: {0}'.format(y))
  ax.set_zlabel('Component: {0}'.format(z))
  ax.scatter(T[:,x],T[:,y], T[:,z], marker='.',alpha=0.7)


# Load dataset
mat = scipy.io.loadmat('Datasets/face_data.mat')
df = pd.DataFrame(mat['images']).T
num_images, num_pixels = df.shape
num_pixels = int(math.sqrt(num_pixels))

# Rotate the pictures, so we don't have to crane our necks:
for i in range(num_images):
  df.loc[i,:] = df.loc[i,:].reshape(num_pixels, num_pixels).T.reshape(-1)


# PCA 2 components
pca = PCA(n_components=2)
pca.fit(df)
PCA(copy=True, n_components=2, whiten=False)
T = pca.transform(df)
Plot2D(T, "PCA 2 Components", 0, 1)

# Isomap 2 components
iso = manifold.Isomap(n_neighbors=4, n_components=2)
iso.fit(df)
manifold.Isomap(eigen_solver='auto', max_iter=None, n_components=2, n_neighbors=4, 
       neighbors_algorithm='auto', path_method='auto', tol=0)
T = iso.transform(df)
Plot2D(T, "Isomap 2 Components", 0, 1)


# PCA 3 components
pca = PCA(n_components=3)
pca.fit(df)
PCA(copy=True, n_components=3, whiten=False)
T = pca.transform(df)
Plot3D(T, "PCA 3 Components", 0, 1, 2)

# Isomap 2 components
iso = manifold.Isomap(n_neighbors=16, n_components=3)
iso.fit(df)
manifold.Isomap(eigen_solver='auto', max_iter=None, n_components=3, n_neighbors=16, 
       neighbors_algorithm='auto', path_method='auto', tol=0)
T = iso.transform(df)
Plot3D(T, "Isomap 3 Components", 0, 1, 2)

plt.show()
