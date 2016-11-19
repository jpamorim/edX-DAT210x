import random, math
import pandas as pd
import numpy as np
import scipy.io
from sklearn.cross_validation import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib

# If you'd like to try this lab with PCA instead of Isomap,
# as the dimensionality reduction technique:
Test_PCA = False

matplotlib.style.use('ggplot') # Look Pretty


def Plot2DBoundary(DTrain, LTrain, DTest, LTest):
  # The dots are training samples (img not drawn), and the pics are testing samples (images drawn)
  # Play around with the K values. This is very controlled dataset so it should be able to get perfect classification on testing entries
  # Play with the K for isomap, play with the K for neighbors. 

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title('Transformed Boundary, Image Space -> 2D')

  padding = 0.1   # Zoom out
  resolution = 1  # Don't get too detailed; smaller values (finer rez) will take longer to compute
  colors = ['blue','green','orange','red']
  

  # ------

  # Calculate the boundaries of the mesh grid. The mesh grid is
  # a standard grid (think graph paper), where each point will be
  # sent to the classifier (KNeighbors) to predict what class it
  # belongs to. This is why KNeighbors has to be trained against
  # 2D data, so we can produce this countour. Once we have the 
  # label for each point on the grid, we can color it appropriately
  # and plot it.
  x_min, x_max = DTrain[:, 0].min(), DTrain[:, 0].max()
  y_min, y_max = DTrain[:, 1].min(), DTrain[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Using the boundaries, actually make the 2D Grid Matrix:
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say about each spot on the chart?
  # The values stored in the matrix are the predictions of the model
  # at said location:
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the mesh grid as a filled contour plot:
  plt.contourf(xx, yy, Z, cmap=plt.cm.terrain, z=-100)


  # ------

  # When plotting the testing images, used to validate if the algorithm
  # is functioning correctly, size them as 5% of the overall chart size
  x_size = x_range * 0.05
  y_size = y_range * 0.05
  
  # First, plot the images in your TEST dataset
  img_num = 0
  for index in LTest.index:
    # DTest is a regular NDArray, so you'll iterate over that 1 at a time.
    x0, y0 = DTest[img_num,0]-x_size/2., DTest[img_num,1]-y_size/2.
    x1, y1 = DTest[img_num,0]+x_size/2., DTest[img_num,1]+y_size/2.

    # DTest = our images isomap-transformed into 2D. But we still want
    # to plot the original image, so we look to the original, untouched
    # dataset (at index) to get the pixels:
    img = df.iloc[index,:].reshape(num_pixels, num_pixels)
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1), alpha=0.8)
    img_num += 1


  # Plot your TRAINING points as well... as points rather than as images
  for label in range(len(np.unique(LTrain))):
    indices = np.where(LTrain == label)
    ax.scatter(DTrain[indices, 0], DTrain[indices, 1], c=colors[label], alpha=0.8, marker='o')

  # Plot
  plt.show()  

def doPCA(data, dimensions=2):
    from sklearn.decomposition import RandomizedPCA
    model = RandomizedPCA(n_components=dimensions)
    model.fit(data)
    return model

def doIsomap(data, dimensions=2, n_neighbors=4):
    from sklearn import manifold
    model = manifold.Isomap(n_neighbors=n_neighbors, n_components=dimensions)
    model.fit(data)
    return model
    
def doKNeighbors(X, y, n_neighbors=9):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors)
    model.fit(X, y)
    return model

# Load dataset
mat = scipy.io.loadmat('Datasets/face_data.mat')
df = pd.DataFrame(mat['images']).T
num_images, num_pixels = df.shape
num_pixels = int(math.sqrt(num_pixels))

# Rotate the pictures, so we don't have to crane our necks:
for i in range(num_images):
  df.loc[i,:] = df.loc[i,:].reshape(num_pixels, num_pixels).T.reshape(-1)

# Last row all 0's
df = df.iloc[0:697,:]

# Load labels
face_labels = pd.read_csv('Datasets/face_labels.csv')
face_labels = face_labels.iloc[:,0]

# Split data in training and test set 
X_train, X_test, y_train, y_test = train_test_split(df, face_labels, test_size=0.2, random_state=7)


if Test_PCA:
    pca_model = doPCA(X_train)
    X_train = pca_model.transform(X_train)
    X_test = pca_model.transform(X_test)
else:
    iso_model = doIsomap(X_train)
    X_train = iso_model.transform(X_train)
    X_test = iso_model.transform(X_test)

model = doKNeighbors(X_train, y_train, 8)
print 'Score: ' , model.score(X_test, y_test)

# Chart the combined decision boundary, the training data as 2D plots, and
# the testing data as small images so we can visually validate performance.
Plot2DBoundary(X_train, y_train, X_test, y_test)


