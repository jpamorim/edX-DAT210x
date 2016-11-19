import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


# If you'd like to try this lab with PCA instead of Isomap,
# as the dimensionality reduction technique:
Test_PCA = False


def plotDecisionBoundary(model, X, y):
  print "Plotting..."
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') # Look Pretty

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  #(2 for benign, 4 for malignant)
  colors = {2:'royalblue',4:'lightsalmon'} 

  
  # Calculate the boundaris
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at at said location
  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  # Plot your testing points as well...
  for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

  p = model.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
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
    model = KNeighborsClassifier(n_neighbors, weights ='distance')
    model.fit(X, y)
    return model

names=['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status']
X = pd.read_csv('Datasets/breast-cancer-wisconsin.data', names=names)

# Check for nans
#print X.isnull().sum()

X.nuclei = X.nuclei.astype("category").cat.codes

# Remove labels
y = X['status']
X = X.drop('status', 1)

# Fill NaNs with the mean
X = X.fillna(X.mean())


# Split data in training and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=7)

# StandardScaler (accuracy = 0.954285714286)
preprocessor = preprocessing.StandardScaler()
preprocessor.fit(X_train)
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)


# MinMaxScaler (accuracy = 0.954285714286)
#preprocessor = preprocessing.MinMaxScaler()
#preprocessor.fit(X_train)
#X_train = preprocessor.transform(X_train)
#X_test = preprocessor.transform(X_test)

# MaxAbsScaler (accuracy = 0.954285714286)
#preprocessor = preprocessing.MaxAbsScaler()
#preprocessor.fit(X_train)
#X_train = preprocessor.transform(X_train)
#X_test = preprocessor.transform(X_test)

# Normalizer (accuracy = 0.908571428571)
#preprocessor = preprocessing.Normalizer()
#preprocessor.fit(X_train)
#X_train = preprocessor.transform(X_train)
#X_test = preprocessor.transform(X_test)


# PCA and Isomap
if Test_PCA:
  print "Computing 2D Principle Components"
  model = doPCA(X_train, 2)
else:
  print "Computing 2D Isomap Manifold"
  model = doIsomap(X_train, 2, 5)
  
X_train = model.transform(X_train)
X_test = model.transform(X_test)


knmodel = doKNeighbors(X_train, y_train, 8)
print 'Score: ' , knmodel.score(X_test, y_test)

#plotDecisionBoundary(knmodel, X_test, y_test)