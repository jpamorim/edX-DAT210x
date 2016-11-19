import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import train_test_split


matplotlib.style.use('ggplot') # Look Pretty



def drawLine(model, X_test, y_test, title, R2):
  # This convenience method will take care of plotting your
  # test observations, comparing them to the regression line,
  # and displaying the R2 coefficient
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(X_test, y_test, c='g', marker='o')
  ax.plot(X_test, model.predict(X_test), color='orange', linewidth=1, alpha=0.7)

  title += " R2: " + str(R2)
  ax.set_title(title)
  print title
  print "Intercept(s): ", model.intercept_

  plt.show()

def drawPlane(model, X_test, y_test, title, R2):
  # This convenience method will take care of plotting your
  # test observations, comparing them to the regression plane,
  # and displaying the R2 coefficient
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.set_zlabel('prediction')

  # You might have passed in a DataFrame, a Series (slice),
  # an NDArray, or a Python List... so let's keep it simple:
  X_test = np.array(X_test)
  col1 = X_test[:,0]
  col2 = X_test[:,1]

  # Set up a Grid. We could have predicted on the actual
  # col1, col2 values directly; but that would have generated
  # a mesh with WAY too fine a grid, which would have detracted
  # from the visualization
  x_min, x_max = col1.min(), col1.max()
  y_min, y_max = col2.min(), col2.max()
  x = np.arange(x_min, x_max, (x_max-x_min) / 10)
  y = np.arange(y_min, y_max, (y_max-y_min) / 10)
  x, y = np.meshgrid(x, y)

  # Predict based on possible input values that span the domain
  # of the x and y inputs:
  z = model.predict(  np.c_[x.ravel(), y.ravel()]  )
  z = z.reshape(x.shape)

  ax.scatter(col1, col2, y_test, c='g', marker='o')
  ax.plot_wireframe(x, y, z, color='orange', alpha=0.7)
  
  title += " R2: " + str(R2)
  ax.set_title(title)
  print title
  print "Intercept(s): ", model.intercept_
  
  plt.show()
  


# Load data
X = pd.read_csv('Datasets/College.csv', index_col=0)
X.Private = X.Private.map({'Yes':1, 'No':0})

from sklearn import linear_model
model = linear_model.LinearRegression()

# Accept as a function of Room & Board
room_board = X[['Room.Board']]
accept = X[['Accept']]

# Split data in training and test set 
X_train, X_test, y_train, y_test = train_test_split(room_board, accept, test_size=0.3, random_state=7)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
drawLine(model, X_test, y_test, "Accept(Room&Board)", score)

# Accept as a function of Number of Enrolled Students
enroll = X[['Enroll']]
accept = X[['Accept']]

# Split data in training and test set 
X_train, X_test, y_train, y_test = train_test_split(enroll, accept, test_size=0.3, random_state=7)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
drawLine(model, X_test, y_test, "Accept(Enroll)", score)

# Accept as a function of Failed Undergraduates
failed_undergrad = X[['F.Undergrad']]
accept = X[['Accept']]

# Split data in training and test set 
X_train, X_test, y_train, y_test = train_test_split(failed_undergrad, accept, test_size=0.3, random_state=7)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
drawLine(model, X_test, y_test, "Accept(F.Undergrad)", score)


# Multivariate linear regression
# Accept as a function of amount charged for room and board AND the number of enrolled students
X_multivariate = X[['Room.Board', 'Enroll']]
accept = X[['Accept']]

# Split data in training and test set 
X_train, X_test, y_train, y_test = train_test_split(X_multivariate, accept, test_size=0.3, random_state=7)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
drawPlane(model, X_test, y_test, "Accept(Room&Board,Enroll)", score)