import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot') # Look Pretty


def drawLine(model, X_test, y_test, title):
  # This convenience method will take care of plotting your
  # test observations, comparing them to the regression line,
  # and displaying the R2 coefficient
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(X_test, y_test, c='g', marker='o')
  ax.plot(X_test, model.predict(X_test), color='orange', linewidth=1, alpha=0.7)

  print "Est 2014 " + title + " Life Expectancy: ", model.predict([[2014]])[0]
  print "Est 2030 " + title + " Life Expectancy: ", model.predict([[2030]])[0]
  print "Est 2045 " + title + " Life Expectancy: ", model.predict([[2045]])[0]

  score = model.score(X_test, y_test)
  title += " R2: " + str(score)
  ax.set_title(title)


  plt.show()

# Load data
X = pd.read_csv('Datasets/life_expectancy.csv', sep='\t')
#print X.describe()

from sklearn import linear_model
model = linear_model.LinearRegression()

X = pd.read_csv('Datasets/life_expectancy.csv', sep='\t')

from sklearn import linear_model
model = linear_model.LinearRegression()

# Predict WhiteMale by Year when Year < 1986
X_train = X[X.Year < 1986][['Year']]
y_train = X[X.Year < 1986][['WhiteMale']]

X_test = X[X.Year >= 1986][['Year']]
y_test = X[X.Year >= 1986][['WhiteMale']]

model.fit(X_train, y_train)

drawLine(model, X_test, y_test, 'WhiteMale')
print 'Actual WhiteMale in 2014: ' , X[X.Year == 2014]['WhiteMale']

#Predict BlackFemale by Year when Year < 1986
X_train = X[X.Year < 1986][['Year']]
y_train = X[X.Year < 1986][['BlackFemale']]

X_test = X[X.Year >= 1986][['Year']]
y_test = X[X.Year >= 1986][['BlackFemale']]

model.fit(X_train, y_train)

drawLine(model, X_test, y_test, 'BlackFemale')
print 'Actual BlackFemale in 2014: ' , X[X.Year == 2014]['BlackFemale']

# Correlation matrix
print X.corr()

plt.show()