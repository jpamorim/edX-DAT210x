import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

# Load dataset
X = pd.read_csv('Datasets/dataset-har-PUC-Rio-ugulino.csv', sep=';', decimal=',')

# Map gender, 0 as male, 1 as female
#X.gender = X.gender.map({'Man':0, 'Woman':1})

# Map gender, 1 as male, 0 as female
X.gender = X.gender.map({'Man':1, 'Woman':0})

# Gender dummies
gender = pd.get_dummies(X.gender)
X = pd.concat([X, gender], axis=1)
X = X.drop('gender', 1)

# Remove problematic row
X = X[X.z4 != '-14420-11-2011 04:50:23.713']


# Convert z4 to numeric
X.z4 = pd.to_numeric(X.z4, errors='raise')


# Check data types
#print X.dtypes


# Slice class to y and drop user column
y = X['class']
X = X.drop('class', 1)
X = X.drop('user', 1)


# Encode class as dummies
#y = pd.get_dummies(y)

# Encode class as numerical
y = y.astype("category").cat.codes


# Check rows with nans
#print X[pd.isnull(X).any(axis=1)]


# Create Random Forest classifier
model = RandomForestClassifier(n_estimators=30, max_depth=10, 
                               oob_score=True, random_state=0)


# Split train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


# Train model
print "Fitting..."
s = time.time()
model.fit(X_train, y_train)
print "Fitting completed in: ", time.time() - s


# Display oob_score
score = model.oob_score_
print "OOB Score: ", round(score*100, 3)

# Score model
print "Scoring..."
s = time.time()
score = model.score(X_test, y_test)
print "Score: ", round(score*100, 3)
print "Scoring completed in: ", time.time() - s