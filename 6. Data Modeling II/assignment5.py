import pandas as pd
from sklearn import tree
from sklearn.cross_validation import train_test_split
from subprocess import call

# Load dataset
column_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
                'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                'stalk-surface-below-ring', 'stalk-color-above-ring',
                'stalk-color-below-ring', 'veil-type', 'veil-color',
                'ring-number', 'ring-type', 'spore-print-color', 'population', 
                'habitat']
                
X = pd.read_csv('Datasets/agaricus-lepiota.data', names=column_names, 
                index_col=False, na_values='?')


# Check for rows with missing values
#print X[pd.isnull(X).any(axis=1)]

         
# Drop rows with missing values
X = X.dropna()


# Copy labels to y variable
y = X['class']
X = X.drop('class', 1)


# Map label variable to 0 or 1
y = y.map({'e':0, 'p':1})


# Encode the entire dataset using dummies
X = pd.get_dummies(X)


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)



# Create an DecisionTree
dt = tree.DecisionTreeClassifier()

dt.fit(X_train, y_train)
score = dt.score(X_test, y_test)
print "High-Dimensionality Score: ", round((score*100), 3)


#
# output a .DOT file
tree.export_graphviz(dt.tree_, out_file='tree.dot', feature_names=X.columns)

call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'], shell=True)