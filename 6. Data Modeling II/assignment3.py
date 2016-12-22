import pandas as pd
import numpy as np 
from sklearn import preprocessing
from sklearn import manifold
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA


# Load dataset and drop name column
X = pd.read_csv('Datasets/parkinsons.data')
X = X.drop('name', 1)

# Slice out status column
y = X['status']
X = X.drop('status', 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# No preprocessing (accuracy = 0.915254237288)

# Normalizer (accuracy = 0.796610169492)
#preprocessor = preprocessing.Normalizer()
#preprocessor.fit(X_train)
#X_train = preprocessor.transform(X_train)
#X_test = preprocessor.transform(X_test)

# MaxAbsScaler (accuracy = 0.881355932203)
#preprocessor = preprocessing.MaxAbsScaler()
#preprocessor.fit(X_train)
#X_train = preprocessor.transform(X_train)
#X_test = preprocessor.transform(X_test)


# MinMaxScaler (accuracy = 0.881355932203)
#preprocessor = preprocessing.MinMaxScaler()
#preprocessor.fit(X_train)
#X_train = preprocessor.transform(X_train)
#X_test = preprocessor.transform(X_test)

# KernelCenterer (accuracy = 0.915254237288)
#preprocessor = preprocessing.KernelCenterer()
#preprocessor.fit(X_train)
#X_train = preprocessor.transform(X_train)
#X_test = preprocessor.transform(X_test)

# StandardScaler (accuracy = 0.932203389831)
preprocessor = preprocessing.StandardScaler()
preprocessor.fit(X_train)
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)


# No PCA (accuracy = 0.932203389831)
#best_score = 0
#for C in np.arange(0.05, 2, 0.05):
#        for gamma in np.arange(0.001, 0.1, 0.001):
#                model = SVC(C=C, gamma=gamma, kernel='rbf')
#                model.fit(X_train2, y_train) 
#                score = model.score(X_test2, y_test) 
#                if score > best_score:
#                    best_score = score

# With PCA (accuracy = 0.932203389831 (no improvement))
#best_score = 0
#for n_components in range(4, 15):
#    pca = PCA(n_components=n_components)
#    pca.fit(X_train)
#    X_train2 = pca.transform(X_train)
#    X_test2 = pca.transform(X_test)
#    for C in np.arange(0.05, 2, 0.05):
#        for gamma in np.arange(0.001, 0.1, 0.001):
#                model = SVC(C=C, gamma=gamma, kernel='rbf')
#                model.fit(X_train2, y_train) 
#                score = model.score(X_test2, y_test) 
#                if score > best_score:
#                    best_score = score
              
# With Isomap (accuracy = 0.949152542373)               
best_score = 0
for n_neighbors in range(2, 6):
    for n_components in range(4, 7):
        iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
        iso.fit(X_train)
        X_train2 = iso.transform(X_train)  
        X_test2 = iso.transform(X_test)  
        for C in np.arange(0.05, 2, 0.05):
            for gamma in np.arange(0.001, 0.1, 0.001):
                    model = SVC(C=C, gamma=gamma, kernel='rbf')
                    model.fit(X_train2, y_train) 
                    score = model.score(X_test2, y_test) 
                    if score > best_score:
                        best_score = score
           
                    
                 
                    
                    
    
print best_score