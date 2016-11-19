import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

matplotlib.style.use('ggplot') # Look Pretty

def doKMeans(df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Plot')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.scatter(df.Longitude, df.Latitude, marker='.', alpha=0.3)
    
    df = df[['Longitude', 'Latitude']]

    kmeans_model = KMeans(n_clusters=7)
    kmeans_model.fit(df)
    
    centroids = kmeans_model.cluster_centers_
    ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
    print centroids

# Load dataset and drop NaNs
df = pd.read_csv('Datasets/Crimes_2001_to_present.csv')
df = df.dropna()
df.Date = pd.to_datetime(df.Date, errors='coerce')

# Plot crimes and the 7 centroids
doKMeans(df)

# Filter crimes after '2011-01-01'
df = df[df.Date > np.datetime64('2011-01-01')]

# Plot crimes and the 7 centroids
doKMeans(df)

plt.show()