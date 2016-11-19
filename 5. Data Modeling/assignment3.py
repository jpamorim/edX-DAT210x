import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot') # Look Pretty

#
# INFO: This dataset has call records for 10 users tracked over the course of 3 years.
# Your job is to find out where the users likely live at!



def showandtell(title=None):
  if title != None: plt.savefig(title + ".png", bbox_inches='tight', dpi=300)
  plt.show()
  exit()

def clusterInfo(model):
  print "Cluster Analysis Inertia: ", model.inertia_
  print '------------------------------------------'
  for i in range(len(model.cluster_centers_)):
    print "\n  Cluster ", i
    print "    Centroid ", model.cluster_centers_[i]
    print "    #Samples ", (model.labels_==i).sum() # NumPy Power

# Find the cluster with the least # attached nodes
def clusterWithFewestSamples(model):
  # Ensure there's at least on cluster...
  minSamples = len(model.labels_)
  minCluster = 0
  for i in range(len(model.cluster_centers_)):
    if minSamples > (model.labels_==i).sum():
      minCluster = i
      minSamples = (model.labels_==i).sum()
  print "\n  Cluster With Fewest Samples: ", minCluster
  return (model.labels_==minCluster)


def doKMeans(data, clusters=0):
    data = data[['TowerLon', 'TowerLat']]
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('K-Means')
    ax.set_xlabel('TowerLon')
    ax.set_ylabel('TowerLat')
    ax.scatter(data.TowerLon, data.TowerLat, marker='.', alpha=0.3)
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
    return kmeans


# Load data
df = pd.read_csv('Datasets/CDR.csv')
df.CallDate = pd.to_datetime(df.CallDate, errors='coerce')
df.CallTime = pd.to_timedelta(df.CallTime, errors='coerce')

np_in = np.array(df.In)
user1 = df[df.In == np_in[0]]

print "\n\nExamining person: ", 0

user1 = user1[(user1.DOW != 'Sat') & (user1.DOW != 'Dom')]
user1 = user1[user1.CallTime < '17:00:00']
print len(user1)

# Plot the Cell Towers the user connected to
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Week Calls < 5pm')
ax.set_xlabel('TowerLon')
ax.set_ylabel('TowerLat')
ax.scatter(user1.TowerLon, user1.TowerLat, c='g', marker='o', alpha=0.2)
showandtell()

# Run k means
model = doKMeans(user1, 3)

# What time, on average, is the user in between home and work, between the midnight and 5pm?
midWayClusterIndices = clusterWithFewestSamples(model)
midWaySamples = user1[midWayClusterIndices]
print "    Its Waypoint Time: ", midWaySamples.CallTime.mean()


# First draw the X's for the clusters:
ax.scatter(model.cluster_centers_[:,1], model.cluster_centers_[:,0], s=169, c='r', marker='x', alpha=0.8, linewidths=2)

# Then save the results:
showandtell('Weekday Calls Centroids')
