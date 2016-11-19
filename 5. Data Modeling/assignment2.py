import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans

matplotlib.style.use('ggplot') # Look Pretty

def showandtell(title=None):
  if title != None: plt.savefig(title + ".png", bbox_inches='tight', dpi=300)
  plt.show()

# Load data
df = pd.read_csv('Datasets/CDR.csv')
df.CallDate = pd.to_datetime(df.CallDate, errors='coerce')
df.CallTime = pd.to_timedelta(df.CallTime, errors='coerce')

np_in = np.array(df.In)
user1 = df[df.In == np_in[0]]

user1.plot.scatter(x='TowerLon', y='TowerLat', c='gray', alpha=0.1, title='Call Locations')
showandtell()


# On Weekdays:
#   1. People probably don't go into work
#   2. They probably sleep in late on Saturday
#   3. They probably run a bunch of random errands, since they couldn't during the week
#   4. They should be home, at least during the very late hours, e.g. 1-4 AM
#
# On Weekdays:
#   1. People probably are at work during normal working hours
#   2. They probably are at home in the early morning and during the late night
#   3. They probably spend time commuting between work and home everyday

user1 = user1[(user1.DOW == 'Sat') | (user1.DOW == 'Dom')]
user1 = user1[(user1.CallTime < '06:00:00') | (user1.CallTime > '22:00:00')]
print len(user1)

# At this point, you don't yet know exactly where the user is located just based off the cell
# phone tower position data; but considering the below are for Calls that arrived in the twilight
# hours of weekends, it's likely that wherever they are bunched up is probably near where the
# caller's residence:
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Weekend Calls (<6am or >10p)')
ax.set_xlabel('TowerLon')
ax.set_ylabel('TowerLat')
ax.scatter(user1.TowerLon, user1.TowerLat, c='g', marker='o', alpha=0.2)
showandtell()

df_for_kmeans = user1[['TowerLon', 'TowerLat']]
kmeans = KMeans(n_clusters=1)
kmeans.fit(df_for_kmeans)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('K-Means')
ax.set_xlabel('TowerLon')
ax.set_ylabel('TowerLat')
ax.scatter(df_for_kmeans.TowerLon, df_for_kmeans.TowerLat, marker='.', alpha=0.3)
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)

for x in range(len(np_in)):
    user1 = df[df.In == np_in[x]]
    user1 = user1[(user1.DOW == 'Sat') | (user1.DOW == 'Dom')]
    user1 = user1[(user1.CallTime < '06:00:00') | (user1.CallTime > '22:00:00')]
    df_for_kmeans = user1[['TowerLon', 'TowerLat']]
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(df_for_kmeans)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('K-Means')
    ax.set_xlabel('TowerLon')
    ax.set_ylabel('TowerLat')
    ax.scatter(df_for_kmeans.TowerLon, df_for_kmeans.TowerLat, marker='.', alpha=0.3)
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
