import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use('ggplot')

df = pd.read_csv('Datasets/wheat.data', sep=',')

fig = plt.figure()
plt.suptitle('Area vs Perimeter vs Asymmetry')
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('area')
ax.set_ylabel('perimeter')
ax.set_zlabel('asymmetry')
ax.scatter(df.area, df.perimeter, df.asymmetry, c='r', marker='o')

fig = plt.figure()
plt.suptitle('Width vs Groove vs Length')
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('width')
ax.set_ylabel('groove')
ax.set_zlabel('length')
ax.scatter(df.width, df.groove, df.length, c='g', marker='o')
plt.show()