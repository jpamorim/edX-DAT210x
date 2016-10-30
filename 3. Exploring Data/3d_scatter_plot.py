import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use('ggplot') # to look pretty

df = pd.read_csv('Datasets/wheat.data', sep=',')

# Scatter plot
fig = plt.figure()
plt.suptitle('Length vs Width vs Asymmetry')
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('length')
ax.set_ylabel('width')
ax.set_zlabel('asymmetry')
ax.scatter(df.length, df.width, df.asymmetry, c='r', marker='o')
plt.show()