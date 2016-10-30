import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

df = pd.read_csv('Datasets/wheat.data', sep=',')

df.plot.scatter(x='area', y='perimeter')
plt.suptitle('Area vs Perimeter')
plt.xlabel('Area')
plt.ylabel('Perimeter')

df.plot.scatter(x='groove', y='asymmetry')
plt.suptitle('Groove vs Asymmetry')
plt.xlabel('Groove')
plt.ylabel('Asymmetry')

df.plot.scatter(x='compactness', y='width')
plt.suptitle('Compactness vs Width')
plt.xlabel('Compactness')
plt.ylabel('Width')

# Same plot as above but with other market
df.plot.scatter(x='compactness', y='width', marker='^')
plt.suptitle('Compactness vs Width')
plt.xlabel('Compactness')
plt.ylabel('Width')

plt.show()