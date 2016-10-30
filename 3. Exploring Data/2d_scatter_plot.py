import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot') # to look pretty

df = pd.read_csv('Datasets/wheat.data', sep=',')

# Scatter plot
df.plot.scatter(x='area', y='asymmetry')
plt.suptitle('Area vs Asymmetry')
plt.xlabel('Area')
plt.ylabel('Asymmetry')
plt.show()