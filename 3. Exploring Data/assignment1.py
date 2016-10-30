import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

df = pd.read_csv('Datasets/wheat.data', sep=',')

s1 = df.loc[:, ['area', 'perimeter']]

s2 = df.loc[:, ['groove', 'asymmetry']]

plt.figure()
s1.plot.hist(alpha=0.3)
s2.plot.hist(alpha=0.6)
plt.show()
