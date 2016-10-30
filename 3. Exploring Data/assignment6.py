import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/wheat.data', sep=',')

# Drop the id feature
df = df.loc[:, df.columns[1:]]

correlation = df.corr()
print(correlation)

plt.imshow(df.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(df.columns))]
plt.xticks(tick_marks, df.columns, rotation='vertical')
plt.yticks(tick_marks, df.columns)
plt.show()