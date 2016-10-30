from pandas.tools.plotting import andrews_curves
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

df = pd.read_csv('Datasets/wheat.data', sep=',')

# Drop the id, area, and perimeter features
df_save = df
df = df.loc[:, df.columns[3:]]

plt.figure()
andrews_curves(df, 'wheat_type', alpha=0.4)

# Add are and perimeter features
df = df_save.loc[:, df.columns[1:]]
plt.figure()
andrews_curves(df, 'wheat_type', alpha=0.4)
plt.show()