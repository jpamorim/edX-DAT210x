import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot') # to look pretty

df = pd.read_csv('Datasets/wheat.data', sep=',')

# Plot the asymmetry feature with 10 bins
df.asymmetry.plot.hist(title='Asymmetry', bins=10)
plt.show()

# Change wheat_type from nominal to numerical
df.wheat_type = df.wheat_type.astype('category',
                                     ordered=False,
                                     categories=['kama', 'canadian', 'rosa']
                                    ).cat.codes

# Plot asymmetry depending on wheat_type value
plt.figure()
df[df.wheat_type==2].asymmetry.plot.hist(alpha=0.3)
df[df.wheat_type==1].asymmetry.plot.hist(alpha=0.6)
df[df.wheat_type==0].asymmetry.plot.hist(alpha=0.6)
plt.show()