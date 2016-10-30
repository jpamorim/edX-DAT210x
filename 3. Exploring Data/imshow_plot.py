import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

matplotlib.style.use('ggplot')

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Show imshow plot
plt.imshow(df.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(df.columns))]
plt.xticks(tick_marks, df.columns, rotation='vertical')
plt.yticks(tick_marks, df.columns)
plt.show()

# Pandas correlation function that return a matrix with feature correlation(-1 to 1)
print(df.corr())