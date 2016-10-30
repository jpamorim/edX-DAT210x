import pandas as pd

df = pd.read_csv('Datasets/iris.csv', sep=',')

print(df.dtypes)

# Correct data types
df.Date = pd.to_datetime(df.Date, errors='coerce')
df.Height = pd.to_numeric(df.Height, errors='coerce')
df.Weight = pd.to_numeric(df.Weight, errors='coerce')
df.Age = pd.to_numeric(df.Age, errors='coerce')

print(df.dtypes)

# Check unique values
print(df['class'].unique())

# Check count of different values
print(df['class'].value_counts())