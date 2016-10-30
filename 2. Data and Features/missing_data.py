import pandas as pd

df = pd.read_csv('Datasets/iris.csv', sep=',')
print(df)

print(df.notnull())
print(df.isnull())

print(df.fillna(0))                         # fill NAN's with scalar

print(df.fillna(method='ffill', limit=1))   # fill fowards with previous value

print(df.interpolate(method='polynomial', order=2)) # interpolate new value

df.dropna(axis=0)   # Drop all rows with null/NAN values

df.dropna(axis=1)   # Drop all columns with null/NAN values

# Drop duplicates
print(df.drop_duplicates(subset=['petal_width', 'petal_length']).reset_index())