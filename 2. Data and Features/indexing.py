import pandas as pd

df = pd.read_csv('Datasets/direct_marketing.csv')

print(df.columns)

# Column Indexing
print(df.recency)   # dataset with slice by column name
print(df['recency'])  # dataset with slice by column name
print(df[['recency']]) # series with slice by column name

print(df.loc[0:5, 'recency'])   # dataset with the first 5 rows with only 1 col
print(df.loc[0:5, ['recency']]) # series with the first 5 rows

print(df.iloc[0:5, 0]) # dataset with the first 5 rows with only 1 col sliced by index
print(df.iloc[0:5, [0]]) # series with the first 5 rows sliced by index
print(df.ix[0:5, 0]) # dataset with the first 5 rows with only 1 col sliced by index

# Boolean Indexing
print(df.recency < 7)   # Boolean series where the value is a boolean
print(df[ df.recency < 7 ]) # Slices rows where recency<7
print(df[ (df.recency < 7) & (df.newbie == 0) ]) # and statement
print(df[ (df.recency < 7) | (df.newbie == 0) ]) # or statement

      