import pandas as pd

# Make new dataframe with 1 col (income)
df = pd.DataFrame({
                   'income': [
                              '$10000 - $50000',
                              '$50000 - $100000',
                              '+$100000',
                              '$1 - $1000',
                              '$1000 - $10000',
                              '$10000 - $50000',
                              '$50000 - $100000',
                              '+$100000',
                              '+$100000'
                              ]
                })
# series of income levels
income_ordered = [
                  '$1 - $1000',
                  '$1000 - $10000',
                  '$10000 - $50000',
                  '$50000 - $100000',
                  '+$100000'
                  ]

print(df)
# change category to numeric
df.income = df.income.astype("category", 
                             ordered=True, 
                             categories=income_ordered
                             ).cat.codes
print(df)

## Unordered
df = pd.DataFrame({'vertebrates': [
                                  'Bird',
                                  'Bird',
                                  'Mammal',
                                  'Fish',
                                  'Amphibian',
                                  'Reptile',
                                  'Mammal',
                                  ]
                })

print(df)
df.vertebrates = df.vertebrates.astype("category").cat.codes
print(df)

# or better -> each category to a new column (better because thats no 
# relation between different types of vertebrates but if they are represented as
# ordered a machine learning algorithm may use the relation)
df = pd.DataFrame({'vertebrates': [
                                  'Bird',
                                  'Bird',
                                  'Mammal',
                                  'Fish',
                                  'Amphibian',
                                  'Reptile',
                                  'Mammal',
                                  ]
                })

print(df)
df = pd.get_dummies(df, columns=['vertebrates'])
print(df)


