import pandas as pd

# Needed to load from sql database table
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')

# Different loading methods
df = pd.read_excel('my_dataset.xlsx', 'Sheet1', na_values=['NA', '?'])
df = pd.read_json('my_dataset.json', orient='columns')
df = pd.read_csv('my_dataset.csv', sep=',')
df = pd.read_html('http://page.com/with/table.html')[0] #.read_html() returns list of dataframes, one for which table in the html page
df = pd.read_sql_table('my_table', engine, columns=['ColA', 'ColB'])

# Loading existing data
df = pd.read_csv('Datasets/iris.csv')
print(df.head(5))       # Display first 5 samples
print(df.tail(5))       # Display last 5 samples
print(df.describe())    # Describes numeric features
print(df.columns)       # Shows feature names
print(df.index)         # Show index values
print(df.dtypes)        # Show data types of each column

# Writing to disk
df.to_sql('table', engine)  # Column names are required, if the file loaded
                            # does not have headers in the .read_*() function
                            # the names parameter should be passed with the names
df.to_excel('dataset.xlsx')
df.to_json('dataset.json')
df.to_csv('dataset.csv')

# Rewrite column names
df.columns = ['new', 'column', 'header', 'labels']
