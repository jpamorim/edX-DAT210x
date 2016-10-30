import pandas as pd

url = 'http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2'
columns = ['RK', 'PLAYER', 'TEAM',	'GP',	'G', 'A', 'PTS',	'+/-', 'PIM', 'PTS/G', 'SOG', 'PCT', 'GWG', 'PPG', 'PPA', 'SHG', 'SHA']

# Load dataframe
df = pd.read_html(url)[0]
df.columns = ['RK', 'PLAYER', 'TEAM',	'GP',	'G', 'A', 'PTS',	'+/-', 'PIM', 'PTS/G', 'SOG', 'PCT', 'GWG', 'PPG', 'PPA', 'SHG', 'SHA']
print(df)

# Drop rows with the headers
df = df.drop(df.index[[1, 13, 25, 37]])
print(df)

# Drop rows with the headers
print(df.drop(df.index[[1]]))

# Get rid of any erroneous rows that has at least 4 NANs in them
df = df[df.isnull().sum(axis=1) < 4]
print(df)

# Get rid of the RK column
print(df.iloc[:, 1:])

# Ensure there are no nan "holes" in your index
#print(df.interpolate(method='polynomial', order=2))

# Check the dtypes of all columns, and ensure those that should be numeric are numeric
print(df.dtypes)
df.RK = pd.to_numeric(df.RK, errors='coerce')
df.GP = pd.to_numeric(df.GP, errors='coerce')
df.G = pd.to_numeric(df.G, errors='coerce')
df.A = pd.to_numeric(df.A, errors='coerce')
df.PTS = pd.to_numeric(df.PTS, errors='coerce')
df['+/-'] = pd.to_numeric(df['+/-'], errors='coerce')
df.PIM = pd.to_numeric(df.PIM, errors='coerce')
df['PTS/G'] = pd.to_numeric(df['PTS/G'], errors='coerce')
df.SOG = pd.to_numeric(df.SOG, errors='coerce')
df.PCT = pd.to_numeric(df.PCT, errors='coerce')
df.GWG = pd.to_numeric(df.GWG, errors='coerce')	
df.PPG = pd.to_numeric(df.PPG, errors='coerce')	
df.PPA = pd.to_numeric(df.PPA, errors='coerce')	
df.SHG = pd.to_numeric(df.SHG, errors='coerce')	
df.SHA = pd.to_numeric(df.SHA, errors='coerce')
print(df.dtypes)

# After completing the 6 steps above, how many rows remain in this dataset? (Not to be confused with the index!)
print(len(df))

# How many unique PCT values exist in the table?
print(len(df.PCT.unique()))

# What is the value you get by adding the GP values at indices 15 and 16 of this table?
print(df.iloc[15].GP + df.iloc[16].GP)
