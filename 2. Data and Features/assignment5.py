import pandas as pd

columns = ['education', 'age', 'capital-gain', 'race', 'capital-loss', 'hours-per-week', 'sex', 'classification']
df = pd.read_csv('Datasets/census.data', sep=',', names=columns)
print(df.head())


# Make sure any value that needs to be replaced with a NAN is set as such.
df = df.interpolate(method='polynomial', order=2)
print(df.head())

# Look through the dataset and ensure all of your columns have appropriate data types.
print(df.dtypes)
df['capital-gain'] = pd.to_numeric(df['capital-gain'], errors='coerce')
print(df.dtypes)

# Properly encode any ordinal features using the method discussed in the chapter.
df.classification = df.classification.astype("category", 
                             ordered=True, 
                             categories=['<=50K', '>50K']
                             ).cat.codes
education_categories = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th' , '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Bachelors', 'Masters','Doctorate']
df.education = df.education.astype("category", 
                             ordered=True, 
                             categories=education_categories
                             ).cat.codes
print(df.head())

# Properly encode any nominal features by exploding them out into new, separate, boolean features.
df = pd.get_dummies(df, columns=['race', 'sex'])
print(df.head())
