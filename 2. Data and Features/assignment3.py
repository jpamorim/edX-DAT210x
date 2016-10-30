import pandas as pd

df = pd.read_csv('Datasets/servo.data', sep=',', names=['motor', 'screw', 'pgain', 'vgain', 'class'])

 # How many samples in this dataset have a vgain feature value equal to 5?
print(len(df[df.vgain == 5]))

# How many samples in this dataset contain the value E for both motor and screw features?
print(len(df[(df.motor == 'E') & (df.screw == 'E')]))

# What is the mean vgain value of those samples that have a pgain feature value equal to 4?
print(len(df[(df.motor == 'E') & (df.screw == 'E')]))
print(df[df.pgain == 4].describe())