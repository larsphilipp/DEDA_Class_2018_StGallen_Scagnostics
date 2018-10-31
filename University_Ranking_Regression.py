import pandas as pd
import numpy as np
import matplotlib as mp

def func1(x):
    x/100
    return -np.log(1/x)#np.exp(x)/(1+np.exp(x))



Y = np.empty((0,0))
X = range(1,100)
for x in X:
    y = func1(x)
    Y = np.append([Y], [y])
    
mp.pyplot.scatter(Y,X)


# Importing and restructuring the data
path = 'C:/Users/Lars Stauffenegger/Documents/MBF Unisg/Smart Data Analytics/University_Ranking/'
file = 'masters-in-management-2018.xlsx'
df = pd.read_excel(path + file)
df.columns = df.iloc[0] # Change Headers of the Columns
df = df.iloc[1:]    # delete row 0 as this is now in the header
mydf = df.loc[:,["Delivery Date", asset_1, asset_2]]   # cut down df to my relevant assets