#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:38:36 2018

@author: PeterlaCour
"""
import pandas as pd
import numpy as np
import time 
import datetime
import matplotlib as plt
import seaborn as sns

path = 'C:/Users/Lars Stauffenegger/Documents/MBF Unisg/Smart Data Analytics/University_Ranking/'
file = 'mmgmt18.csv'
data = pd.read_csv(path + file, encoding='ISO-8859-1', delimiter = ";")



df= pd.DataFrame(data)

df.head()

x = df['Weighted salary (US$)']

sns.relplot(y='2018',x='Weighted salary (US$)',data = df)
sns.relplot(y='2018',x='Salary today (US$)',data = df)

# deleting columns with strings
df_number = df.copy()
string_columns = ["Relevant degree","Programme name","Country","School name","Maximum course fee (local currency)","Employed at three months (%)"]
for i in string_columns:
    df_number.drop(i, axis=1, inplace = True)

column_names = df_number.columns

df_number.head()

#ind_var = 'Female faculty (%)'

    
for i in column_names:
    print("\n","\n",i)
    sns.relplot(y='2018',x=i,data = df_number)
    #plt.show()


df_number_cleaned = df_number.copy()
string_columns_2 = ["2017","2016","3-year average","Salary today (US$)","Number enrolled 2017/18"]
for i in string_columns_2:
    df_number_cleaned.drop(i, axis=1, inplace = True)

c = set(df["Country"])

len(c)

sns.pairplot(df_number_cleaned)

column_numbers = len(column_names)

column_numbers

sns.pairplot(df_number_cleaned, diag_kind="kde", markers="+",plot_kws=dict(s=50, edgecolor="b", linewidth=1),diag_kws=dict(shade=True),kind="reg")


import choix


column_names

ranks = ["df_number_cleaned"]

