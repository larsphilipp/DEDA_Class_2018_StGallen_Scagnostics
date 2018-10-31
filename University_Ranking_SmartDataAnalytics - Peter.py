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
import matplotlib.pyplot as plt
import seaborn as sns


# data scrape
data = pd.read_csv('/Users/PeterlaCour/documents/MIQEF/Smart Data Analytics/masters-in-management-2018.csv',encoding='ISO-8859-1', delimiter = ";")









df= pd.DataFrame(data)


df.head()


# ----------------------------------------------------------------------------


# deleting columns with strings
df_number = df.copy()
string_columns = ["Relevant degree","Programme name","Country","School name","Maximum course fee (local currency)","Employed at three months (%)"]
for i in string_columns:
    df_number.drop(i, axis=1, inplace = True)



column_names = df_number.columns


df_number.head()
column_names


# drop unnecessary columns
df_number_cleaned = df_number.copy()
string_columns_2 = ["2017","2016","3-year average","Salary today (US$)","Number enrolled 2017/18"]
for i in string_columns_2:
    df_number_cleaned.drop(i, axis=1, inplace = True)


# scatter-plots with ranking on y axis

for i in column_names:
    print("\n","\n",i)
    p = sns.relplot(y='2018',x=i,data = df_number)
    #p.show()
    #p.savefig('/Users/PeterlaCour/documents/MIQEF/Smart Data Analytics/'+i)


# Pair Plot
pairs = sns.pairplot(df_number_cleaned, diag_kind="kde", markers="+",plot_kws=dict(s=50, edgecolor="b", linewidth=1),diag_kws=dict(shade=True))





c = set(df["Country"])




top_10_unis = df_number_cleaned[0:10]
top_10_unis.head()


column_names = top_10_unis.columns

# scatter-plots with ranking on y axis

for i in column_names:
    print("\n","\n",i)
    p = sns.relplot(y='2018',x=i,data = top_10_unis)
    p.savefig('/Users/PeterlaCour/documents/MIQEF/SDA_UniRanking/Scatter Plots/top10 '+i)


# Pair Plot
pairs10 = sns.pairplot(top_10_unis, diag_kind="kde", markers="+",plot_kws=dict(s=50, edgecolor="b", linewidth=1),diag_kws=dict(shade=True))
pairs10.savefig('/Users/PeterlaCour/documents/MIQEF/SDA_UniRanking/top10pairs.png')






# ----------------------------------------------------



import networkx as nx
import pickle


g = nx.from_pandas_edgelist(df_number_cleaned[["2018","Career progress rank"]], source='Career progress rank', target='2018') 





nx.write_gpickle(g, '/Users/PeterlaCour/documents/MIQEF/SDA_UniRanking/UniRanking.gpickle')

filename = '/Users/PeterlaCour/documents/MIQEF/SDA_UniRanking/UniRanking.gpickle'
G = nx.read_gpickle(filename)
mst = nx.minimum_spanning_tree(G) 

# Uses Kruskal’s algorithm.
print(sorted(mst.edges(data=True)))

# minimum_spanning_edges(G, weight='weight', data=True)

nx.draw(G, markers="+")

nx.draw_networkx(G)

nx.draw_networkx(mst)

nx.draw_shell(mst)


mst

import pygraphviz as pgv
import pydot
A = pgv.write_dot(G)

A.layout()
A.draw('simple.png')




# ----------------------------------------------------

cor_plot = plt.matshow(df_number_cleaned.corr(),cmap = "RdYlGn")
plt.show()

plt.savefig('/Users/PeterlaCour/documents/MIQEF/Smart Data Analytics/correlation.png')



correlation_table = df_number_cleaned.corr()
c_table  = sns.heatmap(correlation_table, 
            xticklabels=correlation_table.columns.values,
            yticklabels=correlation_table.columns.values,
            cmap = "RdYlGn")
c_table.savefig('/Users/PeterlaCour/documents/MIQEF/Smart Data Analytics/correlation.png')











#-------------------------------------------------------------


import os
import rpy2.robjects as robjects
path = '.'

def scagnostics(x, y):
    all_scags = {}
    r_source = robjects.r['source']
    r_source('/Users/PeterlaCour/documents/MIQEF/Smart Data Analytics/scagnostics-master/get_scag.r')
    r_getname = robjects.globalenv['scags']
    scags = r_getname(robjects.FloatVector(x), robjects.FloatVector(y))
    all_scags['outlying'] = scags[0]
    all_scags['skewed'] = scags[1]
    all_scags['clumpy'] = scags[2]
    all_scags['sparse'] = scags[3]
    all_scags['striated'] = scags[4]
    all_scags['convex'] = scags[5]
    all_scags['skinny'] = scags[6]
    all_scags['stringy'] = scags[7]
    all_scags['monotonic'] = scags[8]
    return all_scags


x = scagnostics(df_number_cleaned["2018"],df_number_cleaned["Career progress rank"])



all_scags = {}
r_source = robjects.r['source']
r_source('/Users/PeterlaCour/documents/MIQEF/Smart Data Analytics/scagnostics-master/get_scag.r')
r_getname = robjects.globalenv['scags']
scags = r_getname(robjects.FloatVector([1,2,3,4,5,6,7,7]), robjects.FloatVector([1,2,3,4,5,3,1,7]))
all_scags['outlying'] = scags[0]
all_scags['skewed'] = scags[1]
all_scags['clumpy'] = scags[2]
all_scags['sparse'] = scags[3]
all_scags['striated'] = scags[4]
all_scags['convex'] = scags[5]
all_scags['skinny'] = scags[6]
all_scags['stringy'] = scags[7]
all_scags['monotonic'] = scags[8]


y=

df_number_cleaned["Career progress rank"].iloc[:].values

np.asarray(df_number_cleaned["Career progress rank"])


all_scags = {}
r_source = robjects.r['source']
r_source('/Users/PeterlaCour/documents/MIQEF/Smart Data Analytics/scagnostics-master/get_scag.r')
r_getname = robjects.r['scags']
scags = r_getname(robjects.FloatVector([1,2,3,4,5,6,7,7]), robjects.FloatVector([1,2,3,4,5,3,1,7]))
all_scags['outlying'] = scags[0]
all_scags['skewed'] = scags[1]
all_scags['clumpy'] = scags[2]
all_scags['sparse'] = scags[3]
all_scags['striated'] = scags[4]
all_scags['convex'] = scags[5]
all_scags['skinny'] = scags[6]
all_scags['stringy'] = scags[7]
all_scags['monotonic'] = scags[8]

r_getname



scagnostics_names = ["Outlying", "Skewed", "Clumpy", "Sparse", "Striated", "Convex", "Skinny", "Stringy", "Monotonic"]


from rpy2.robjects.packages import STAP
# if rpy2 < 2.6.1 do:
# from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
# STAP = SignatureTranslatedAnonymousPackage
with open('/Users/PeterlaCour/documents/MIQEF/Smart Data Analytics/scagnostics-master/get_scag.r', 'r') as f:
    string = f.read()
myfunc = STAP(string, "scags")

string


import rpy2.robjects as robjects
r_source = robjects.r['source']
r_source('/Users/PeterlaCour/documents/MIQEF/Smart Data Analytics/scagnostics-master/R/scagnostics.r')
r_getname = robjects.globalenv['scagnostics']

x = [1,2,4,1,45,31]
y = [5,3,4,1,45,31]
r_getname(robjects.FloatVector(x),robjects.FloatVector(y))






import choix


column_names

ranks = df_number_cleaned["2018"]
parameters = df_number_cleaned["Career progress rank"]

x = choix.generate_rankings(parameters,99,1)
x
choix.log_likelihood_rankings(df_number_cleaned,parameters)

lists = [df["2018"],df["Career progress rank"]]

l = []
for i in df["2018"]:
    l.append(i)
l 

l2 = []
for i in df["Career progress rank"]:
    l2.append(i)

l3 = [l,l2]
    
params = choix.generate_rankings(df_number_cleaned["Career progress rank"],100,3) 
    
choix.compare(l,params, rank = True)  
params




y = df_number_cleaned[["Career progress rank", "International mobility rank"]]


params = choix.generate_rankings( y,100,3) 
params
choix.probabilities(ranks, params)
pairs
pairs.savefig("PairPlot.png")
columns = df_number_cleaned.columns
columns

