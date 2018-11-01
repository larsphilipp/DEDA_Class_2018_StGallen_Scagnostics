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
#data = pd.read_csv('/Users/PeterlaCour/documents/MIQEF/Smart Data Analytics/masters-in-management-2018.csv',encoding='ISO-8859-1', delimiter = ";")


import requests
from bs4 import BeautifulSoup
import pandas as pd

url_wo_year='http://rankings.ft.com/businessschoolrankings/masters-in-management-'
years = ['2014','2015','2016','2017','2018']
output_dict = {}
for year in years:
    url = url_wo_year + year
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    rankings_table = soup.find(id = "rankingstable")
    rankings_table.prettify()
    output = pd.read_html(url)
    k = 0
    for i in output[0].columns:
        if "Unnamed" in i:
            output[0].drop("Unnamed: " + str(k),axis =1, inplace = True)
            k += 1
    output = output[0][:-1]
    
    output_dict[year] = output
    
print(output_dict)  


df = output_dict["2018"]

column_names = df.columns
column_names
# ----------------------------------------------------------------------------

# clean column names?

#df.to_csv(path_or_buf= '/Users/PeterlaCour/documents/MIQEF/SDA_UniRanking/2018 Scrape.csv')


# deleting columns with strings
df_number = df.copy()
string_columns = ["Relevant degree","Programme name","Country","School name","Maximum course fee (local currency)","Employment[5]"]
for i in string_columns:
    df_number.drop(i, axis=1, inplace = True)





df_number.head()
column_names


# drop unnecessary columns
df_number_cleaned = df_number.copy()
string_columns_2 = ["2017","2016","3-year average","Salary today (US$)[1]","Number enrolled 2017/18"]
for i in string_columns_2:
    df_number_cleaned.drop(i, axis=1, inplace = True)

df_number_cleaned.dropna(axis = 1, how = 'any')
column_names = df_number_cleaned.columns

# scatter-plots with ranking on y axis
for i in column_names:
    print("\n","\n",i)
    p = sns.relplot(y='2018',x=i,data = df_number_cleaned)
    #p.show()
    #p.savefig('/Users/PeterlaCour/documents/MIQEF/Smart Data Analytics/'+i)


# Pair Plot
pairs = sns.pairplot(df_number_cleaned, diag_kind="kde", markers="+",plot_kws=dict(s=50, edgecolor="b", linewidth=1),diag_kws=dict(shade=True))

type(df_number_cleaned["2018"][1])



c = set(df["Country"])





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




# ----------------------------------------------------






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



df_number_cleaned[["2018","Career progress rank"]]


tuple_array = []

for i in range(len(df_number_cleaned["2018"])):
    tuple_array.append((df_number_cleaned["2018"][i],df_number_cleaned["Career progress rank"][i]))


# minimum_spanning tree
minimum_edge = []

for o in range(len(tuple_array)):
    edges_length = []
    m,n = tuple_array[o]
    
    for l in range(len(tuple_array)):
        i,j = tuple_array[l]
        edges_length.append(np.sqrt((m-i)**2+(n-j)**2))
    edges_length.pop(o)
    minimum_edge.append(min(edges_length))

total_edge_length = sum(minimum_edge)

# outliers = c = length(T_outliers)/length(T)
q25                     = np.quantile(minimum_edge,0.25)
q75                     = np.quantile(minimum_edge,0.75)
length_of_long_edges    = q75 - 1.5 * (q75 - q25)
c_outliers              = length_of_long_edges / total_edge_length



# convex = area(A)/area(H) - area of alpha hull / convex hull
    



    
# skinny = 1 - sqrt(4pi * area(a))) / perimeter(a)



# skewness: ratio of edge lengths in edge distribution: (q90 - q50) / (q90 - q10)
q90         = np.quantile(minimum_edge,0.9)
q50         = np.quantile(minimum_edge,0.5)
q10         = np.quantile(minimum_edge,0.1)
skewness    = (q90 - q50) / (q90 - q10)



# clumpy = max(1-max(legnth(ek))/length(ej)) - ????????
    
    
    

# sparsity = min(1,q90)?
sparsity = min(1,q90) 
sparsity






# delaunay triangulation
from scipy import spatial
delaunay_triangulation = spatial.Delaunay(tuple_array)
convex_hull = delaunay_triangulation.convex_hull








# R PACKAGE EXPLANATION: https://cran.r-project.org/web/packages/scagnostics/index.html













from scipy.spatial import Delaunay, ConvexHull
import networkx as nx
 


def concave(points,alpha_x=150,alpha_y=250):
    de = Delaunay(points)
    dec = []
    a = alpha_x
    b = alpha_y
    for i in de.simplices:
        tmp = []
        j = [points[c] for c in i]
        if abs(j[0][1] - j[1][1])>a or abs(j[1][1]-j[2][1])>a or abs(j[0][1]-j[2][1])>a or abs(j[0][0]-j[1][0])>b or abs(j[1][0]-j[2][0])>b or abs(j[0][0]-j[2][0])>b:
            continue
        for c in i:
            tmp.append(points[c])
        dec.append(tmp)
    G = nx.Graph()
    for i in dec:
            G.add_edge(i[0], i[1])
            G.add_edge(i[0], i[2])
            G.add_edge(i[1], i[2])
    '''
    ret = []
    for graph in nx.connected_component_subgraphs(G):
        ch = ConvexHull(graph.nodes())
        print(graph.nodes())
        tmp = []
        for i in ch.simplices:
            tmp.append(graph.nodes()[i[0]])
            tmp.append(graph.nodes()[i[1]])
        ret.append(tmp)
    return ret  
    #return [graph.nodes() for graph in nx.connected_component_subgraphs(G)] - all points inside the shape
    '''
    return G

p = concave(tuple_array)
pos=nx.spring_layout(p)

pos = dict( (n, n) for n in tuple_array)
#nx.draw_networkx(p,node_size = 15, dim = 3, pos = pos)
mst = nx.minimum_spanning_tree(p,algorithm = 'boruvka') 
nx.draw(mst,node_size = 15, dim = 3, pos = pos)
nx.draw_networkx(mst,node_size = 15, dim = 3, pos = pos)

list(mst)

plt.scatter(df_number_cleaned["2018"],df_number_cleaned["Career progress rank"])


'''
circular_layout(G[, dim, scale, center])	Position nodes on a circle.
random_layout(G[, dim, center])	Position nodes uniformly at random in the unit square.
shell_layout(G[, nlist, dim, scale, center])	Position nodes in concentric circles.
spring_layout(G[, dim, k, pos, fixed, ...])	Position nodes using Fruchterman-Reingold force-directed algorithm.
spectral_layout(G[, dim, weight, scale, center])	Position nodes using the eigenvectors of the graph Laplacian.
''''





'''

# R SCAGNOSTICS


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


'''








'''

import networkx as nx
import pickle


g = nx.from_pandas_edgelist(df_number_cleaned[["2018","Career progress rank"]], source='Career progress rank', target='2018') 





nx.write_gpickle(g, '/Users/PeterlaCour/documents/MIQEF/SDA_UniRanking/UniRanking.gpickle')

filename = '/Users/PeterlaCour/documents/MIQEF/SDA_UniRanking/UniRanking.gpickle'
G = nx.read_gpickle(filename)
mst = nx.minimum_spanning_tree(G, data = True) 

# Uses Kruskal’s algorithm.
print(sorted(mst.edges(data=True)))

# minimum_spanning_edges(G, weight='weight', data=True)

nx.draw(G, markers="+")

nx.draw_networkx(G)

nx.draw_networkx(mst)
nx.draw_shell(mst)


G = nx.Graph()
G.add_nodes_from((1,2))
list(G)
nx.draw(G)

G = nx.Graph()
e = (2, 3)
G.add_edge(*e)
nx.draw(G)

print(mst)

pos=nx.spring_layout(G)

plt.figure()
nx.draw(mst, with_labels=False, pos = pos, node_size = 15, dim = 3)
plt.show()

di_g=nx.DiGraph(g)
mst = nx.minimum_spanning_tree(di_g) 

nx.connected_components(G)

nx.draw_random(mst)

import pygraphviz as pgv
import pydot
A = pgv.write_dot(G)

A.layout()
A.draw('simple.png')

mse = nx.minimum_spanning_edges(G, algorithm='prim')

nx.draw_networkx(mse)

edge_list = list(mse)
nx.draw(edge_list)

list(G)

y = nx.complete_graph(mst)
nx.draw_networkx(mst)


# prim, boruvka or kruskal

'''

