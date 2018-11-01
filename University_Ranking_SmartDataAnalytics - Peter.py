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


df = output_dict["2017"]

column_names = df.columns
column_names
# ----------------------------------------------------------------------------

# clean column names?

#df.to_csv(path_or_buf= '/Users/PeterlaCour/documents/MIQEF/SDA_UniRanking/2017 Scrape.csv')


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


import numpy as np

df_number_cleaned[["2018","Career progress rank"]] 

M = np.array(df_number_cleaned[["2018","Career progress rank"]])

M 

n = 100 # number of observations in M / dots in plot

D = np.zeros((n,n))

 

for p in range(0,n):

    x1 = M[p,1]

    y1 = M[p,0]

    print(x1)
    for q in range(0,n):

        x2 = M[q,1]

        y2 = M[q,0]

        D[p,q] = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

D[0]  
   

####### minimum spanning tree

import scipy.sparse as scs

from scipy.sparse.csgraph import minimum_spanning_tree

       

# n*n matrice with the (normalized?) edges that give the MST

MST = minimum_spanning_tree(D).toarray().astype(float)

MST

nx.draw_networkx(MST[0])
### Now we need to find a way how to disply only the edges of those pairs that have a non-zero value in this Matrix.

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


minimum_edge
# outlier has degree 1 and has an associated edge weight greater than w = q75 + 1.5(q75-q25)



# outliers = c = length(T_outliers)/length(T)
q25                     = np.quantile(minimum_edge,0.25)
q75                     = np.quantile(minimum_edge,0.75)
length_of_long_edges    = q75 - 1.5 * (q75 - q25)

# false: c_outliers              = length_of_long_edges / total_edge_length

# t-outliers is the lengths of outliers mst length not the weitghts

# convex = area(A)/area(H) - area of alpha hull / convex hull
 

   
#plt.hist(D[1]/100, bins = 100)


    
# skinny = 1 - sqrt(4pi * area(a))) / perimeter(a)




total_count = 100 # ????

# skewness: ratio of edge lengths in edge distribution: (q90 - q50) / (q90 - q10)
q90                 = np.quantile(minimum_edge,0.9)
q50                 = np.quantile(minimum_edge,0.5)
q10                 = np.quantile(minimum_edge,0.1)
skewness            = (q90 - q50) / (q90 - q10)
# !!!!!!!
t                   = total_count / 500 # total count of binned data
# !!!!!!!
correction          = 0.7 + 0.3 * (1 + t**2)

corrected_skewness  = 1 - correction * (1 - skewness) 
corrected_skewness
# need to get count of binned data

# clumpy = max(1-max(legnth(ek))/length(ej)) - ????????
    
    
    

# sparsity = min(1,q90)?

# min(90th quantile of mstlengths / 1000, 1)

sparsity = min(q90 / 1000, 1)
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

df_number_cleaned[["2018","Career progress rank"]]


y_data = np.asarray(df_number_cleaned["2018"])
x_data = np.asarray(df_number_cleaned["Career progress rank"])
counts = len(y_data)


def binhex(x,y,nBins):
    # x = list of xs
    # y = list of ys
    # nBins = number of bins
    
    
    # declarations
    count   = [0] * nBins
    xBin    = [0] * nBins
    yBin    = [0] * nBins
    
    
    
    n       = len(x)
    con1    = 0.25
    con2    = 1.0/3.0
    c1      = nBins -1
    c2      = c1 / np.sqrt(3.0)
    jinc    = nBins
    iinc    = 2 * nBins
    nBin    = (nBins + 20) * (nBins + 20)
    
    #count,xbin,ybin
    
    for i in range(n):
        if np.isnan(x[i]) or np.isnan(y[i]):
            continue
        sx      = c1 * x[i]
        sy      = c2 * y[i]
        i1      = sy + 0.5
        j1      = sx + 0.5
        
        dy      = sy - i1
        dx      = sx - j1
        
        dist1   = dx**2 + 3.0 * (dy)**2
        
        if dist1 < con1:
            m = i1 * iinc + j1
        elif dist1 > con2:
            m = sy * iinc + sx + jinc
        else:
            i2 = sy
            j2 = sx
            dist2 = dx**2 + 3 * dy**2
            
            if dist1 <= dist2:
                m = i1 * iinc + j1
            else:
                m = i2 * iinc + j2 + jinc
        m = int(m)     
        count[i] += 1
        xBin[i]  += (x[i] - xBin[i]) / count[i]
        yBin[i]  += (y[i] - yBin[i]) / count[i]    
            
        # delete Empty bins
        
            # need to add code?
        
        # ----------------
        

        
        
    tcount = copy.deepcopy(count)
    xtBin  = copy.deepcopy(xBin)
    ytBin  = copy.deepcopy(yBin)
        
    
          
            
binhex(x_data,y_data,1000)           
            
            
            
plt.hexbin(x_data, y_data, gridsize=15, cmap = "viridis")           
            
plt.scatter(x_data,y_data)        
        



private static void normalizePoints(double[][] points) {
        double[] min = new double[points.length];
        double[] max = new double[points.length];
        for (int i = 0; i < points.length; i++)
            for (int j = 0; j < points[0].length; j++) {
		if (j == 0)
		    max[i] = min[i] = points[i][0];
		else if (min[i] > points[i][j])
		    min[i] = points[i][j];
		else if (max[i] < points[i][j])
		    max[i] = points[i][j];
            }
        for (int i = 0; i < points.length; i++)
            for (int j = 0; j < points[0].length; j++)
                points[i][j] = (points[i][j] - min[i]) / (max[i] - min[i]);
    }
            
def normalise(points):
    min1 = [0.0,0.0] * len(points)
    max1 = [0.0,0.0] * len(points)
    for i in range(len(points)):
        for j in range(len(points)):
            if j == 0:
                max1[i] = min1[i] = points[i][0]
            elif min1[i] > points[i][j]


[x_data, y_data]




tuple_array = []

for i in range(len(df["2018"])):
    tuple_array.append((df["2018"][i],df["Career progress rank"][i]))

points = np.asarray(df[["Career progress rank","2018"]])

len(points)

#find_outliers

def find_cutoff(MST):
    q25                     = np.quantile(MST,0.25)
    q75                     = np.quantile(MST,0.75)
    cutoff                  = q75 - 1.5 * (q75 - q25)

    return cutoff

MST_sorted2 = me.sort()

def minimum_edge(tuple_array):
    min_edge = []
    for o in range(len(tuple_array)):
        edges_length = []
        m,n = tuple_array[o]
        
        for l in range(len(tuple_array)):
            i,j = tuple_array[l]
            edges_length.append(np.sqrt((m-i)**2+(n-j)**2))
        edges_length.pop(o)
        min_edge.append(min(edges_length))
    return min_edge

me = minimum_edge(tuple_array)

total_edge_length = sum(me)




school_names =  np.asarray(df["School name"])
me_school = pd.DataFrame({'ME':me, 'School name':school_names})
me_school.sort_values(by="ME")


cut = find_cutoff(me_school["ME"])
cut
outliers = []
outliers_schools = []
MST_outliers_length = 0

# find outliers
for i in range(len(points)):
    if me[i] < cut:
        outliers.append(y_data[i])
        outliers_schools.append(me_school["School name"][i])
        MST_outliers_length += me[i]
# c for outlier
outlier_c = MST_outliers_length /  total_edge_length 
outlier_c

outliers
outliers_schools

def alpha_value():
    q90 = np.quantile(me_school["ME"],0.90)
    alpha = q90
    return min(alpha,100)

alpha_value()


def computePearson(x,y,weights):
    n = len(x)
    xmean   = 0
    ymean   = 0
    xx      = 0
    yy      = 0
    xy      = 0
    sumwt   = 0
    for i in range(n):
        wt = weights[i]
        