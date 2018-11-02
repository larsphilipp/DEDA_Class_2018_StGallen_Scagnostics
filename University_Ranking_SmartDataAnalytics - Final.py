
## Title:           Scagnostics of University Rankings
## Subtitle:        Analysis of Financial Times Rankings for the Master in Management
##
## Course:          Smart Data Analysis
## M.A. Students:   Peter De Cour, Lars Stauffenegger, Davide Furlan, Angie Hoang
## Lecturer:        Professor Wolfgang Haerdle (HU Berlin)
## Place, Time:     University of St. Gallen, 29.10.18 - 2.11.18 

#-------------------------------------------------------------

## Packages used
import pandas as pd
import numpy as np
import time 
import datetime
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt
import os
import seaborn as sns
import networkx as nx
import requests
from bs4 import BeautifulSoup
import pandas as pd




# data scrape
data = pd.read_csv('/Users/PeterlaCour/documents/MIQEF/Smart Data Analytics/masters-in-management-2018.csv',encoding='ISO-8859-1', delimiter = ";")




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


#df = output_dict["2018"]
df = data
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
column_names = df_clean.columns

# scatter-plots with ranking on y axis
for i in column_names:
    print("\n","\n",i)
    p = sns.lmplot(y='2018',x=i,data = df_clean)
    #p.show()
    p.savefig('/Users/PeterlaCour/documents/MIQEF/SDA_UniRanking/Scatter Plots/With Reg Line'+i)


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

convex_hull


indep_var = "Career progress rank"

def MST_plot(P,plot_path,indep_var):
    X = squareform(pdist(P))
    edge_list = minimum_spanning_tree(X)
    
    plt.scatter(P[:, 0], P[:, 1])
    for edge in edge_list:
        i, j = edge
        plt.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], c='r')
    
    plt.ylabel(indep_var)
    ax = plt.gca()
    ax.set_aspect('equal')
    
    # filepath for plot image
    figure_path = plot_path + indep_var.replace("/", "-") + '.png'
    plt.savefig(figure_path)
    
    plt.show()
   
P = np.array(df_clean[[year,indep_var]])
edge_list = delaunay_triangulation.convex_hull
plt.scatter(P[:, 0], P[:, 1])
for edge in edge_list:
    i, j = edge
    plt.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], c='r')
    
    plt.ylabel(indep_var)
    ax = plt.gca()
    ax.set_aspect('equal')

# MSTS

## Definition of Functions
# Found Code on:
# http://peekaboo-vision.blogspot.com/2012/02/simplistic-minimum-spanning-tree-in.html
# Author: Andreas Mueller
# Slightly Modified for this Project by Peter and Lars

df   = pd.DataFrame(data)

# Drop Columns with Text and NAN
df_num   = df.select_dtypes(['number'])
df_clean = df_num.dropna(axis=1)

# drop year col -does not work
year = "2018"



def minimum_spanning_tree(X, copy_X=True):
    """X are edge weights of fully connected graph"""
    if copy_X:
        X = X.copy()
 
    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []
     
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf
     
    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices                                                      
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                      
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf                                                                     
        num_visited += 1
    return np.vstack(spanning_edges)

   
def MST_plot(P,plot_path,indep_var):
    X = squareform(pdist(P))
    edge_list = minimum_spanning_tree(X)
    
    plt.scatter(P[:, 0], P[:, 1])
    for edge in edge_list:
        i, j = edge
        plt.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], c='r')
    
    plt.ylabel(indep_var)
    ax = plt.gca()
    ax.set_aspect('equal')
    
    # filepath for plot image
    figure_path = plot_path + indep_var.replace("/", "-") + '.png'
    plt.savefig(figure_path)
    
    plt.show()
   
#-------------------------------------------------------------

## Minimal Spanning Tree Plots for our Data Sets
size = df_clean.shape   # size of dataframe
nr_col = size[1]           # nr of columns

for c in range (1,nr_col):
    # Col Name
    indep_var = df_clean.columns[c]
    
    # Normalise Idnependent Variable between 0 and 100
    df_clean[indep_var] = (df_clean[indep_var] - df_clean[indep_var].min()) / \
                          (df_clean[indep_var].max() - df_clean[indep_var].min()) \
                          * 100
    
    # Prepare Dependendt and Independent Variable in 2 Column Vector
    M = np.array(df_clean[[year,indep_var]])

    # Run MST Plot
    MST_plot(M,plot_path,indep_var)



f.pivot(index='foo', columns='bar', values='baz')

ranking = pd.read_csv('/Users/PeterlaCour/documents/MIQEF/SDA_UniRanking/Scagnostics for Ranking.csv',encoding='ISO-8859-1', delimiter = ";")

ranking_pivot = ranking.pivot(index = "Name")

cols = list(ranking.columns)

ranking.pivot("Name",columns = cols[1:-1])


'''
points = np.array(df_clean[[year,indep_var]])

def concave(points,alpha_x=100,alpha_y=100):
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
    
    ret = []
    for graph in nx.connected_component_subgraphs(G):
        ch = ConvexHull(graph.nodes())
        tmp = []
        for i in ch.simplices:
           
            tmp.append(list(graph.nodes())[i[0]])
            #print("success")
            tmp.append(list(graph.nodes())[i[1]])
            
        ret.append(tmp)
    #return ret  
    #return [graph.nodes() for graph in nx.connected_component_subgraphs(G)] - all points inside the shape
    return ret

p = concave(tuple_array)

p[0]
edge_list = p[0]
plt.scatter(P[:, 0], P[:, 1])
for edge in edge_list:
    i, j = edge
    plt.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], c='r')
    
    plt.ylabel(indep_var)
    ax = plt.gca()
    ax.set_aspect('equal')

'''





list(p)

nx.draw(p)

y_data = np.asarray(df_number_cleaned["2018"])
x_data = np.asarray(df_number_cleaned["Career progress rank"])
counts = len(y_data)


            
#plt.hexbin(x_data, y_data, gridsize=15, cmap = "viridis")           







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


alph = alpha_value
        
        
        