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
import matplotlib.pyplot as plt
import os

#-------------------------------------------------------------

# TO DO: Web Mining

#-------------------------------------------------------------

## Data Import and Cleansing
# WD
path = os.path.dirname(os.path.realpath("__file__"))
plot_path = path + '\\MST_Plots\\'
file ='/mmgmt18.csv'

# Import into Data Frame
data = pd.read_csv(path + '/mmgmt18.csv',encoding='ISO-8859-1', delimiter = ";")
df   = pd.DataFrame(data)

# Drop Columns with Text and NAN
df_num   = df.select_dtypes(['number'])
df_clean = df_num.dropna(axis=1)

# drop year col -does not work
year = "2018"
#df_indep = df_number.copy()
#df_indep.drop(year, axis=1)

#-------------------------------------------------------------

## All scatter plots with ranking on Y axis and independent variable on X
column_names = df_clean.columns
for i in column_names:
    print("\n","\n",i)
    p = sns.relplot(y='2018',x=i,data = df)

# Pair Plot
pairs = sns.pairplot(df_clean, diag_kind="kde", markers="+",plot_kws=dict(s=50, edgecolor="b", linewidth=1),diag_kws=dict(shade=True))

#-------------------------------------------------------------

## Definition of Functions
# Found Code on:
# http://peekaboo-vision.blogspot.com/2012/02/simplistic-minimum-spanning-tree-in.html
# Author: Andreas Mueller
# Slightly Modified for this Project by Peter and Lars

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
    

#-------------------------------------------------------------
    
# TO DO: OLS

#-------------------------------------------------------------


## TO FIX: savefig ##
cor_plot = plt.matshow(df.corr(),cmap = "RdYlGn")
plt.show()

correlation_table = df.corr()
c_table  = sns.heatmap(correlation_table, 
            xticklabels=correlation_table.columns.values,
            yticklabels=correlation_table.columns.values,
            cmap = "RdYlGn")

c_table.savefig(path)

#-------------------------------------------------------------

####################
##### not used #####
####################

### Euclidean Distances: Alternative to squareform(pdist(P)) 
### Numpy Matrix with dependent and independent variable
#M = np.array(df[["2018","Career progress rank"]])
#
#size = M.shape      # size of the array
#n = size[0]         # number of observations in M / dots in plot
#D = np.zeros((n,n)) # assign space for Euclidean Distances Matrix
#
### Create Euclidean Distances Matrix
#for p in range(0,n):
#    x1 = M[p,1]
#    y1 = M[p,0]
#    for q in range(0,n):
#        x2 = M[q,1]
#        y2 = M[q,0]
#        D[p,q] = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))





#-------------------------------------------------------------
