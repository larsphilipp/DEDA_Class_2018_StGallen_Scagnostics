import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

## Year of the Ranking
year = "2018"

#-------------------------------------------------------------


#-------------------------------------------------------------

## Data Import and Cleansing
# WD
path = os.path.dirname(os.path.realpath("__file__")) + '\\'
plot_path = path + '\\MST_Plots\\'
ols_path = path + '\\OLS_Summaries\\'
file ='house.csv'

data = pd.read_csv(path + file,encoding='ISO-8859-1', delimiter = ",")
df_boston   = pd.DataFrame(data).rename(columns={year: Ranking_Col})




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
    # Normalise Independent Variable between 0 and 100 if not rank or percentage
#    if 'rank' in indep_var:
#        print(indep_var)
#    elif '%' in indep_var:
#        print(indep_var)
#    else:
#        X_min = np.amin(P[:,0])
#        X_max = np.amax(P[:,0])
#        P[:,0] = (P[:,0] - X_min) / (X_max - X_min) * 100
#        indep_var = 'normalised ' + indep_var
#        print(indep_var)
    
    # MST Edges
    X = squareform(pdist(P))
    edge_list = minimum_spanning_tree(X)
    
    # MST scatter plot
    plt.scatter(P[:, 0], P[:, 1])
    for edge in edge_list:
        i, j = edge
        plt.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], c='r')
    
    plt.xlabel(indep_var)
    ax = plt.gca()
    ax.set_aspect('equal')
    
    # filepath for plot image
    figure_path = plot_path + indep_var.replace("/", "-") + '.png'
    plt.savefig(figure_path)
    
    plt.show()
        
        
indep_var = 'dis'
dep_var = 'medv'
      
# Prepare Dependendt and Independent Variable in 2 Column Vector
M = np.array(df_boston[[indep_var,dep_var]])

# Run MST Plot
MST_plot(M,plot_path,indep_var)

plt.plot(df_boston[[indep_var,dep_var]])
