
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#simulation of 40 data points
np.random.seed(1)
X = np.r_[np.random.randn(20, 2) - [1.5, 1.5], 
          np.random.randn(20, 2) + [1.5, 1.5]]
Y = [0] * 20 + [1] * 20


def plot_svm_linear(X, Y, penalty):
    #fit the model
    fignum = 1
    clf = svm.SVC(kernel='linear', C=penalty)
    clf.fit(X, Y)

    #get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    #get the margin and the parallels
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    lower = yy - np.sqrt(1 + a ** 2) * margin
    upper = yy + np.sqrt(1 + a ** 2) * margin

    #plot the separating plane and the parallels
    plt.figure(fignum, figsize=(4, 4))
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, lower, 'k--')
    plt.plot(xx, upper, 'k--')

    #plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=5, cmap='bwr',
                edgecolors='k')
    #mark the support vectors
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=20, edgecolors='k')

    plt.axis('tight')
    plt.xlim(-4.5, 4.5)
    plt.ylim(-4.5, 4.5)
    
    fignum = fignum + 1
    
plot_svm_linear(X, Y, 1000)
plt.savefig('linear_1.png', transparent=True, dpi=200)
plt.clf()

plot_svm_linear(X, Y, 0.05)
plt.savefig('linear_0.05.png', transparent=True, dpi=200)
plt.clf()

#---------------------------------------------------

from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# Simulate data
X, y = make_moons(n_samples=100, noise=0.2)


def plot_svm_nonlinear(x, y, model_class, **model_params):
    #Fit model
    model = model_class(**model_params)
    model.fit(x, y)
    
    #Define grid
    h = .001     
    x_min, x_max = x[:, 0].min() - 0.2, x[:, 0].max() + 0.2
    y_min, y_max = x[:, 1].min() - 0.2, x[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    #Prediction on grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    #Contour + scatter plot
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.4, cmap='coolwarm')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.yticks(np.arange(-0.75, 1.75, step=0.5))

    return plt

plot_svm_nonlinear(X,y,svm.SVC,C=1,kernel='linear')
plt.savefig('nonlinear1.png', transparent=True, dpi=200)
plt.clf()

plot_svm_nonlinear(X,y,svm.SVC,C=100,kernel='poly',degree=3)
plt.savefig('nonlinear2.png', transparent=True, dpi=200)
plt.clf()

#---------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#Read data
spiral = np.loadtxt("C:/Users/Lars Stauffenegger/Documents/MBF Unisg/Smart Data Analytics/spiral.txt")
x, y = spiral[:, :2], spiral[:, 2]

def plot_svm_decision(X, y, model_class, **model_params):
    #Fit model
    model = model_class(**model_params)
    model.fit(X, y)
    
    #Define grid
    h = .001     
    x_min, x_max = x[:, 0].min() - 0.2, x[:, 0].max() + 0.2
    y_min, y_max = x[:, 1].min() - 0.2, x[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
                         np.arange(y_min, y_max, h))
    
    #Prediction on grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    #Contour + scatter plot
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.4, cmap='coolwarm')
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig('spiral.png', transparent=True, dpi=200)
    return plt

plot_svm_decision(x,y,svm.SVC,C=100,kernel='rbf',gamma=20)
plt.clf()

#---------------------------------------------------

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import svm

#Load data
swiss = genfromtxt('swiss.txt', delimiter='', skip_header=1)

Y = swiss[:,1]
X = swiss[:,[5,6]]

#Simple scatterplot
plt.scatter(X[:,0],X[:,1],c=Y,cmap='bwr',s=4)


def plot_svm_nonlinear(x, y, model_class, **model_params):
    #Fit model
    model = model_class(**model_params)
    model.fit(x, y)
    
    #Define grid
    h = .001     
    x_min, x_max = 7,13
    y_min, y_max = 7,13
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    #Prediction on grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    #Contour + scatter plot
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.4, cmap='coolwarm',s=4)
    plt.gca().set_aspect('equal', adjustable='box')

    return plt

plot_svm_nonlinear(X,Y,svm.SVC,C=10,kernel='poly',degree=2)
plt.savefig('swiss.png', transparent=True, dpi=200)
plt.clf()


