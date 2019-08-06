import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def sigmoid(X, theta):
    temp = np.dot(X,theta)
    temp = np.exp(-temp)
    temp = temp + 1
    temp = 1/temp
    return temp

def costfunc(h, y, theta, lam):
    theta = theta[1:,:]
    m = h.shape[0]
    y1 = y.transpose()
    h1 = np.log(h)
    p1 = np.dot(y1,h1)
    h2 = np.log(1-h)
    p2 = np.dot((1- y).transpose(), h2)
    p3 = sum(lam*(theta**2))/(2*m)
    ans = -(p1 +p2)/m + (p3)
    return ans

def gradDescLogisticRegularised(X, y, alpha, niter):
    theta = np.zeros(shape= (X.shape[1],1))
    h = sigmoid(X, theta)
    m = h.shape[0]
    Jhist = np.zeros(shape = (niter,1))
    
    for i in range(0, niter):
        Jhist[i] = costfunc(h,y,theta,0.0001*m/alpha)
        temp = theta
        temp = temp*(0.999) -((alpha*np.dot(X.transpose(), h-y))/m)
        temp[0] = theta[0] - ((alpha*np.dot(X.transpose(), h-y))/m)[0]
        theta = temp
        h = sigmoid(X, theta)
        
    return (theta, Jhist)



#list = [1,1,1,2,1,3,1,4]
#X = np.array(list).reshape(4,2)
#y = np.array([0,1,1,1]).reshape(4,1)
#(theta, J) = gradDescLogisticRegularised(X, y, 1, 10000)
#plt.plot([i for i in range(0,10000)], J)
#pred = sigmoid(X, theta)
#pred = pred.reshape(1,4)
#y = y.reshape(1,4)
#plt.scatter(X[:,1].reshape(1,4), y, color = 'blue')
#plt.scatter(X[:,1].reshape(1,4), y, color= 'red' )


