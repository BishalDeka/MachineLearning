#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Creating the data
from sklearn.datasets import make_moons
X,y = make_moons(250,random_state=19,noise=0.01)

sig=0.1   #Scaling parameter
k=2  #Number of clusters

#Defining affinity matrix
A=np.zeros((len(X),len(X)))
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if i!=j:
            A[i,j]=np.exp((-sum((X[i]-X[j])**2))/(2*(sig)**2))

D=np.zeros((len(X),len(X)))
D_isq=np.zeros((len(X),len(X)))
for i in range(A.shape[0]):
        D[i,i]=sum(A[i,:])
        D_isq[i,i]=(D[i,i])**(-1/2)
        
L=D_isq.dot(A).dot(D_isq)
#To get the eigen vectors for L
eigvals,eigvecs = np.linalg.eig(L)
#Getting the first k eigen vectors
W=eigvecs[:,0:k]

#Transformed version of X
Y = np.zeros(W.shape)
for i in range(W.shape[0]):
    Y[i] = W[i]/(sum(W[i]**2)**0.5)

#creating the clusters using kmeans
km_sc= KMeans(n_clusters=k)
clust_sc = km_sc.fit_predict(Y)
plt.scatter(X[:,0],X[:,1],c=clust_sc)

#Checking the plot for Kmeans
km_ori = KMeans(n_clusters=k)
clust_km = km_ori.fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=clust_km)

