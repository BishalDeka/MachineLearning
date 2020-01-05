import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=20,n_features=2,centers=2,random_state=19) 
# X is the data and y are cluster labels

#Defining distance metic
def dist(p1,q1):
    sum_sq=sum((p1-q1)**2)
    return(sum_sq**(1/2))

#Creating data frame to update values
col=['points','cluster','reach_dist','core_dist']
idx=range(len(X))
df=pd.DataFrame(columns=col,index=idx)
df.iloc[:,0]=list(X)
df.iloc[:,1]=0
df.iloc[:,2]=1000
df.iloc[:,3]=1000

#Passing the parameter values
e=10
mnpt=3

#Running the algorithm
pr_pts=[]  #List to store processed points
clstr=0
for i in range(len(X)):
    if list(X[i]) not in pr_pts:
        pr_pts.append(list(X[i]))  #Updating processed points list   
        nbd=[(h,dist(X[i],X[h])) for h in range(len(X)) if dist(X[i],X[h])<e and h!=i] #Finding neighbourhood
        nbd.sort(key=lambda x: x[1])  #Sorting nbd wrt distance from core point
        if len(nbd)>=mnpt:
            clstr=clstr+1    
            df.iloc[i,1]=clstr    #Assigning cluster to X[i]
            core_dist=nbd[mnpt-1][1]
            df.iloc[i,3]=core_dist   #updating core distance of X[i]
            for j in range(len(nbd)):   #Proceeding to do operations on neighbours
                df.iloc[nbd[j][0],2]=min(df.iloc[nbd[j][0],2],max(core_dist,dist(X[i],X[nbd[j][0]]))) #updating reach dist of neighbours
                if list(X[nbd[j][0]]) not in pr_pts:
                    pr_pts.append(list(X[nbd[j][0]])) #Updating processed points list
                    nbd1=[(h,dist(X[nbd[j][0]],X[h])) for h in range(len(X)) if dist(X[nbd[j][0]],X[h])<e and h!=nbd[j][0]] #Finding nbd
                    nbd1.sort(key=lambda x: x[1]) #Sorting neighbours of the neighbours of X[i]
                    if len(nbd1)>=mnpt:
                        df.iloc[nbd[j][0],1]=clstr  #Assigning clusters to the neighbours of X[i]
                        core_dis_neigh=nbd1[mnpt-1][1]  
                        df.iloc[nbd[j][0],3]=core_dis_neigh  #Updating core distance of neighbours
                        for k in range(len(nbd1)):   #Loop to update reachability distance of neighbours of the neighbours of X[i]
                            df.iloc[nbd1[k][0],2]=min(df.iloc[nbd1[k][0],2],max(core_dis_neigh,dist(X[nbd[j][0]],X[nbd1[k][0]])))
                        
print('The updated data frame is:\n',df)