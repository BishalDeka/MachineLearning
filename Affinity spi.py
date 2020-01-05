import numpy as np
import pandas as pd

#Creating a df for the data
df = pd.DataFrame()
df["participant"] = ["Alice","Bob","Cary","Doug","Edna"]
df["tax rate"] = [3,4,3,2,1]
df["fee"] = [4,3,5,1,1]
df["interest rate"] = [3,5,3,3,3]
df["quantity limit"] = [2,1,3,3,2]
df["price limit"] = [1,1,3,2,3]

#Converting df to a matrix
fmat = df.iloc[:,1:].values


#Defining differnt matrices
s_mat = np.zeros(fmat.shape) #Similarity Matrix
r_mat = np.zeros(fmat.shape) #Responsibility Matrix
a_mat = np.zeros(fmat.shape) #Availability Matrix
c_mat = np.zeros(fmat.shape) #Criterion Matrix


#Finding similirarity matrix
for i in range(fmat.shape[0]):
    for k in range(fmat.shape[1]):
        s_mat[i,k]=-sum((fmat[i,:]-fmat[k,:])**2)
np.fill_diagonal(s_mat,np.amin(s_mat))

#Defining responsibility matrix
def r_dash(a):
    for i in range(r_mat.shape[0]):
        for k in range(r_mat.shape[1]):
            r_mat[i,k]=s_mat[i,k]-max(np.delete(a[i,:]+s_mat[i,:],k))
    return(r_mat)

#Defining availability matrix
def a_dash(r):
    for i in range(a_mat.shape[0]):
        for k in range(a_mat.shape[1]):
            if i==k:
                colm=np.delete(r[:,k],k)
                a_mat[i,k]=sum(colm[colm>0])
            else:
                colm=np.delete(r[:,k],[i,k])
                a_mat[i,k]=min(0,r[k,k]+sum(colm[colm>0]))
    return(a_mat)

#iterations for r_mat and a_mat
d=0    #Damping value
t=2     #Number of iterations
for i in range(t):
    if i==0:
        r_mat=r_dash(a_mat)
        a_mat=a_dash(r_mat)
    else:
        r_mat=d*r_mat+(1-d)*r_dash(a_mat)
        a_mat=d*a_mat+(1-d)*a_dash(r_mat)

#Critical Matrix after t iterations 
c_mat=r_mat+a_mat
c_mat


clusters=[]    #initiatin the list of clusters
for i in range(c_mat.shape[0]):
    c=int(np.arange(c_mat.shape[0])[c_mat[i,:]==max(c_mat[i,:])][0])
    clusters.append(c)
print('The formed clusters are:\n',clusters)

