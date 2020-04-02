
# coding: utf-8

# In[26]:

import numpy as np
import matplotlib.pyplot as plt


# In[33]:

#N=np.random.randint(3 , 8)
N = 4
Nt = N + 1
Arreglo=(np.random.random((N,N))*10.0)-5.0
B=(np.random.random((N,1))*10.0)-5.0
Mat = np.ones((N,N+1))
Mat[:,0:-1] = Arreglo[:,:]
Mat[:,N] = B[:,0]
Mat


# In[34]:

for k in range(0,N-1):
    #verifando la condicion de los ceros en la diagonal
    needToRoll = False
    if(Mat[k,k] == 0.0):
        needToRoll = True
        
    while(needToRoll):
        Mat = np.roll(Mat,1,0)
        needToRoll = Mat[k,k] == 0
        
    A = 1/Mat[k,k]
    for i in range(k+1,N):
        C = A*Mat[i,k]
        Mat[i,k:Nt-1] = Mat[i,k:Nt-1] - C*Mat[k,k:Nt-1]
print (Mat)


# In[44]:

def matriz_one():
    Matt=np.copy(Mat)
    for i in range(len(Matt[:,0])):  
        Matt[i,:]= Matt[i,:]/Matt[i,i]
    return (Mat1)
print(matriz_one())


# In[45]:

Positions = np.arange(0,N)
Positions = Positions[::-1]
X = np.ones((N,1))
for k in Positions:
    X[k] = Mat[k,N]
    for j in range(k+1,N):
        X[k] = X[k] - Mat[k,j]*X[j]
    X[k] = X[k]/Mat[k,k]   


# In[46]:

print('La solución es:',B,)


# In[47]:

print('La verificación de la solución es:',Mat[:,0:-1].dot(X),) #verifanco la solucion 


# In[25]:

#sol= np.linalg.solve(Mat[:,0:4],B) Hace la solución para el B ya resuelto 
#print(sol)


# In[ ]:




# In[ ]:



