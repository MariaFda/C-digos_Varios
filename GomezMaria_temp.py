
# coding: utf-8

# In[23]:

import numpy as np
import matplotlib.pyplot as plt
import itertools


# In[28]:

#Se lee el archivo de los datos
dat=np.loadtxt('647_Global_Temperature_Data_File.txt')
#se separan los datos por las columnas mencionadas
año=dat[:,0]
anomalia=dat[:,1]
a_suave=dat[:,2]
#se grafica el año en funcion de las anomalias
plt.xlabel('Año')
plt.ylabel('Anomalia')
plt.plot(año, anomalia)
plt.savefig('grafica_temp.pdf')
plt.close()


#Punto d encontrar los años con diferencias mayores a 0.5
arreglo=[]
def anomalia_mayores(anomalia):
    
    for i in range(137):
        if anomalia[i]>0.5:
            arreglo.append(año[i])
            i=i+1
    return (arreglo)
    
    
anomalia_mayores(anomalia)


# In[25]:

def a_calientes(año, anomalia, a_suave):
    calientes=[]
    n=len(anomalia)
    for i in range(n):
        nueva_anomalia=sorted(dat[:,1])
        nueva_anomalia.reverse()
        nueva_anomalia[:20]
        return nueva_anomalia[:20]
print (a_calientes(año, anomalia, a_suave) )  


# In[27]:



#punto f
plt.xlabel('Año')
plt.ylabel('Anomalia suavizada')
plt.plot(año, a_suave)
plt.savefig('grafica_temp_2.pdf')
plt.close()


# In[ ]:



