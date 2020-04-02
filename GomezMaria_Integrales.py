
# coding: utf-8

# In[1]:

import numpy as np
from scipy. integrate import quad
import matplotlib.pyplot as plt


# In[4]:

#definimos los metodos de integracion
def trapezio(func, x, puntos):
    #limite superior
    a=x[-1]
    #limite inferior 
    b=x[0]
    h=(a-b)/puntos
    inte=(h*2)*(func(a)+func(b))
    for i in range(len(x)):
        inte+=(h)*(f(x[i]))
    return inte


#func es la función que se quiere integrar, inf es el límite inferior y sup es el límite superior  
x=np.linspace(-np.pi/2, np.pi, 1001)

def simpson(func, x, puntos):
    inf=x[0]
    sup=x[-1]
    h=abs(sup-inf)/puntos
    inte=(h/3)*func(inf)+func(sup)
    for i in range(1, len(x)):
        inte +=(h/6)*(f(x[i])+ 4*f((x[i-1]+x[i])/3)+f(x[i]))
        
    return inte

    
       
def monte_carlo(func,x, puntos):
    min_x=x[0]
    max_x=x[-1]
    y_min=min(f(x))
    y_max=max(f(x))
    
    random_x = np.random.rand(puntos) * (max_x - min_x) + min_x
    
    random_y_pos = np.random.rand(puntos)*(y_max)
    random_y_neg = np.random.rand(puntos)*(y_min)
    
    
    delta1 = (abs(f(random_x))+f(random_x))/2-(random_y_pos)
    below1= np.where(delta1>0.0)
    
    delta2=(f(random_x)-abs(f(random_x)))/2-random_y_neg
    below2=np.where(delta2<0.0)

    puntos_Adentro=np.size(below1)+np.size(below2)
    puntos_totales=(np.size(random_y_neg))
    intervalo=np.pi/2
    integral=intervalo*(puntos_Adentro/(1.0*puntos_totales))
    return integral


    
def valor_medio(func, x,  puntos):
    inf = x[0]
    sup= x[-1]
   
    x = np.random.random(1000000) * (sup - inf) + inf
    y = func(x)

    integral = np.average(y) * (sup - inf)

    return  integral

def integrar(func, x , puntos, metodo=""): 
     
    if metodo=="T":
        return trapezio(func, x , puntos)
    
    elif metodo =="S":
        return simpson(func, x , puntos)
    
    elif metodo=="MC":
        return monte_carlo(func, x , puntos)
    
    elif metodo=="VM":
        return valor_medio(func, x , puntos)



# In[5]:

#Primera parte 
x=np.linspace(-np.pi/2, np.pi, 1001)
h=10001
def f(x):
    return np.cos(x)

def analitica(funcion, x ):
    inf=x[0]
    sup=x[-1]

    inte=quad(funcion,inf,sup)
    return inte



def error(funcion,x,h,metodo=""):
    error=0
    if metodo=="T":
        error=abs((integrar(funcion,x ,h, metodo="T") - 1)/1)
    elif metodo =="S":
        error=abs((integrar(funcion,x ,h, metodo="S") - 1)/1)
    elif metodo=="MC":
        error=abs((integrar(funcion,x,h, metodo="MC") - 1)/1)
    elif metodo=="VM":
        error=abs((integrar(funcion,x,h, metodo="VM") - 1)/1) 
    return error
    
print ('Con el metodo de Trapecio el valor de la integral es:', integrar(f, x, 1001, metodo="T"), 'y el error', error(f,x,1001,"T"))
print ('Con el metodo de Simpson el valor de la integral es:', integrar(f, x, 1001, metodo="S"), 'y el error', error(f,x,1001,"S"))
print ('Con el metodo de Monte Carlo el valor de la integral es:', integrar(f,x , 1001, metodo="MC"), 'y el error', error(f,x,1001,"MC"))
print ('Con el metodo de Valor Medio  el valor de la integral es:', integrar(f, x, 1001, metodo="VM"), 'y el error', error(f,x,1001,"VM"))




plt.plot()


# In[4]:

#parte b
n=[]
x1=1+np.logspace(2, 7, 6)

error_T=[]
error_S=[]
error_Mc=[]
error_Vm=[]

for i in range(len(x1)):
    x2=np.linspace(-np.pi/2, np.pi, x1[i])
    n.append(x1[i])
    error_T.append(abs((integrar(f,x2 ,x1[i], metodo="T") - 1)/1))
    error_S.append(abs((integrar(f,x2 ,x1[i], metodo="S") - 1)/1))
    error_Vm.append(abs((integrar(f,x2 ,x1[i], metodo="VM") - 1)/1))
    
    

plt.plot(n,error_T,label="T")
plt.plot(n,error_S, label="S")
plt.plot(n,error_Vm, label="V.M")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.show()



# In[3]:

#Punto 3-c, para ese punto cambia la función pero se aplican los métodos iguales 
x_1=np.linspace(0,1,1001)
def func(x):
    f=1/np.sqrt(np.sin(x))
    return f
def analitica_func(funcion, x):
    inf=x[0]
    sup=x[-1]

    inte=quad(funcion,inf,sup)
    return inte[0]
analitica_func(func, np.linspace(0,1))



# In[6]:

def error(funcion,x,h,metodo=""):
    error=0
    if metodo=="T":
        error=abs((integrar(funcion,x ,h, metodo="T") - analitica(funcion, x))/analitica(funcion, x))
    elif metodo =="S":
        error=abs((integrar(funcion,x ,h, metodo="S") - analitica(funcion, x))/analitica(funcion, x))
    elif metodo=="MC":
        error=abs((integrar(funcion,x,h, metodo="MC") - 2.0348053192075737)/2.0348053192075737)
    elif metodo=="VM":
        error=abs((integrar(funcion,x,h, metodo="VM") - 2.0348053192075737)/2.0348053192075737)
    return error
    
print ('Con el metodo de Trapecio el valor de la integral es:', integrar(func, x_1, 1025, metodo="T"), 'y el error', error(func,x_1,1025,"T"))
print ('Con el metodo de Simpson el valor de la integral es:', integrar(func, x_1, 1025, metodo="S"), 'y el error', error(func,x_1,1025,"S"))
print ('Con el metodo de Monte Carlo el valor de la integral es:', integrar(func,x_1 , 1025, metodo="MC"), 'y el error', error(func,x_1,1025,"MC"))
print ('Con el metodo de Valor Medio  el valor de la integral es:', integrar(func, x_1, 1025, metodo="VM"), 'y el error', error(func,x_1,1025,"VM"))




# In[ ]:

#arreglando la singularidad quitando el infinito de los límites 
        
def simpson_arreglado(func, x, puntos):
    inf=x[0]
    sup=x[-1]
    if func(inf)>(10**6):
        print("entre")
        h=abs(sup-inf)/puntos
        inte=(h/3)*(10**6+func(sup))
    elif func(inf)>(10**6):
        h=abs(x[-1]-inf)/puntos
        inte=(h/3)*(func(inf)+func(sup))
    for i in range(1, len(x)):
        if f(x[i])>10**6:
            inte+=(h/6)*(10**6+ (4*10**6/3)+10**6)
        elif f(x[i])<10**6:
            inte+=(h/6)*(f(x[i])+ 4*f((x[i-1]+x[i])/3)+f(x[i]))
    return inte
    
        
    
print(simpson_arreglado(func,x_1,1001))
#solucion de la singularidad 
#print ('Con el metodo de Trapecio el valor de la integral es:', integrar(func, x1, 1001, metodo="T"), 'y el error', error(func,x1,1001,"T"))
#print ('Con el metodo de Simpson el valor de la integral es:', simpson_arreglado(func,x_1,1025), 'y el error', error(func,x_1,1025,"S"))
#print ('Con el metodo de Monte Carlo el valor de la integral es:', integrar(func,x1 , 1001, metodo="MC"), 'y el error', error(func,x1,1001,"MC"))
#print ('Con el metodo de Valor Medio  el valor de la integral es:', integrar(func, x1, 1001, metodo="VM"), 'y el error', error(func,x1,1001,"VM"))




# In[ ]:

#arreglando el infinito eliminando el cero con el metodo original sin arreglar el infinito 
x1=np.linspace(0.000001,1)
print ('Con el metodo de Simpson el valor de la integral es:', integrar(func, x1, 1025, metodo="S"), 'y el error', error(func,x1,1025,"S")[1])


# In[6]:

#arreglando con la resta de funciones 
def funcion(x):
    a=1/np.sqrt(x)
    b=1/np.sqrt(x)
    return (a-b)
def resta_funcion(x):
    f=1/(np.sqrt(x))
    return f
def analitica_resta_funcion(func, x):
    inf=x[0]
    sup=x[-1]
    inte=quad(func,inf,sup)
    return inte[0]
analitica= analitica_resta_funcion(resta_funcion, x_1)

print ('Restando la singularidad el resultado es:', analitica,)
print ('El metodo que mejor funciono fue el de restar la singularidad' )


# In[ ]:




# In[ ]:



