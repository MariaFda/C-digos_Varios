
# coding: utf-8

# In[37]:

get_ipython().magic('pylab inline')
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import pyplot
import itertools 


# In[43]:

#interseccion de rectangulos con centros en (x1,y1) y (x2,y2), lados a1=b1, c1=d1, a2=b2, c2=d2

def rectangulos(x1,y1,x2,y2, l1, a1, l2, a2):
    #construccion de ñps rectangulos conociendo sus puntos centrales 
    #b1=a1 #ancho
    #l1=d1 #largo
    #b2=a2
    #l2=d2
    #d1=((a1/2)**2)+((c1/2)**2)
    #diag=np.sqrt((d1))
    #para el primer rectangulo
    esquina_inferior_iz_1=(x1-(l1/2),y1-(a1/2))
    esquina_inferior_d_1=(x1+(l1/2),y1-(a1/2))
    esquina_superior_iz_1=(x1-(l1/2),y1+(a1/2))
    esquina_superior_d_1=(x1+(l1/2),y1+(a1/2))
    #para el segundo rectangulo
    esquina_inferior_iz_2=(x2-(l2/2),y2-(a2/2))
    esquina_inferior_d_2=(x2+(l2/2),y2-(a2/2))
    esquina_superior_iz_2=(x2-(l2/2),y2+(a2/2))
    esquina_superior_d_2=(x2+(l2/2),y2+(a2/2))
    #contruyendo el primer rectangulo 
    plt.vlines(x1-(l1/2), y1-(a1/2), y1+(a1/2))
    plt.vlines(x1+(l1/2), y1-(a1/2), y1+(a1/2))
    plt.hlines(y1-(a1/2),x1-(l1/2), x1+(l1/2) )
    plt.hlines(y1+(a1/2),x1-(l1/2), x1+(l1/2) )
    #conttuyendo el segundo rectangulo 
    plt.vlines(x2-(l2/2), y2-(a2/2), y2+(a2/2))
    plt.vlines(x2+(l2/2), y2-(a2/2), y2+(a2/2))
    plt.hlines(y2-(a2/2),x2-(l2/2), x2+(l2/2) )
    plt.hlines(y2+(a2/2),x2-(l2/2), x2+(l2/2) )
    return(x1,y1,x2,y2, l1, a1, l2, a2)
def intersecciones(line_a, line_b):
    xdiff = (line_a[0][0] - line_a[1][0], line_b[0][0] - line_b[1][0])
    ydiff = (line_a[0][1] - line_a[1][1], line_b[0][1] - line_b[1][1]) 

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return  ('No se intersectan')

    d = (det(*line_a), det(*line_b))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


    
x1,y1,x2,y2, l1, a1, l2, a2 =rectangulos(2,2,3,1, 2, 2, 1, 1) 


    


# In[89]:

#intersecciones posibles 
if intersecciones((((x1-(l1/2),y1-(a1/2))), (x1+(l1/2),y1-(a1/2))), ((x2-(l2/2),y2-(a2/2)), (x2+(l2/2),y2-(a2/2))))!= 'No se intersectan':
    print (intersecciones(((x1-(l1/2), y1-(a1/2)), (x1+(l1/2), y1-(a1/2))), ((x2-(l2/2), y2-(a2/2)), (x2+(l2/2), y2-(a2/2)))))

if intersecciones(((x1-(l1/2),y1-(a1/2)), (x1+(l1/2), y1-(a1/2))), ((x2-(l2/2),y2+(a2/2)), (x2+(l2/2), y2+(a2/2))))!= 'No se intersectan':
    print (intersecciones(((x1-(l1/2), y1-(a1/2)), (x1+(l1/2), y1-(a1/2))), ((x2-(l2/2), y2+(a2/2)), (x2+(l2/2), y2+(a2/2)))))

if intersecciones(((x1-(l1/2), y1-(a1/2)), (x1+(l1/2), y1-(a1/2))), ((x2-(l2/2),y2-(a2/2)), (x2-(l2/2),y2+(a2/2))))!= 'No se intersectan':
    print (intersecciones(((x1-(l1/2), y1-(a1/2)), (x1+(l1/2), y1-(a1/2))), ((x2-(l2/2), y2-(a2/2)), (x2-(l2/2), y2+(a2/2)))))    

if intersecciones(((x1-(l1/2), y1-(a1/2)), (x1+(l1/2), y1-(a1/2))), ((x2+(l2/2),y2-(a2/2)), (x2+(l2/2),y2+(a2/2))))!= 'No se intersectan':
    print (intersecciones(((x1-(l1/2), y1-(a1/2)), (x1+(l1/2), y1-(a1/2))), ((x2+(l2/2), y2-(a2/2)), (x2+(l2/2), y2+(a2/2)))))

if intersecciones(((x1-(l1/2),y1-(a1/2)), (x1-(l1/2),y1+(a1/2))), ((x2-(l2/2),y2-(a2/2)), (x2-(l2/2),y2+(a2/2))))!= 'No se intersectan':
    print (intersecciones(((x1-(l1/2), y1-(a1/2)), (x1-(l1/2), y1+(a1/2))), ((x2-(l2/2), y2-(a2/2)), (x2-(l2/2), y2+(a2/2)))))

if intersecciones(((x1+(l1/2),y1-(a1/2)), (x1+(l1/2),y1+(a1/2))), ((x2+(l2/2),y2-(a2/2)), (x2+(l2/2),y2+(a2/2))))!= 'No se intersectan':
    print (intersecciones(((x1+(l1/2), y1-(a1/2)), (x1+(l1/2), y1+(a1/2))), ((x2+(l2/2), y2-(a2/2)), (x2+(l2/2), y2+(a2/2)))))    

if intersecciones(((x2-(l2/2),y2+(a2/2)), (x2+(l2/2),y2+(a2/2))), ((x1+(l1/2),y1-(a1/2)), (x1+(l1/2),y1+(a1/2))))!= 'No se intersectan':
    print (intersecciones(((x2-(l2/2), y2+(a2/2)), (x2+(l2/2), y2+(a2/2))), ((x1+(l1/2), y1-(a1/2)), (x1+(l1/2), y1+(a1/2)))))

if intersecciones(((x1-(l1/2),y1+(a1/2)), (x1+(l1/2),y1+(a1/2))), ((x2+(l2/2),y2-(a2/2)), (x2+(l2/2),y2+(a2/2))))!= 'No se intersectan':
    print (intersecciones(((x1-(l1/2), y1+(a1/2)), (x1+(l1/2), y1+(a1/2))), ((x2+(l2/2), y2-(a2/2)), (x2+(l2/2), y2+(a2/2)))))



# In[73]:




# In[ ]:



