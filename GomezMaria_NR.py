
# coding: utf-8

# In[13]:

import numpy as np
import matplotlib.pyplot as plt


# In[16]:

x=np.linspace(-4,4)
def f(x):
    p= (x**5) - 2*x**4 - 10*x**3 + 20*x**2 + 9*x -18.
    return p

plt.plot(f(x))
plt.savefig('ApellidoNombre_NRpoli')
plt.close()
#plt.show()


# In[17]:

def df(x):
    der= 5*x**4 - 8*x**3 - 30*x**2 + 40*x + 9
    return der
 
def newton(f, df, x, e):
    err = f(x)
    xR = x
    N = 0
    while(err > e):
        f_g = f(xR)
        df_g = df(xR)
        xR = xR - (f_g/df_g)
        err = f(xR)
        N = N + 1
    return [xR, N]


# In[18]:

newton(f,df,4,10**(-10))


# In[19]:

losGuess = np.random.uniform(-4,4,1000)
lasIter = []
lasRoot = []


# In[20]:

for i in losGuess:
    res = newton(f,df,i,10**(-10))
    lasIter.append(res[1])
    lasRoot.append(res[0])


# In[23]:

plt.scatter(losGuess,lasIter,s = 4)
plt.xlabel('Initial Guess')
plt.ylabel('Num Iteraciones')
plt.savefig('ApellidoNombre_NR_itera')
plt.close()
#plt.show()


# In[24]:

plt.scatter(losGuess,lasRoot,s = 4)
plt.xlabel('Initial Guess')
plt.ylabel('Root')
plt.ylim([-10,10])
plt.savefig('ApellidoNombre_NRxguess')
plt.close()
#plt.show()


# In[27]:

print('Los puntos problema se dan cuando resulvo para un guess inicial que se encuentra entre dos raices y se demora más en saber hacia cual irse por lo tanto el numero de iteraciones es mayor, adicionalmente para las raices la grafica diverge')


# In[ ]:



