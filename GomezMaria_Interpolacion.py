
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


# In[2]:

data=np.loadtxt('chicamocha.txt')
#print (data)
plt.plot(data[:,0], data[:,1],'ko')
plt.show()
x=data[:,0]
y=data[:,1]


# In[3]:

xnew = np.linspace(x[0],x[-1],1000)
flinear = interpolate.interp1d(x, y)
fquadratic= interpolate.interp1d(x, y, kind="quadratic")
fcubic = interpolate.interp1d(x,y,kind = "cubic")

#plt.plot(xnew,fcubic(xnew),'ko')
#plt.show()


# In[4]:

plt.scatter(x,y)
plt.plot(xnew , flinear(xnew ))
plt.plot(xnew , fquadratic(xnew ))
plt.plot(xnew , fcubic(xnew ))
plt.legend(['linear', 'quadratic','cubic' ,'data'])
plt.savefig('ApellidoNombre_Interpola')
plt.close()
#plt.show()


# In[7]:

Deriv = np.ones((len(fquadratic(xnew)),1))
h = xnew[1]-xnew[0]
Deriv = (np.roll(fquadratic(xnew),-1,0) - np.roll(fquadratic(xnew),1,0))/(2*h)
Deriv[0] = (fquadratic(xnew)[1]-fquadratic(xnew)[0])/h
Deriv[len(fquadratic(xnew))-1] = (fquadratic(xnew)[len(fquadratic(xnew))-2] - fquadratic(xnew)[len(fquadratic(xnew))-1])/h

Deriv2 = np.ones((len(flinear(xnew)),1))
h2 = xnew[1]-xnew[0]
Deriv2 = (np.roll(flinear(xnew),-1,0) - np.roll(flinear(xnew),1,0))/(2*h2)
Deriv2[0] = (flinear(xnew)[1]-flinear(xnew )[0])/h2
Deriv2[len(flinear(xnew))-1] = (flinear(xnew)[len(flinear(xnew))-2] - flinear(xnew)[len(flinear(xnew))-1])/h2

Deriv3 = np.ones((len(fcubic(xnew)),1))
h3 = xnew[1]-xnew[0]
Deriv3 = (np.roll(fcubic(xnew),-1,0) - np.roll(fcubic(xnew),1,0))/(2*h3)
Deriv3[0] = (fcubic(xnew)[1]-fcubic(xnew )[0])/h3
Deriv3[len(fcubic(xnew))-1] = (fcubic(xnew)[len(fcubic(xnew))-2] - fcubic(xnew)[len(fcubic(xnew))-1])/h3

Derivdata = np.ones((len(y)))
hd = x[1]-x[0]
Derivdata = (np.roll(y,-1,0) - np.roll(y,1,0))/(2*hd)
Derivdata[0] = (y[1]-y[0])/hd
Derivdata[len(y)-1] = (y[len(y)-2] - y[len(y)-1])/hd


# In[8]:

plt.plot(xnew,Deriv2)
plt.plot(xnew,Deriv3)
plt.plot(xnew,Deriv)
plt.plot(x,Derivdata)
plt.legend(['linear', 'quadratic','cubic' ,'data'])
plt.savefig('ApellidoNombre_Deriv')
plt.close()
#plt.show()


# In[ ]:




# In[ ]:



