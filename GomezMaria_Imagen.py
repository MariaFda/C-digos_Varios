
# coding: utf-8

# In[ ]:

from scipy import misc
from scipy.fftpack import ifft, fft, fftfreq, fft2, ifft2, fftn, ifftn
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from scipy import fftpack


# In[ ]:

# leyendo la imagen y tomandola como array 

# se tomo como bibliografía lo siguiente http://www.scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html#sphx-glr-intro-scipy-auto-examples-solutions-plot-fft-image-denoise-py
arbol = plt.imread('Arboles.png')
#print (type(arbol))
plt.figure(figsize=(10,10))
plt.imshow(1.5*arbol,cmap=plt.cm.gray)

#plt.show()


# In[3]:

# Ahora encontremos la transformada de Fourier del arbol 
plt.figure(figsize=(10,10))
arbol_dft=fft2(arbol)
print("La imagen leída como un arreglo es", arbol_dft,)
plt.title('Transformada de Fourier')
plt.xlabel('Frecuencias')
plt.ylabel('Señal')
plt.plot(arbol_dft)
plt.savefig('GomezMaria_FT2D')
plt.close()
mag_arbol_dft=np.log(np.abs(arbol_dft))
fx, fy = np.meshgrid(np.linspace(0,1,len(arbol)),np.linspace(0,1,len(arbol)))

#print(len(arbol_dft))


# In[6]:

#lena_dft[np.sqrt(fx**2+fy**2) > 0.2]=0.
#plt.subplot(2,2,1)
#plt.imshow(mag_arbol_dft)
#lim=0.1
#pick=(np.sqrt((fx-1)**2+fy**2) > lim)*(np.sqrt((fx)**2+fy**2) > lim)*(np.sqrt(fx**2+(fy-1)**2) > lim)*(np.sqrt((fx-1)**2+(fy-1)**2) > lim)
#lena_dft[pick]=0.
#mag_arbol_dft=np.log(np.abs(arbol_dft))
#plt.subplot(2,2,2)
#plt.imshow(mag_arbol_dft)
#angle_arbol_dft=np.angle(arbol_dft)
#arbol_recovered=np.real(ifft2(arbol_dft))
#plt.subplot(2,2,3)
#plt.imshow(arbol,interpolation=None, cmap=plt.cm.gray)
#plt.subplot(2,2,4)
#plt.imshow(arbol_recovered,interpolation=None,cmap=plt.cm.gray)
#plt.show()


# In[7]:

def filtro_corte(corte1, corte2):
    
    for i in range(len(arbol_dft)):
        for j in range(len(arbol_dft)):
            if(arbol_dft[i][j]>corte1):
                arbol_dft[i][j]=0
            if(arbol_dft[i][j]<-corte2):
                arbol_dft[i][j]=0
                
    return arbol_dft

def espectro(arbol_dft):
    
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(arbol_dft), norm=LogNorm(vmin=3))
    plt.colorbar()
    
    

#plt.figure()
#espectro(arbol_dft)
#plt.title('Fourier transform')  

f_arbol1= filtro_corte(1000,100)
#plt.plot(f_arbol1)
#plt.show()

#plt.figure()
f_arbol2= filtro_corte(900,90)
#plt.plot(f_arbol2)
#plt.show()

plt.figure()
f_arbol3= filtro_corte(600,60)
plt.plot(f_arbol3)
plt.title('GomezMaria_FT2D_filtrada')
plt.savefig('GomezMaria_FT2D_filtrada')
#plt.show()
plt.close()

#plt.figure()
#f_arbol4= filtro_corte(100,10)
#plt.plot(f_arbol4)
#plt.show()
arbol_fft3 = f_arbol3.copy()
plt.figure()
espectro(arbol_fft3)
plt.title('Espectro filtrado')
#arbol_fft1 = f_arbol1.copy()
#arbol_fft2 = f_arbol2.copy()
#arbol_fft3 = f_arbol3.copy()
#arbol_fft4 = f_arbol4.copy()
#arbol_new1 = fftpack.ifft2(arbol_fft1).real
#arbol_new2 = fftpack.ifft2(arbol_fft2).real


arbol_new3 = fftpack.ifft2(arbol_fft3).real


#arbol_new4 = fftpack.ifft2(arbol_fft4).real
plt.figure()
plt.title('Imagen inversa-reconstruida')
plt.imshow(arbol_new3, plt.cm.gray)
plt.savefig('GomezMaria_Imagen_filtrada')
plt.close()

print('Hay varios filtros, fueron comentados, y se realiza la grafica con el mejor resultado')
#plt.show()
#plt.imshow(arbol_new2, plt.cm.gray)
#plt.show()
#plt.imshow(arbol_new3, plt.cm.gray)
#plt.show()
#plt.imshow(arbol_new4, plt.cm.gray)
#se muestra la reconstrucción de la imagen


# In[ ]:




# In[ ]:



