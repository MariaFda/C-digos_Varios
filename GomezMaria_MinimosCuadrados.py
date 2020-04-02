
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


data = np.loadtxt('data_mean_sq.txt')
#print (data)
len(data[:,1])
x=data[:,0]
y=data[:,1]

#creamos una matriz con algunas entradas en uno y las demás en cero 
Mat_ini=np.zeros((50,2))
Mat_ini[:,0]=x
Mat_ini[:,1]=np.ones(50)
def solucion():
    sol_Mat=np.matmul(Mat_ini.T, Mat_ini)
    sol_B=np.matmul(Mat_ini.T, y)
    sol=np.linalg.solve(sol_Mat, sol_B)
    return sol
sol=solucion()
print ('La pendiente es:', sol[0], 'y la intersección con el eje y es:', sol[1])
x1=np.linspace(0,40, 40)
solu=sol[0]*x1 + sol[1]
plt.plot(x1, solu, 'orange')
plt.scatter(x,y)
plt.title('Regresión lineal')
plt.legend(['m=2.04894, b=26.51298 ', 'data'])
plt.show()

#PARA LA PARTE B GENERAMOS UN INTERVALO ALEATORIO 
def ruido_gauss():
    mu, sigma = 0, 0.2 # media y desvio estandar
    normal = norm(mu, sigma)
    x_n = np.linspace(normal.ppf(0.01),normal.ppf(0.99), 30)
    fp = normal.pdf(x_n) # Función de Probabilidad 
    return(x,fp) 
#print(len(ruido_gauss()[1]))

def funcion_y():
    x_r=6*np.random.rand(30)
    x1,f=ruido_gauss()
    y_r=0.16*np.exp(-x_r*0.5) + f 
    y_r= y_r +  abs(y_r) +1
    return (y_r)
#print (funcion_y())    
def linealizar():
    
    new_y=np.log10(funcion_y())
    
    return (new_y)

#print (len(linealizar()))

def Minimos_cuadrados(A,B):
    t=np.transpose(A)
    matrixA=np.dot(t,A)
    vecB=np.dot(t,B)
    inversa=np.linalg.inv(matrixA)
    sol=np.dot(inversa,vecB)
    return sol


def minimos():   
    m=[]
    b=[]    
    x_r=6*np.random.rand(30)
    for i in range(10):
        new_y=linealizar()
        
        yp=np.zeros((len(new_y),1))
        for i in range(len(new_y)):
            yp[i][0]=new_y[i]
        #print(len(yp))
        xp=np.zeros((len(x_r),2))
        for i in range(len(x_r)):
            xp[i]=x_r[i],1
        #print(len(x_r))
        m.append(Minimos_cuadrados(xp,yp)[0][0])
        b.append(Minimos_cuadrados(xp,yp)[1][0])
    return(m, b)
    
m1,b1=minimos()        
Varianza_A=np.var(m1)
Varianza_T=np.var(b1)
print ('La varianza de A:', Varianza_A,)
print('La varianza de T:', Varianza_T, )




