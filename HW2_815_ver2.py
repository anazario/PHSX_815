#Author: Andres Abreu 
#Date: 2/8/2019
#Homework #2 PSHX 815 

#libraries for analytical solution
from math import sin,cos

#libraries for plotting
import numpy as np
import pylab
import matplotlib.pyplot as plt

#Euler method
def Euler(r,h):

    N = int(1/h)
    x = r[0]
    p = r[1]
    w = 1

    for t in range(N):
         p += -w**2*h*x
         x += h*p

    return [x,p]

#4th-order Runge-Kutta method
def RK4O(r,h):

    N = int(1/h)
    c=1/6.0
    
    res = list(r)

    for t in range(N):

        k1 = [h*i for i in f(res)]
        k2 = [h*i for i in f([res[j]+0.5*k1[j] for j in range(len(res))])]
        k3 = [h*i for i in f([res[j]+0.5*k2[j] for j in range(len(res))])]
        k4 = [h*i for i in f([res[j]+k3[j] for j in range(len(res))])]
    
        for i in range(len(res)):
            res[i] += (k1[i]+2*k2[i]+2*k3[i]+k4[i])*c

    error = (h)**4 + (1e-16*N)
    print ("RK4 error: "+ str(error))

    return(res)

#Function that returns x and p time derivative definition
def f(r):

    w = 1

    x = r[0]
    p = r[1]
    
    fx = p
    fp = -x*w**2

    return [fx,fp]

step_size = [1e0,1e-1,1e-2,1e-3,1e-4]

#main code
x_0 = 0
p_0 = 1
w = 1
r = [x_0,p_0]

#analytical solution
ana_sol = [sin(w*1),w*cos(w*1)]

error_RK4 = []
error_Euler = []

#loop over the different step sizes h
for h in step_size:
    
    val1 = RK4O(r,h)
    val2 = Euler(r,h)

    #calculate error for both methods 
    error_RK4.append([abs(val1[i]-ana_sol[i])*100/ana_sol[i] for i in range(len(val1))])
    error_Euler.append([abs(val2[i]-ana_sol[i])*100/ana_sol[i] for i in range(len(val2))])

    print("Step size: "+str(h))
    print("Solution euler ([x,p]):  "+str(val2))
    print("Solution RK ([x,p]):  "+str(val1)+"\n")

#plots
t_0 = 0
t_f = 1

#make list of 1/h
total_steps = [(t_f - t_0)/h for h in step_size]

f = plt.figure()
axes = plt.gca()

x_err_RK4 = [x[0] for x in error_RK4]
x_err_Euler = [x[0] for x in error_Euler]

p_err_RK4 = [p[1] for p in error_RK4]
p_err_Euler = [p[1] for p in error_Euler]

plt.loglog(total_steps, x_err_RK4, basex=10, label = 'RK4 position')
plt.loglog(total_steps, p_err_RK4, basex=10, label = 'RK4 momentum')
plt.loglog(total_steps, x_err_Euler, basex=10, label = 'Euler position')
plt.loglog(total_steps, p_err_Euler, basex=10, label = 'Euler momentum')
plt.title('Error vs Total Steps for RK4 and Euler methods')
plt.xlabel('Total Steps (1/h)')
plt.ylabel('Error (%)')
plt.grid(True)

pylab.legend(loc='upper right')
pylab.ylim(10^-14, 1000000)
f.savefig("HW2plot.pdf", bbox_inches='tight')
