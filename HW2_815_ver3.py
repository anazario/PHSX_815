#Author: Andres Abreu 
#Date: 2/8/2019
#Homework #2 PSHX 815 

#libraries for analytical solution
from math import sin,cos,sqrt

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
    
    s = list(r)
    r_len = len(r)

    for t in range(N):

        k1 = [h*i for i in f(s)]
        k2 = [h*i for i in f([s[j]+0.5*k1[j] for j in range(r_len)])]
        k3 = [h*i for i in f([s[j]+0.5*k2[j] for j in range(r_len)])]
        k4 = [h*i for i in f([s[j]+k3[j] for j in range(r_len)])]
    
        for i in range(r_len):
            s[i] += (k1[i]+2*k2[i]+2*k3[i]+k4[i])*c

    return(s)

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
    error_RK4.append(sqrt((ana_sol[0]-val1[0])**2+(ana_sol[1]-val1[1])**2));
    error_Euler.append(sqrt((ana_sol[0]-val2[0])**2+(ana_sol[1]-val2[1])**2));

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

plt.loglog(total_steps, error_RK4, basex=10, label = '4th order Runge-Kutta')
plt.loglog(total_steps, error_Euler, basex=10, label = 'Euler method')

plt.title('Error vs Total Steps for RK4 and Euler methods')
plt.xlabel('Total Steps (1/h)')
plt.ylabel('Error')
plt.grid(True)

pylab.legend(loc='upper right')
pylab.ylim(10^-14, 1000000)
f.savefig("HW2plotv2.pdf", bbox_inches='tight')
