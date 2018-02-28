# -*- coding: utf-8 -*-
"""
Supplementary File 3 - Relationship of [B*] with beta_+ and beta_self parameters

'A continuous model of physiological prion aggregation suggests 
a role for Orb2 in gating long-term synaptic information'
Michele Sanguanini & Antonino Cattaneo

@author: michele sanguanini
"""

from scipy import signal as sqw
from scipy import integrate as rk
import numpy as np
import bigfloat # Need to install bigfloat and supporting libraries gmp & mpfr 
bigfloat.exp(5000,bigfloat.precision(100))

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

# Model parameters initialisation

nu = 0.15 # Hz
dc = 0.5 # Duty cycle
T_sigma = 4000 # seconds
T_end = 9000 # seconds
alpha_acc = 0.005 # units*Hz
alpha_ex = 0.001 # Hz
alpha_deg = 0.002 # Hz
alpha_agg = 0.008 # Hz
beta_agg_B = 0.05 # units*Hz
As_th = 3 # units
Bs_th = 3 # units
beta_ex = 0.0005 # Hz
beta_del = 0.0004 # Hz

# Initialise variable parameters

plus = np.linspace(0.00005,0.01,20, endpoint=False) # units*Hz
selfb = np.linspace(0.00005,0.001,20, endpoint=False) # Hz/units

# Definition of the system of ODEs 


def f(y, t, *varargs):
    A = y[0]
    As = y[1]
    B =y[2]  
    Bs = y[3]
    
    # Ordinary Differential Equations from Sanguanini and Cattaneo 2017
    # the stimulus sigma is a square function with frequency nu given 
    # to the system for T_sigma seconds
    
    if t<=T_sigma:
        dAdt = alpha_acc * (sqw.square(2*np.pi*nu*t,duty=dc)+1)/2 * 1/np.sqrt(1
        + np.exp(50 * (Bs - Bs_th))) + alpha_ex * As - 1 * (alpha_deg + alpha_agg) * A
        dAsdt = alpha_agg * A - alpha_ex * As
        dB =  beta_gen - beta_del * B + beta_ex * Bs - beta_agg_B * (sqw.square(2*np.pi*nu*t,duty=dc)+1)/2 * 1/np.sqrt(1+ np.exp(-50 * (As - As_th))) * B - beta_self * B * 1/np.sqrt(1+ np.exp(-50 * (Bs - Bs_th)))
        dBsdt = beta_agg_B * (sqw.square(2*np.pi*nu*t,duty=dc)+1)/2 * 1/np.sqrt(1
            + np.exp(-50 * (As - As_th))) * B + beta_self * B * 1/np.sqrt(1+ np.exp(-50 * (Bs - Bs_th))) - beta_ex * Bs
    else:
        dAdt = alpha_acc * 0 * 1/np.sqrt(1
        + np.exp(50 * (Bs - Bs_th))) + alpha_ex * As - 1 * (alpha_deg + alpha_agg) * A
        dAsdt = alpha_agg * A - alpha_ex * As
        dB =  beta_gen - beta_del * B + beta_ex * Bs - beta_agg_B * 0 * 1/np.sqrt(1+ np.exp(-50 * (As - As_th))) * B - beta_self * B * 1/np.sqrt(1+ np.exp(-50 * (Bs - Bs_th)))
        dBsdt = beta_agg_B * 0 * 1/np.sqrt(1
            + np.exp(-50 * (As - As_th))) * B + beta_self * B * 1/np.sqrt(1+ np.exp(-50 * (Bs - Bs_th))) - beta_ex * Bs
    return [dAdt, dAsdt, dB, dBsdt]

# Initialisation of the time array
    
t = np.linspace(0,T_end,1000000, endpoint=False)

# Data result matrix

result = []

result.append(['beta_+', 'beta_self', 'B*_steady'])

for beta_gen in plus:
    for beta_self in selfb: 

        beta_gen = float(beta_gen) 
        beta_del = float(beta_del)

        # Initialise the resting level of Orb2B

        B0 = beta_gen/beta_del

        # Create tuple of parameters for ODE numerical integration

        Parameters = (nu, dc, T_sigma, alpha_acc, alpha_ex, beta_gen, beta_del, alpha_deg,
                       beta_self, alpha_agg, beta_agg_B, As_th, Bs_th, beta_ex)
        
        # Ordinary Differential Equation solver
        
        Orb2 = rk.odeint(f, [0, 0, B0, 0], t, Parameters)
        
        temp = []
        temp.append(beta_gen)
        temp.append(beta_self)
        temp.append(Orb2[-1,2])
        result.append(temp)

result = np.asarray(result)


## Visualise results as 3D surface plot and 2D contour plot

fig = plt.figure(2, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.set_xlabel(r'$\beta_+$')
ax.set_ylabel(r'$\beta_{\mathrm{self}}$')
ax.set_zlabel('[B*]')


ax.plot_trisurf([float(i) for i in result[1:,0]], [float(i) for i in result[1:,1]], [float(i) for i in result[1:,2]], cmap=plt.cm.CMRmap)
plt.savefig('3dplot_betarequirements.pdf')

fig = plt.figure(3, figsize=(8, 6))

plt.tricontour([float(i) for i in result[1:,0]], [float(i) for i in result[1:,1]], [float(i) for i in result[1:,2]], 50, cmap=plt.cm.CMRmap)
plt.xlabel(r'$\beta_+$')
plt.ylabel(r'$\beta_{\mathrm{self}}$')
plt.colorbar().set_label(' [B*]', rotation=360)
plt.savefig('2dplot_betarequirements.pdf')
# Export data from simulation as csv 


np.savetxt('B_param_aggregation.csv', result, delimiter = ',', fmt = '%s')


    

