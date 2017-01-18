# -*- coding: utf-8 -*-
"""
Supplementary File 2 - Orb2 ODE aggregation model

'A continuous model of physiological prion aggregation suggests 
a role for Orb2 in gating long-term synaptic information'
Michele Sanguanini & Antonino Cattaneo

@author: michele sanguanini
"""

from scipy import signal as sqw
from scipy import integrate as rk
import numpy as np
from matplotlib import pyplot as plt

# Model parameters initialisation

nu = 0.15 # Hz
dc1 = 0.65 # Duty cycle
dc2 = 0.2 # Duty cycle
T_sigma1 = 1200 # seconds
T_sigma2 = 4000 # seconds
T_end = 6000 # seconds
alpha_tr_RNA = 0.005 # units*Hz
alpha_ex = 0.001 # Hz
alpha_deg = 0.002 # Hz
alpha_agg = 0.008 # Hz
beta_agg_B = 1 # units*Hz
As_th = 3 # units
Bs_th = 3 # units
beta_ex = 0.0004 # Hz

# Create tuple of parameters for ODE numerical integration

Parameters = (nu, dc1, dc2, T_sigma1, T_sigma2, alpha_tr_RNA, Bs_th, alpha_ex, alpha_deg,
                   alpha_agg, beta_agg_B, As_th, Bs_th, beta_ex)

# Definition of the system of ODEs 


def f(y, t, *varargs):
    A = y[0]
    As = y[1]
    Bs = y[2]
    
    # Ordinary Differential Equations from Sanguanini and Cattaneo 2017
    # the stimulus sigma is a square function with frequency nu given 
    # to the system for T_sigma seconds
    
    if t<=T_sigma1:
        dAdt = alpha_tr_RNA * (sqw.square(2*np.pi*nu*t,duty=dc1)+1)/2 * 1/np.sqrt(1
        + np.exp(50 * (Bs - Bs_th))) + alpha_ex * As - 1 * (alpha_deg + alpha_agg) * A
        dAsdt = alpha_agg * A - alpha_ex * As
        dBsdt = (beta_agg_B * (sqw.square(2*np.pi*nu*t,duty=dc1)+1)/2 * 1/np.sqrt(1
        + np.exp(-50 * (As - As_th))) - beta_ex * Bs) * 1/np.sqrt(1+ np.exp(50 * (Bs - Bs_th)))
    elif t<=T_sigma2 and t>T_sigma1 :
        dAdt = alpha_tr_RNA * (sqw.square(2*np.pi*nu*t,duty=dc2)+1)/2 * 1/np.sqrt(1
        + np.exp(50 * (Bs - Bs_th))) + alpha_ex * As - 1 * (alpha_deg + alpha_agg) * A
        dAsdt = alpha_agg * A - alpha_ex * As
        dBsdt = (beta_agg_B * (sqw.square(2*np.pi*nu*t,duty=dc2)+1)/2 * 1/np.sqrt(1
        + np.exp(-50 * (As - As_th))) - beta_ex * Bs) * 1/np.sqrt(1+ np.exp(50 * (Bs - Bs_th)))
    else:
        dAdt = alpha_tr_RNA * 0 * 1/np.sqrt(1
        + np.exp(50 * (Bs - Bs_th))) + alpha_ex * As - 1 * (alpha_deg + alpha_agg) * A
        dAsdt = alpha_agg * A - alpha_ex * As
        dBsdt = (beta_agg_B * 0 * 1/np.sqrt(1
        + np.exp(-50 * (As - As_th))) - beta_ex * Bs) * 1/np.sqrt(1+ np.exp(50 * (Bs - Bs_th)))
        
    return [dAdt, dAsdt, dBsdt]

# Initialisation of the time array
    
t = np.linspace(0,T_end,100000, endpoint=False)

# Ordinary Differential Equation solver

Orb2 = rk.odeint(f, [0,0,0], t, Parameters)

# Plot the levels of Orb2A and Orb2B oligomers as a function of time

plt.plot(t,Orb2[:,1]) # Orb2A oligomers
plt.figure()
plt.plot(t,Orb2[:,2]) # Orb2B oligomers

## Export file production

# Export data from simulation as csv

Orb2_dump = np.ones((np.shape(Orb2)[0], 4))
Orb2_dump[:, 1:4] = Orb2
Orb2_dump[:,0] = t

np.savetxt('Orb2_export_comp.csv', Orb2_dump, delimiter=',')

# Add File header

with open('Orb2_export_comp_full.csv', 'w') as header:
    header.write('Composite stimulation pattern \n')
    header.write('Frequency of stimulus square function: %5.5f Hz \n' %(nu))
    header.write('Translation rate of Orb2A: %5.5f units*Hz \n' %(alpha_tr_RNA))
    header.write('Degradation rate of Orb2A: %5.5f Hz \n' %(alpha_deg))
    header.write('Aggregation rate of Orb2A: %5.5f Hz \n' %(alpha_agg))
    header.write('Exit rate of Orb2A: %5.5f Hz \n' %(alpha_ex))
    header.write('Seeded aggregation rate of Orb2B: %5.5f units*Hz \n' %(beta_agg_B))
    header.write('Exit rate of Orb2B: %5.5f Hz \n' %(beta_ex))
    header.write('Threshold of Orb2A aggregate: %d units \n' %(As_th))
    header.write('Threshold of Orb2B aggregate: %d units \n' %(Bs_th))
    header.write('\n \n')
    with open('Orb2_export_comp.csv', 'r') as file:
        header.write(file.read())

