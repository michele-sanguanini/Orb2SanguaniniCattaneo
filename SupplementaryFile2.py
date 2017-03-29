# -*- coding: utf-8 -*-
"""
Supplementary File 2 - Orb2 ODE aggregation model, composite stimulation

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

nu1 = 0.15 # Hz
dc1 = 0.65 # Duty cycle
nu2 = 0.15 # Hz
dc2 = 0.20 # Duty cycle
T_sigma1 = 1200 # seconds
T_sigma2 = 4000
T_end = 6000
alpha_acc = 0.005 # units*Hz
alpha_ex = 0.001 # Hz
alpha_deg = 0.002 # Hz
alpha_agg = 0.008 # Hz
beta_agg_B = 0.05 # units*Hz
As_th = 3 # units
Bs_th = 3 # units
beta_ex = 0.0005 # Hz
beta_gen = 0.005 # units*Hz
beta_del = 0.0004 # Hz
beta_self = 0.0002 # Hz/units
B0 = beta_gen/beta_del

# Create tuple of parameters for ODE numerical integration

Parameters = (nu1, dc1, nu2, dc2, T_sigma1, T_sigma2, alpha_acc, alpha_ex, beta_gen, beta_del, alpha_deg,
               beta_self, alpha_agg, beta_agg_B, As_th, Bs_th, beta_ex)

# Definition of the system of ODEs 


def f(y, t, *varargs):
    A = y[0]
    As = y[1]
    B =y[2]  
    Bs = y[3]

    # Ordinary Differential Equations from Sanguanini and Cattaneo 2017
    # the stimulus sigma is a square function with frequency nu given 
    # to the system for T_sigma seconds
    
    if t<=T_sigma1:
        dAdt = alpha_acc * (sqw.square(2*np.pi*nu1*t,duty=dc1)+1)/2 * 1/np.sqrt(1
        + np.exp(50 * (Bs - Bs_th))) + alpha_ex * As - 1 * (alpha_deg + alpha_agg) * A
        dAsdt = alpha_agg * A - alpha_ex * As
        dB =  beta_gen - beta_del * B + beta_ex * Bs - beta_agg_B * (sqw.square(2*np.pi*nu1*t,duty=dc1)+1)/2 * 1/np.sqrt(1+ np.exp(-50 * (As - As_th))) * B - beta_self * B * 1/np.sqrt(1+ np.exp(-50 * (Bs - Bs_th)))
        dBsdt = beta_agg_B * (sqw.square(2*np.pi*nu1*t,duty=dc1)+1)/2 * 1/np.sqrt(1
            + np.exp(-50 * (As - As_th))) * B + beta_self * B * 1/np.sqrt(1+ np.exp(-50 * (Bs - Bs_th))) - beta_ex * Bs
    elif t<=T_sigma2 and t>T_sigma1 :
        dAdt = alpha_acc * (sqw.square(2*np.pi*nu2*t,duty=dc2)+1)/2 * 1/np.sqrt(1
        + np.exp(50 * (Bs - Bs_th))) + alpha_ex * As - 1 * (alpha_deg + alpha_agg) * A
        dAsdt = alpha_agg * A - alpha_ex * As
        dB =  beta_gen - beta_del * B + beta_ex * Bs - beta_agg_B * (sqw.square(2*np.pi*nu2*t,duty=dc2)+1)/2 * 1/np.sqrt(1+ np.exp(-50 * (As - As_th))) * B - beta_self * B * 1/np.sqrt(1+ np.exp(-50 * (Bs - Bs_th)))
        dBsdt = beta_agg_B * (sqw.square(2*np.pi*nu2*t,duty=dc2)+1)/2 * 1/np.sqrt(1
            + np.exp(-50 * (As - As_th))) * B + beta_self * B * 1/np.sqrt(1+ np.exp(-50 * (Bs - Bs_th))) - beta_ex * Bs
    else:
        dAdt = alpha_ex * As - 1 * (alpha_deg + alpha_agg) * A
        dAsdt = alpha_agg * A - alpha_ex * As
        dB =  beta_gen - beta_del * B + beta_ex * Bs - beta_self * B * 1/np.sqrt(1+ np.exp(-50 * (Bs - Bs_th)))
        dBsdt = beta_self * B * 1/np.sqrt(1+ np.exp(-50 * (Bs - Bs_th))) - beta_ex * Bs
    return [dAdt, dAsdt, dB, dBsdt]

# Initialisation of the time array
    
t = np.linspace(0,T_end,1000000, endpoint=False)

# Ordinary Differential Equation solver

Orb2 = rk.odeint(f, [0, 0, B0, 0], t, Parameters)

# Plot the levels of Orb2A and Orb2B oligomers as a function of time

plt.plot(t,Orb2[:,0]) # Orb2A monoomers
plt.figure()
plt.plot(t,Orb2[:,1]) # Orb2A oligomers
plt.figure()
plt.plot(t,Orb2[:,2]) # Orb2B monomers
plt.figure()
plt.plot(t,Orb2[:,3]) # Orb2B oligomers


# Export data from simulation as csv (less points)
counter = 0
Orb2_dump = np.ones((np.shape(Orb2)[0], 5)) 
Orb2_dump[:, 1:5] = Orb2
Orb2_dump[:,0] = t

Orb2_dumps = [Orb2_dump[0,:].tolist()]

for i in range(len(Orb2_dump[:,0])):
    if counter==1000:
        Orb2_dumps.append(Orb2_dump[i,:].tolist())
        counter = 0
    else:
        counter += 1

Orb2_dumps = np.asarray(Orb2_dumps)
    

np.savetxt('Orb2_export_comp.csv', Orb2_dumps, delimiter=',')

# Add File header

with open('Orb2_export_full.csv', 'w') as header:
    header.write('Frequency of stimulus square function 1: %5.5f Hz \n' %(nu1))
    header.write('Duty cycle of stimulus square function 1: %5.5f \n' %(dc1))
    header.write('Frequency of stimulus square function 2: %5.5f Hz \n' %(nu2))
    header.write('Duty cycle of stimulus square function 2: %5.5f \n' %(dc2))
    header.write('Total time of stimulation: %5.5f s \n' %(T_sigma))
    header.write('Accumulation rate of Orb2A: %5.5f units*Hz \n' %(alpha_acc))
    header.write('Degradation rate of Orb2A: %5.5f Hz \n' %(alpha_deg))
    header.write('Aggregation rate of Orb2A: %5.5f Hz \n' %(alpha_agg))
    header.write('Exit rate of Orb2A: %5.5f Hz \n' %(alpha_ex))
    header.write('Basal levels of Orb2B: %5.5f units \n' %(B0))    
    header.write('Accumulation rate of Orb2B: %5.5f units*Hz \n' %(beta_gen))
    header.write('Degradation rate of Orb2B: %5.5f Hz \n' %(beta_del))
    header.write('Self-seeded aggregation rate of Orb2B: %5.5f Hz \n' %(beta_self))
    header.write('Seeded aggregation rate of Orb2B: %5.5f units*Hz \n' %(beta_agg_B))
    header.write('Exit rate of Orb2B: %5.5f Hz \n' %(beta_ex))
    header.write('Threshold of Orb2A aggregate: %d units \n' %(As_th))
    header.write('Threshold of Orb2B aggregate: %d units \n' %(Bs_th))
    header.write('\n \n')
    with open('Orb2_export_comp.csv', 'r') as file:
        header.write(file.read())


# Export full data from simulation as csv
#
#
#np.savetxt('Orb2_export_comp.csv', Orb2_dump, delimiter=',')
#
## Add File header
#
#with open('Orb2_export_full.csv', 'w') as header:
#    header.write('Frequency of stimulus square function 1: %5.5f Hz \n' %(nu1))
#    header.write('Duty cycle of stimulus square function 1: %5.5f \n' %(dc1))
#    header.write('Frequency of stimulus square function 2: %5.5f Hz \n' %(nu2))
#    header.write('Duty cycle of stimulus square function 2: %5.5f \n' %(dc2))
#    header.write('Total time of stimulation: %5.5f s \n' %(T_sigma))
#    header.write('Accumulation rate of Orb2A: %5.5f units*Hz \n' %(alpha_acc))
#    header.write('Degradation rate of Orb2A: %5.5f Hz \n' %(alpha_deg))
#    header.write('Aggregation rate of Orb2A: %5.5f Hz \n' %(alpha_agg))
#    header.write('Exit rate of Orb2A: %5.5f Hz \n' %(alpha_ex))
#    header.write('Basal levels of Orb2B: %5.5f units \n' %(B0))    
#    header.write('Accumulation rate of Orb2B: %5.5f units*Hz \n' %(beta_gen))
#    header.write('Degradation rate of Orb2B: %5.5f Hz \n' %(beta_del))
#    header.write('Self-seeded aggregation rate of Orb2B: %5.5f Hz \n' %(beta_self))
#    header.write('Seeded aggregation rate of Orb2B: %5.5f units*Hz \n' %(beta_agg_B))
#    header.write('Exit rate of Orb2B: %5.5f Hz \n' %(beta_ex))
#    header.write('Threshold of Orb2A aggregate: %d units \n' %(As_th))
#    header.write('Threshold of Orb2B aggregate: %d units \n' %(Bs_th))
#    header.write('\n \n')
#    with open('Orb2_export_comp.csv', 'r') as file:
#        header.write(file.read())