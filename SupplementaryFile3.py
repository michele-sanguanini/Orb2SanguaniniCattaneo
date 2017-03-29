# -*- coding: utf-8 -*-
"""
Supplementary File 3 - Orb2 ODE aggregation model with smooth functions (Supplementary text)

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

nu = 10 # Hz
dc = 0.6 # Duty cycle
T_sigma = 10000 # seconds
T_end = 10000 #seconds
alpha_acc = 0.005 # units*Hz
alpha_deg = 0.005 
alpha_ex = 0.0001 # Hz
alpha_agg = 0.0004 # Hz
beta_agg_B = 0.0001 # units*Hz
As_th = 3 # units
Bs_th = 3 # units
beta_ex = 0.00008 # Hz
beta_gen = 0.005
beta_del = 0.0004
beta_self = 0.00003
B0 = beta_gen/beta_del

# Create tuple of parameters for ODE numerical integration

Parameters = (nu, dc, T_sigma, alpha_acc, alpha_ex, beta_gen, beta_del, alpha_deg,
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
    
    if t<=T_sigma:
        dAdt = alpha_acc * (sqw.square(2*np.pi*nu*t,duty=dc)+1)/2/(1 + Bs **3) - 1* (alpha_deg + alpha_agg) * A
        + alpha_ex * As
        dAsdt = alpha_agg * A - alpha_ex * As
        dB =  beta_gen - beta_del * B + beta_ex * Bs**4  - beta_agg_B * (sqw.square(2*np.pi*nu*t,duty=dc)+1)/2 * As**3 * B - beta_self * B * Bs**3
        dBsdt = beta_agg_B * (sqw.square(2*np.pi*nu*t,duty=dc)+1)/2 * As**3 * B + beta_self  * B * Bs**3 - beta_ex * Bs **4
    else:
        dAdt = alpha_acc * 0/(1 + Bs **3) - 1 * (alpha_deg + alpha_agg) * A + alpha_ex * As
        dAsdt = alpha_agg * A - alpha_ex * As
        dB =  beta_gen - beta_del * B + beta_ex * Bs**4  - beta_agg_B * 0 * As**3 * B - beta_self * B * Bs**3
        dBsdt = beta_agg_B * 0 * As**3 * B + beta_self  * B * Bs**3 - beta_ex * Bs **4 
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


## Export file production

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
    

np.savetxt('Orb2_export_cont.csv', Orb2_dumps, delimiter=',')

# Add File header

with open('Orb2_export_cont_full.csv', 'w') as header:
    header.write('Frequency of stimulus square function: %5.5f Hz \n' %(nu))
    header.write('Duty cycle of stimulus square function: %5.5f \n' %(dc))
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
    with open('Orb2_export.csv', 'r') as file:
        header.write(file.read())


# Export full data from simulation as csv
#
#
#np.savetxt('Orb2_export.csv', Orb2_dump, delimiter=',')
#
## Add File header
#
#with open('Orb2_export_full.csv', 'w') as header:
#    header.write('Frequency of stimulus square function: %5.5f Hz \n' %(nu))
#    header.write('Duty cycle of stimulus square function: %5.5f \n' %(dc))
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
#    with open('Orb2_export.csv', 'r') as file:
#        header.write(file.read())

