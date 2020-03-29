#!/usr/bin/env python3

import brian2
from brian2 import NeuronGroup, SpikeMonitor, StateMonitor
from brian2 import mV, pA, ms, second, Hz, Gohm
import matplotlib.pyplot as plt
import numpy as np

# Create a network
net = brian2.Network() # explicitly handles all devices, compilation, and running


###### Task 2a
# calculate rheobase current
theta_rh = -50 * mV
E_L = -65 * mV
delta_T = 3 * mV
R_m = 0.02 * Gohm
tau_m = 20 * ms

I_rh = (theta_rh - E_L - delta_T) / R_m


u = np.arange(-100, -40, 0.1) * mV
du_dt = 1/tau_m * (-u + E_L + delta_T * np.exp((u-theta_rh)/delta_T) + R_m * I_rh)

# plot phase diagram
# don't forget to label all axes correctly

plt.figure()
plt.plot(u/mV, du_dt * ms/mV, label=r'$I_\mathrm{{injected}}$ = {}pA'.format(I_rh/pA))
plt.legend(loc='best')
plt.grid(b=True)
plt.ylabel(r'$du/dt$ / $\frac{\mathrm{mV}}{\mathrm{ms}}$')
plt.xlabel('$u$ / mV')
plt.title('Phase diagram')
plt.savefig('phase_diagram.pdf', format='pdf')


###### Task 2b,c
# simulate neuron in different conditions
TASK = 'c' # choose task to execute 'b','c'

if (TASK == 'b'):
    # Differential equation:
    eqs = '''
    du/dt = (-u + E_L + delta_T * exp((u-theta_rh)/delta_T) + I_rh*R_m)/tau_m : volt (unless refractory)
    '''
elif (TASK == 'c'):
    sigma_u = 1 * mV
    # Differential equation:
    eqs = '''
    du/dt = (-u + E_L + delta_T * exp((u-theta_rh)/delta_T) + I_rh*R_m)/tau_m
            + sigma_u * sqrt(2/tau_m) * xi: volt (unless refractory)
    '''
    
    
u_thresh = theta_rh + 10*mV
u_reset = -80 * mV
t_ref = 5 * ms

u_0 = np.array([theta_rh/mV - 1, theta_rh/mV + 1, theta_rh/mV + 0.1]) * mV

# I = 0 and u0 = El:
u_0 = np.array([E_L/mV - 1, E_L/mV + 1, E_L/mV + 0.1]) * mV
I_rh = 0 * pA

# Create neurongroup:
neuron = NeuronGroup(3, eqs, threshold='u >= u_thresh', 
                     reset='u = u_reset', method='euler', refractory=t_ref)


# Set starting value to u_reset:
neuron.u = u_0

# Setup state and spike monitor:
state_mon = StateMonitor(neuron, ['u'], record=True)
spike_mon = SpikeMonitor(neuron)

# Add neurons to the net:
net.add([neuron,state_mon,spike_mon])

# Set simulation time and run the model:
#defaultclock.dt = 0.01*ms
t_sim = 500 * ms
net.run(t_sim)

# Store the values:
t_ = state_mon.t
u_ = state_mon.u
s_ = spike_mon.spike_trains()


if (TASK == 'b'):
    # plot time course of u(t) in different conditions
    plt.figure()
    plt.plot(t_ / ms, u_[0] / mV, label=r'$u_0$ = {}mV'.format(u_0[0]/mV))
    plt.plot(t_ / ms, u_[1] / mV, label=r'$u_0$ = {}mV'.format(u_0[1]/mV))
    plt.plot(t_ / ms, u_[2] / mV, label=r'$u_0$ = {}mV'.format(u_0[2]/mV))
    plt.legend(loc='best')
    plt.grid(b=True)
    plt.ylabel('$u$ / mV')
    plt.xlabel('$t$ / ms')
    plt.title('Trace of different initial conditions')
    plt.savefig('simulations.pdf', format='pdf')

elif (TASK == 'c'):
    # plot time course of u(t) in different conditions
    plt.figure()
    plt.plot(t_ / ms, u_[0] / mV, label=r'$u_0$ = {}mV + noise'.format(u_0[0]/mV))
    plt.plot(t_ / ms, u_[1] / mV, label=r'$u_0$ = {}mV + noise'.format(u_0[1]/mV))
    plt.plot(t_ / ms, u_[2] / mV, label=r'$u_0$ = {}mV + noise'.format(u_0[2]/mV))
    plt.legend(loc='best')
    plt.grid(b=True)
    plt.ylabel('$u$ / mV')
    plt.xlabel('$t$ / ms')
    plt.title('Traces of different initial conditions with noise')
    plt.savefig('simulations_noise.pdf', format='pdf')

#plt.show() # avoid having multiple plt.show()s in your code
