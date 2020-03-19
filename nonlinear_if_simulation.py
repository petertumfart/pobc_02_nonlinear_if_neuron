#!/usr/bin/env python3

import brian2
from brian2 import NeuronGroup, SpikeMonitor, StateMonitor
from brian2 import mV, pA, ms, second, Hz, Gohm
import matplotlib.pyplot as plt
import numpy as np


# calculate rheobase current

...


# plot phase diagram
# don't forget to label all axes correctly

plt.figure()

...

plt.savefig('phase_diagram.png')


# simulate neuron in different conditions

...

neuron = NeuronGroup(..., method='euler')

brian2.run(...)


# plot time course of u(t) in different conditions

plt.figure()

...

plt.savefig(join('simulations.png'))

plt.show() # avoid having multiple plt.show()s in your code
