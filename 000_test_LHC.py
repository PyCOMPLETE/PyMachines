import sys

sys.path.append('../')


macroparticlenumber = 50000
n_turns = 512

epsn_x  = 2.5e-6
epsn_y  = 3.5e-6
sigma_z = 0.05

intensity = 1e11

mode = 'smooth'
# mode = 'non-smooth'

from LHC import LHC
import pickle

if mode == 'smooth':
    machine = LHC(machine_configuration='Injection', n_segments=5)
elif mode == 'non-smooth':
    with open('lhc_2015_80cm_optics.pkl') as fid:
        optics = pickle.load(fid)
    optics.pop('circumference')
    
    machine = LHC(machine_configuration='Injection', optics_mode = 'non-smooth', V_RF=10e6,  **optics)

bunch   = machine.generate_6D_Gaussian_bunch_matched(
    macroparticlenumber, intensity, epsn_x, epsn_y, sigma_z=sigma_z)

bunch.x +=.001
bunch.y +=.002

machine.one_turn_map.remove(machine.longitudinal_map)

beam_alpha_x = []
beam_beta_x = []
beam_alpha_y = []
beam_beta_y = []




for m in machine.one_turn_map[:]:
    beam_alpha_x.append(bunch.alpha_Twiss_x())
    beam_beta_x.append(bunch.beta_Twiss_x())
    beam_alpha_y.append(bunch.alpha_Twiss_y())
    beam_beta_y.append(bunch.beta_Twiss_y())
    m.track(bunch)

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')


fig, axes = plt.subplots(2, sharex=True)

axes[0].plot(np.array(beam_beta_x), 'bo')
axes[0].plot(machine.transverse_map.beta_x, 'b-')
axes[0].plot(np.array(beam_beta_y), 'ro')
axes[0].plot(machine.transverse_map.beta_y, 'r-')
axes[0].grid('on')
axes[0].set_ylabel('beta_x, beta_y')

axes[1].plot(np.array(beam_alpha_x), 'bo')
axes[1].plot(machine.transverse_map.alpha_x, 'b-')
axes[1].plot(np.array(beam_alpha_y), 'ro')
axes[1].plot(machine.transverse_map.alpha_y, 'r-')
axes[1].grid('on')
axes[1].set_ylabel('alpha_x, alpha_y')
axes[1].set_xlabel('# point')

if mode == 'non-smooth':
    axes[0].plot(np.array(optics['beta_x']), 'xk')
    axes[0].plot(np.array(optics['beta_y']), 'xk')
    axes[1].plot(np.array(optics['alpha_x']), 'xk')
    axes[1].plot(np.array(optics['alpha_y']), 'xk')
    

bunch   = machine.generate_6D_Gaussian_bunch_matched(
    macroparticlenumber, intensity, epsn_x, epsn_y, sigma_z=sigma_z)

beam_x = []
beam_y = []
for _ in xrange(n_turns):
    machine.track(bunch)
    beam_x.append(bunch.mean_x())
    beam_y.append(bunch.mean_y())
    
    
plt.figure(2)
plt.subplot(2,2,1)
plt.plot(beam_x)
plt.subplot(2,2,2)
plt.plot(beam_y)
plt.subplot(2,2,3)
plt.plot(np.fft.rfftfreq(len(beam_x), d=1.), np.abs(np.fft.rfft(beam_x)))
plt.subplot(2,2,4)
plt.plot(np.fft.rfftfreq(len(beam_y), d=1.), np.abs(np.fft.rfft(beam_y)))

#~ plt.figure(100)
#~ plt.plot(optics['s'][:],optics['beta_x'][:], '-o')



plt.show()
