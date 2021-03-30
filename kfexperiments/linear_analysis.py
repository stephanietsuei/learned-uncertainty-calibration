# Linear covariance analysis for a spring-mass damper system
import sys, os
import numpy as np
import matplotlib.pyplot as plt

from systems import SpringMass
from simulate import DiscreteTimeSystemSimulation

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from pltutils import chi2_overlay, interval_bounds_plot


# Initial state
x0 = 2.5
v0 = 0.0
init_state = np.array([x0, v0])
init_cov = 0.5*np.eye(2)

# Simulation
dt = 0.01
simtime = 100.0
num_timesteps = int(simtime/dt)
time_axis = np.linspace(0, simtime, num_timesteps)
nruns = 50
inputs = 0.75*np.sin(2*np.pi*0.25*time_axis)
inputs = np.reshape(inputs, (num_timesteps,1))

# Define the spring mass system
k = 4    # spring constant
m = 1    # mass
c = 0.1  # damping
dyn_noise = 0.005
meas_noise = 0.003
(sys, est) = SpringMass(dt, c, k, m, x0, v0, dyn_noise, meas_noise)


simdata = DiscreteTimeSystemSimulation(sys, est, init_state, init_cov, 1,
                                       inputs, nruns)
simdata.analysis()

# Timeseries Plot
simdata.plt_timeseries(stepsize=20)

# Overlays with chi2 pdf
chi2_overlay(simdata.NEES_est[0,:], simdata.state_dim, "", plot_nbins='auto')
chi2_overlay(simdata.NEES_true[0,:], simdata.state_dim, "", plot_nbins='auto')

# Confidence Bounds plot
alpha = 0.95
NEES_est_sum = np.sum(simdata.NEES_est, axis=0)
NEES_true_sum = np.sum(simdata.NEES_true, axis=0)
interval_bounds_plot(NEES_est_sum, simdata.state_dim, alpha,
  "{} interval for KF Estimated NEES".format(alpha), nruns)
interval_bounds_plot(NEES_true_sum, simdata.state_dim, alpha,
  "{} interval for KF True NEES".format(alpha), nruns)

# White noise visualization
simdata.plt_inn(1/dt)

plt.show()