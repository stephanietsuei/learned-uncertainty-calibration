# covariance analysis of 2d localization problem with known beacons
import argparse
import sys, os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from systems import TwoDLocalizationSystem, TwoDLocalizationEKF
from simulate import DiscreteTimeSystemSimulation

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from utils import upper_triangular_list
from pltutils import chi2_overlay, interval_bounds_plot


parser = argparse.ArgumentParser()
parser.add_argument("-mode", default="test1", help="test1 | saveall")
parser.add_argument("-case", default=0, type=int)
parser.add_argument("-nruns", default=50, type=int)
parser.add_argument("-hist_bins", default="auto")
parser.add_argument("-plot_nbins", default=40, type=int)
args = parser.parse_args()


# Beacon locations
beacon_x = [  3.5, 10, -5, -10 ]
beacon_y = [ -1.1, 10, 15, -8.2 ]

# Initial state
x0 = 0.0
y0 = 0.0
v0 = 0.1
theta0 = 0.0
x_init = np.array([x0, y0, v0, theta0])
x_init_est = x_init + np.array([0.03, -0.05, 0.05, 0.008])
P_init = np.diag(np.array([0.1, 0.1, 0.05, 0.01]))

# Simulation parameters
dt = 0.1 # for linearization
simtime = 40.0
num_timesteps = int(simtime/dt)
time_axis = np.linspace(0, simtime, num_timesteps)

# Different sets of inputs
num_total_cases = 12
all_inputs = np.zeros((num_timesteps, 2, num_total_cases))
# 1.
all_inputs[:,0,0] = 2*np.sin(2*np.pi*0.5*time_axis)
all_inputs[:,1,0] = 0.2 * np.ones(num_timesteps)
# 2.
all_inputs[:,0,1] = 2*np.sin(2*np.pi*0.5*time_axis)
all_inputs[:,1,1] = -0.2 * np.ones(num_timesteps)
# 3.
all_inputs[:,0,2] = 2.3*np.sin(2*np.pi*0.5*time_axis)
all_inputs[:,1,2] = -0.3 * np.ones(num_timesteps)
# 4.
all_inputs[:,0,3] = 1*np.sin(2*np.pi*0.5*time_axis)
all_inputs[:,1,3] = -0.3 * np.ones(num_timesteps)
# 5.
all_inputs[:,0,4] = 3.0*np.sin(2*np.pi*0.5*time_axis)
all_inputs[:,1,4] = -0.5 * np.ones(num_timesteps)
# 6.
all_inputs[:,0,5] = 1.8*np.sin(2*np.pi*0.5*time_axis)
all_inputs[:,1,5] = 0.75 * np.ones(num_timesteps)
# 7.
all_inputs[:,0,6] = 1.8*np.sin(2*np.pi*0.5*time_axis)
all_inputs[:,1,6] = -0.75 * np.ones(num_timesteps)
# 8.
all_inputs[:,0,7] = 3.2*np.cos(2*np.pi*0.5*time_axis)
all_inputs[:,1,7] = .75 * np.ones(num_timesteps)
# 9.
all_inputs[:,0,8] = 3.2*np.cos(2*np.pi*0.5*time_axis)
all_inputs[:,1,8] = 1.0 * np.ones(num_timesteps)
# 10.
all_inputs[:,0,9] = 2.2*np.cos(2*np.pi*0.5*time_axis)
all_inputs[:,1,9] = 1.0 * np.ones(num_timesteps)
# 11.
all_inputs[:,0,10] = 2.2*np.cos(2*np.pi*0.5*time_axis)
all_inputs[:,1,10] = -1.0 * np.ones(num_timesteps)
# 12. this is the test sequence
all_inputs[:,0,11] = 1.1*np.cos(2*np.pi*0.5*time_axis)
all_inputs[:,1,11] = -0.8 * np.ones(num_timesteps)

nruns = args.nruns


# covariance parameters
dyn_noise = np.power(np.array([ 0.005*dt, 0.005*dt, 0.001*dt, 0.001*dt ]), 2)
dyn_noise_ekf = dyn_noise
r_noise = 0.01*0.01
r_heading = 0.005*0.005
meas_noise = np.array([r_noise, r_noise, r_noise, r_noise,
                       r_heading, r_heading, r_heading, r_heading])


# get system
sys = TwoDLocalizationSystem(beacon_x, beacon_y, x_init, dyn_noise, meas_noise, dt)
est = TwoDLocalizationEKF(beacon_x, beacon_y, x_init_est, P_init, dyn_noise_ekf, meas_noise, dt)

# simulate and make plots for one
if args.mode == "test1":
  inputs = all_inputs[:,:,args.case]
  simdata = DiscreteTimeSystemSimulation(sys, est, x_init, P_init, 2*len(beacon_x), inputs, nruns)
  simdata.analysis()

  # Chi2 Overlay Plots
  chi2_overlay(simdata.NEES_est[0,:], simdata.state_dim,
    r"Overlay with EKF $\rho_k$", plot_nbins=args.plot_nbins, plt_chi2_pdf=False)
  chi2_overlay(simdata.NEES_true[0,:], simdata.state_dim,
    r"Overlay with EKF Monte-Carlo $\rho_k$", plot_nbins=args.plot_nbins)

  # MC Confidence Bounds plot
  alpha = 0.95
  NEES_est_sum = np.sum(simdata.NEES_est, axis=0)
  NEES_true_sum = np.sum(simdata.NEES_true, axis=0)
  interval_bounds_plot(NEES_est_sum, simdata.state_dim, alpha,
    "{} interval for EKF Estimated NEES".format(alpha), simdata.nruns)
  interval_bounds_plot(NEES_true_sum, simdata.state_dim, alpha,
    "{} interval for EKF True NEES".format(alpha), simdata.nruns)

  # Timeseries plot
  simdata.plt_timeseries(stepsize=2)

  # White noise plot
  simdata.plt_inn(1/dt)

  plt.show()


# Save one run from all of them
elif args.mode == "saveall":
  num_test_cases = 1
  num_train_cases = num_total_cases - num_test_cases
  state_dim = 4
  cov_tri_len = 10
  train_simdatas = []
  test_simdatas = []
  x_train_cov = np.zeros((num_train_cases*num_timesteps, cov_tri_len))
  x_train_statecov = np.zeros((num_train_cases*num_timesteps, cov_tri_len+state_dim))
  x_test_cov = np.zeros((num_test_cases*num_timesteps, cov_tri_len))
  x_test_statecov = np.zeros((num_test_cases*num_timesteps, cov_tri_len+state_dim))
  y_train = np.zeros((num_train_cases*num_timesteps, cov_tri_len))
  y_test = np.zeros((num_test_cases*num_timesteps, cov_tri_len))

  # Collect training data
  idx = 0
  for i in range(num_train_cases):
    inputs = all_inputs[:,:,i]
    simdata = DiscreteTimeSystemSimulation(sys, est, x_init, P_init,
      2*len(beacon_x), inputs, nruns)
    simdata.sim_all_runs()
    simdata.calc_mean()
    simdata.calc_true_cov()
    train_simdatas.append(simdata)

    for j in range(num_timesteps):
      Pest_upper_tri = upper_triangular_list(simdata.Pest[0,j,:,:], return_numpy=True, ret_dim=False)
      Ptrue_upper_tri = upper_triangular_list(simdata.P[j,:,:], return_numpy=True, ret_dim=False)
      xest = simdata.xest[0,j,:]

      x_train_cov[idx,:] = Pest_upper_tri
      x_train_statecov[idx,:] = np.concatenate((xest, Pest_upper_tri))
      y_train[idx,:] = Ptrue_upper_tri
      idx += 1

  # Collect test data
  idx = 0
  for i in range(num_test_cases):
    case_idx = num_train_cases + i
    inputs = all_inputs[:,:,case_idx]
    simdata = DiscreteTimeSystemSimulation(sys, est, x_init, P_init,
      2*len(beacon_x), inputs, nruns)
    simdata.sim_all_runs()
    simdata.calc_mean()
    simdata.calc_true_cov()
    test_simdatas.append(simdata)

    for j in range(num_timesteps):
      Pest_upper_tri = upper_triangular_list(simdata.Pest[0,j,:,:], return_numpy=True, ret_dim=False)
      Ptrue_upper_tri = upper_triangular_list(simdata.P[j,:,:], return_numpy=True,
      ret_dim=False)
      xest = simdata.xest[0,j,:]

      x_test_cov[idx,:] = Pest_upper_tri
      x_test_statecov[idx,:] = np.concatenate((xest, Pest_upper_tri))
      y_test[idx,:] = Ptrue_upper_tri
      idx += 1

  ekf_data = {
    'x_train_cov': x_train_cov,
    'x_train_statecov': x_train_statecov,
    'x_test_cov': x_test_cov,
    'x_test_statecov': x_test_statecov,
    'y_train': y_train,
    'y_test': y_test,
    'train_simdata': train_simdatas,
    'test_simdata': test_simdatas
  }
  with open("ekf_data.pkl", "wb") as fid:
    pickle.dump(ekf_data, fid)
