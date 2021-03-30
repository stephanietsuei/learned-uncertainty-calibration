import pickle
import sys, os
import copy

import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

from ipopt import minimize_ipopt

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from pltutils import chi2_overlay, interval_bounds_plot


def cost_fcn(x):
  A = np.reshape(x, (cov_dim,cov_dim))
  obj = 0.0
  for simdata in simdata_train:
    for k in range(simdata.num_timesteps):
      Pest = simdata.Pest[0,k,:,:]
      P = simdata.P[k,:,:]
      diff = (A @ Pest @ A.T) - P
      diff_sq = diff * diff

      for i in range(cov_dim):
        for j in range(i,cov_dim):
          obj += diff_sq[i,j]
  return obj


# compute or collect
mode = sys.argv[2]

# id
run_id = int(sys.argv[3])
np.random.seed(run_id)

# number of bins in divergence calculation
if sys.argv[4] == "auto":
  nbins = "auto"
else:
  nbins = int(sys.argv[4])

# Load data
datafile = sys.argv[1]
with open(datafile, "rb") as fid:
  data_dict = pickle.load(fid)
Pest_train = data_dict['x_train_cov']
Pest_test = data_dict['x_test_cov']
Ptrue_train = data_dict['y_train']
Ptrue_test = data_dict['y_test']
simdata_train = data_dict['train_simdata']
simdata_test = data_dict['test_simdata']

n_train_pts = Pest_train.shape[0]
n_test_pts = Pest_test.shape[0]

cov_dim = simdata_train[0].state_dim



if mode=="compute":

  # Initial guess
  if run_id == 0:
    x0 = 25*np.random.randn(cov_dim*cov_dim)
  elif run_id == 1:
    x0 = 10*np.random.randn(cov_dim*cov_dim)
  elif run_id == 2:
    x0 = 10*np.ones(cov_dim*cov_dim)
  elif run_id == 3:
    x0 = 25*np.ones(cov_dim*cov_dim)
  elif run_id == 4:
    x0 = np.random.rand(cov_dim*cov_dim)
  elif run_id == 5:
    x0 = 20*np.random.rand(cov_dim*cov_dim)
  elif run_id == 6:
    mat = np.eye(cov_dim)
    x0 = np.reshape(mat, (cov_dim*cov_dim,))
  elif run_id == 7:
    mat = 25*np.eye(cov_dim)
    x0 = np.reshape(mat, (cov_dim*cov_dim,))
  elif run_id == 8:
    mat = 50*np.eye(cov_dim)
    x0 = np.reshape(mat, (cov_dim*cov_dim,))
  elif run_id == 9:
    mat = 75*np.eye(cov_dim)
    x0 = np.reshape(mat, (cov_dim*cov_dim,))
  elif run_id == 10:
    mat = 100*np.eye(cov_dim)
    x0 = np.reshape(mat, (cov_dim*cov_dim,))

  # solve problem
  res = minimize_ipopt(cost_fcn, x0, jac=None)

  # grab results
  A = np.reshape(res.x, (cov_dim,cov_dim))
  print("A: \n{}".format(A))

  # Save the result
  with open("ekf_matrix_A_{}.pkl".format(run_id), "wb") as fid:
    pickle.dump(A, fid)

elif mode=="collect":
  with open("ekf_matrix_A_{}.pkl".format(run_id), "rb") as fid:
    A = pickle.load(fid)

  simdata = data_dict['test_simdata'][0]
  for i in range(simdata.nruns):
    for k in range(simdata.num_timesteps):
      simdata.Pest[i,k,:,:] = A @ simdata.Pest[i,k,:,:] @ A.T

  # Calc NEES and divergence
  simdata.calc_tot_NEES()
  simdata.calc_divergence(nbins=nbins)

  simdata.compute_sigma_percentages()

  # Chi2 overlay plot
  chi2_overlay(simdata.NEES_est[0,:], simdata.state_dim,
    "", plot_nbins=40, plt_chi2_pdf=False)

  # Confidence bounds plot
  NEES_est_sum = np.sum(simdata.NEES_est, axis=0)
  alpha = 0.95
  interval_bounds_plot(NEES_est_sum, simdata.state_dim, alpha,
    "{} interval for EKF Matrix-Adjusted NEES".format(alpha), simdata.nruns)


  plt.show()
