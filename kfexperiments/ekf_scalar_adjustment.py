import pickle
import sys, os
import copy

import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from pltutils import chi2_overlay, interval_bounds_plot

datafile = sys.argv[1]

if sys.argv[2] == 'auto':
  nbins = 'auto'
else:
  nbins = int(sys.argv[2])

# Load data
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

# Get scalar mapping on training data 
quad_coef = 0.0
linear_coef = 0.0
for k in range(n_train_pts):
  idx = 0
  for i in range(4):
    for j in range(i,4):
      quad_coef += Pest_train[k,idx] * Pest_train[k,idx]
      linear_coef += -2*Pest_train[i,idx] * Ptrue_train[k,idx]
      idx += 1
quad_coef = np.array([[quad_coef]])
linear_coef = np.array([linear_coef])

scale = cvx.Variable(1)
obj_func = cvx.quad_form(scale, quad_coef) + (linear_coef.T @ scale)
obj = cvx.Minimize(obj_func)
constr = [ scale >= 0]
prob = cvx.Problem(obj, constr)
obj_value = prob.solve(solver=cvx.GUROBI)

print("Optimal value {}".format(obj_value))
print("Scale: {}".format(scale.value))


# Apply scalar mapping to all estimated covariances
s = scale.value
for simdata in simdata_test:
  simdata.Pest = s * simdata.Pest


# Evaluate covariance on test data
simdata = simdata_test[0]

# Calculate NEES and divergence
simdata.calc_tot_NEES()
simdata.calc_divergence(nbins=nbins)

simdata.compute_sigma_percentages()

# Chi2 Overlay Plot
chi2_overlay(simdata.NEES_est[0,:], simdata.state_dim,
  "", plot_nbins=40, plt_chi2_pdf=True)

# Confidence bounds plot
NEES_est_sum = np.sum(simdata.NEES_est, axis=0)
alpha = 0.95
interval_bounds_plot(NEES_est_sum, simdata.state_dim, alpha,
  "{} interval for EKF Scalar-Adjusted NEES".format(alpha), simdata.nruns)


plt.show()