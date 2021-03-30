import pickle
import sys, os

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from pltutils import chi2_overlay, interval_bounds_plot


datafile = sys.argv[1]
if sys.argv[2] == 'auto':
  nbins = 'auto'
else:
  nbins = int(sys.argv[2])

with open(datafile, "rb") as fid:
  data_dict = pickle.load(fid)

simdata = data_dict['test_simdata'][0]

simdata.calc_tot_NEES()
simdata.calc_divergence(nbins=nbins)

simdata.compute_sigma_percentages()

# Timeseries plot
simdata.plt_timeseries(stepsize=2)

# Chi2 Overlay Plots
chi2_overlay(simdata.NEES_est[0,:], simdata.state_dim,
  "", plot_nbins=40, plt_chi2_pdf=False)
chi2_overlay(simdata.NEES_true[0,:], simdata.state_dim,
  "", plot_nbins=40, plt_chi2_pdf=True)

# MC Confidence Bounds plot
alpha = 0.95
NEES_est_sum = np.sum(simdata.NEES_est, axis=0)
NEES_true_sum = np.sum(simdata.NEES_true, axis=0)
interval_bounds_plot(NEES_est_sum, simdata.state_dim, alpha,
  "{} interval for EKF Estimated NEES".format(alpha), simdata.nruns)
interval_bounds_plot(NEES_true_sum, simdata.state_dim, alpha,
  "{} interval for EKF True NEES".format(alpha), simdata.nruns)

# White noise visualization
simdata.plt_inn(1/0.1)

plt.show()