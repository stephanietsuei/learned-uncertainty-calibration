import argparse
import sys, os
import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from constants import KIF_DNNDUMP
from pltutils import chi2_overlay
from utils import from_upper_triangular_list, upper_triangular_list, \
  chi2_divergence



parser = argparse.ArgumentParser("Inferencing for all the neural networks")
parser.add_argument("-data", default="ekf_data.pkl", help="pickle of ekf data")
parser.add_argument("-dnndump", default=KIF_DNNDUMP, help="where neural networks are stored")
parser.add_argument("-nbins", default="auto")
parser.add_argument("-neural_net", help="name of network")
args = parser.parse_args()

# histogram bins
if args.nbins == "auto":
  nbins = "auto"
else:
  nbins = int(args.nbins)

# Load training and test data
datafile = args.data
with open(datafile, "rb") as fid:
  data_dict = pickle.load(fid)
num_train_pts = data_dict["x_train_cov"].shape[0]
num_test_pts = data_dict["x_test_cov"].shape[0]
state_dim = 4
cov_dim = state_dim
cov_tri_len = int(cov_dim*(cov_dim+1)/2)

# get the neural net
neural_net = os.path.join(args.dnndump, args.neural_net)
with open(neural_net+".pkl", "rb") as fid:
  net_params = pickle.load(fid)
  train_input_maxes = net_params["input_maxes"]
  train_output_maxes = net_params["output_maxes"]
  input_mode = net_params["input_mode"]
  net_config = net_params["config"]

# scale data input
if input_mode=="cov":
  net_input_test = data_dict["x_test_cov"] / \
    np.tile(train_input_maxes, (num_test_pts,1))
elif input_mode=="statecov":
  net_input_test = data_dict["x_test_statecov"] / \
    np.tile(train_input_maxes, (num_test_pts,1))



# Load the network and make prediction
network = tf.keras.Sequential.from_config(net_config)
network.load_weights(neural_net)
net_output_test = np.float64(network.predict(net_input_test))


# Print network parameters
print("Network parameters:")
print("hidden layers: {}".format(net_params["hidden_layers"]))
print("num epochs: {}".format(net_params["nepochs"]))
print("l2_reg: {}".format(net_params["l2_reg"]))
if "cov_loss_weights" in net_params:
  print("Weights: {}".format(net_params["cov_loss_weights"]))

# Put the output back into the test data
for iii in range(len(data_dict["test_simdata"])):
  simdata = copy.deepcopy(data_dict["test_simdata"][iii])
  npts = simdata.nruns * simdata.num_timesteps
  all_states = np.reshape(simdata.xest, (npts, state_dim))
  all_Pest = np.zeros((npts, cov_tri_len))
  idx = 0
  for i in range(simdata.nruns):
    for j in range(simdata.num_timesteps):
      all_Pest[idx,:] = upper_triangular_list(
        simdata.Pest[i,j,:,:], return_numpy=True, ret_dim=False
      )
      idx += 1

  if input_mode=="cov":
    net_input = all_Pest
  elif input_mode=="statecov":
    net_input = np.hstack((all_states, all_Pest))
  net_input = net_input / np.tile(train_input_maxes, (npts,1))
  net_output = network.predict(net_input)

  idx = 0
  for i in range(simdata.nruns):
    for j in range(simdata.num_timesteps):
      Q = np.reshape(net_output[idx,:], (cov_dim, cov_dim))
      cov = Q @ Q.T
      tri_idx = 0
      for l in range(cov_dim):
        for m in range(l,cov_dim):
          cov[l,m] *= train_output_maxes[tri_idx]
          if l !=m:
            cov[m,l] *= train_output_maxes[tri_idx]
          tri_idx += 1
      simdata.Pest[i,j,:,:] = cov
      idx += 1

  # Compute test divergence
  simdata.calc_tot_NEES()
  (test_divergence, _) = simdata.calc_divergence(nbins=nbins)

  simdata.compute_sigma_percentages()

  # Make plots
  singular_indices = simdata.singular_indices[0]
  all_indices = set([i for i in range(simdata.num_timesteps)])
  nonsingular_indices = list(all_indices - singular_indices)
  nonsingular_indices.sort()
  chi2_vars_est = simdata.NEES_est[0,nonsingular_indices]
  chi2_overlay(chi2_vars_est, simdata.state_dim, "", plot_nbins=40,
    plt_chi2_pdf=True)

  plt.show()
