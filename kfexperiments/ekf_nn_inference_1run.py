import argparse
import sys, os
import copy
import numpy as np
import pickle
import glob

import tensorflow as tf

sys.path.append(os.path.join(os.getcwd(), "analysis"))
sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from constants import KIF_DNNDUMP
from utils import from_upper_triangular_list, upper_triangular_list, \
  chi2_divergence


parser = argparse.ArgumentParser("Inferencing for all the neural networks")
parser.add_argument("-mode", help="compute | collect")
parser.add_argument("-gpu_id", type=int)
parser.add_argument("-data", default="ekf_data.pkl", help="pickle of ekf data")
parser.add_argument("-dnndump", default=KIF_DNNDUMP, help="where neural networks are stored")
parser.add_argument("-nbins", default="auto")
parser.add_argument("-num_print", default=10, type=int, help="for collect mode")
parser.add_argument("-int_upper_lim", default=100.0, type=float)
args = parser.parse_args()


# compute mode or collect mode
mode = args.mode

# histogram bins
if args.nbins == "auto":
  nbins = "auto"
else:
  nbins = int(args.nbins)

if mode == "compute":
  gpu_id = args.gpu_id
  os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
dnndumpdir = args.dnndump

processed_nn_pickles = glob.glob(os.path.join(dnndumpdir, "*.pkl"))
processed_nn_pickles.sort()
if mode == "compute":
  halfway = int(len(processed_nn_pickles) / 2)
  if gpu_id == 0:
    processed_nn_pickles = processed_nn_pickles[0:halfway]
  elif gpu_id == 1:
    processed_nn_pickles = processed_nn_pickles[halfway:]
processed_nn_names = [ val[:-4] for val in processed_nn_pickles ]



# The main data processing
if mode == "compute":

  # Load training and test data
  datafile = args.data
  with open(datafile, "rb") as fid:
    data_dict = pickle.load(fid)
  num_train_pts = data_dict["x_train_cov"].shape[0]
  num_test_pts = data_dict["x_test_cov"].shape[0]
  state_dim = 4
  cov_dim = state_dim
  cov_tri_len = int(cov_dim*(cov_dim+1)/2)

  # chi2 divergences for each network
  chi2_divs_cov = [] 
  chi2_divs_statecov = []

  for k,neural_net in enumerate(processed_nn_names):

    neural_net_name = neural_net.split('/')[-1]

    # What we're computing
    adj_rho = np.zeros((0,))

    # get the neural net
    with open(neural_net+".pkl", "rb") as fid:
      net_params = pickle.load(fid)
      train_input_maxes = net_params["input_maxes"]
      train_output_maxes = net_params["output_maxes"]
      input_mode = net_params["input_mode"]
      net_config = net_params["config"]
    
    # scale data input
    if input_mode=="cov":
      net_input_train = data_dict["x_train_cov"] / \
        np.tile(train_input_maxes, (num_train_pts,1))
      net_input_test = data_dict["x_test_cov"] / \
        np.tile(train_input_maxes, (num_test_pts,1))
    elif input_mode=="statecov":
      net_input_train = data_dict["x_train_statecov"] / \
        np.tile(train_input_maxes, (num_train_pts,1))
      net_input_test = data_dict["x_test_statecov"] / \
        np.tile(train_input_maxes, (num_test_pts,1))
    
    # Load the network and make prediction
    network = tf.keras.Sequential.from_config(net_config)
    network.load_weights(neural_net)
    net_output_train = np.float64(network.predict(net_input_train))
    net_output_test = np.float64(network.predict(net_input_test))

    # Scale the output
    #net_output_train *= np.tile(train_output_maxes, (num_train_pts,1))
    #net_output_test *= np.tile(train_output_maxes, (num_test_pts,1))


    # put the output back into the simulation data and compute NEES
    for iii in range(len(data_dict["train_simdata"])):
      simdata = copy.deepcopy(data_dict["train_simdata"][iii])
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
        simdata.Pest[0,j,:,:] = cov
        idx += 1

      # Compute NEES
      simdata.calc_tot_NEES()
      adj_rho = np.concatenate((adj_rho, simdata.NEES_est[0,:]))


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
      #net_output = net_output * np.tile(train_output_maxes, (npts,1))

      idx = 0
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
        simdata.Pest[0,j,:,:] = cov
        idx += 1

      # Compute test divergence
      simdata.calc_tot_NEES()
      adj_rho = np.concatenate((adj_rho, simdata.NEES_est[0,:]))
      (test_divergence, _) = simdata.calc_divergence(ret1=True, nbins=nbins,
        int_upper_lim=args.int_upper_lim)

      num_singular = len(simdata.singular_indices[0])

    # Compute overall divergence
    overall_div = chi2_divergence(adj_rho, cov_dim, hist_nbins=nbins,
      int_upper_lim=args.int_upper_lim)

    # Put into list
    if input_mode == "cov":
      chi2_divs_cov.append((test_divergence, overall_div, neural_net_name, num_singular))
    elif input_mode == "statecov":
      chi2_divs_statecov.append((test_divergence, overall_div, neural_net_name, num_singular))
    chi2_divs = (chi2_divs_cov, chi2_divs_statecov)

  with open("ekf_chi2_divs_1run_{}".format(gpu_id), "wb") as fid:
    pickle.dump(chi2_divs, fid)



elif mode=="collect":
  with open("ekf_chi2_divs_1run_0", "rb") as fid:
    (chi2_divs_cov_0, chi2_divs_statecov_0) = pickle.load(fid)
  with open("ekf_chi2_divs_1run_1", "rb") as fid:
    (chi2_divs_cov_1, chi2_divs_statecov_1) = pickle.load(fid) 

  chi2_divs_cov = chi2_divs_cov_0 + chi2_divs_cov_1 
  chi2_divs_statecov = chi2_divs_statecov_0 + chi2_divs_statecov_1

  # sort by sum and print top args.num_print
  chi2_divs_cov.sort(key = lambda x: x[0] + x[1])
  chi2_divs_statecov.sort(key = lambda x: x[0] + x[1])

  num_to_print = min(args.num_print, len(chi2_divs_cov))
  print("Sort by divergence sum")
  print("cov: Network, avg test div, overall div, num singular")
  for i in range(num_to_print):
    avg_test_div = chi2_divs_cov[i][0]
    overall_div = chi2_divs_cov[i][1]
    network_name = chi2_divs_cov[i][2]
    num_singular = chi2_divs_cov[i][3]
    print("{}: {}, {}, {}".format(network_name, avg_test_div, overall_div, num_singular))

  num_to_print = min(args.num_print, len(chi2_divs_cov))
  print("statecov: Network, avg test div, overall div, num singular")
  for i in range(num_to_print):
    avg_test_div = chi2_divs_statecov[i][0]
    overall_div = chi2_divs_statecov[i][1]
    network_name = chi2_divs_statecov[i][2]
    num_singular = chi2_divs_statecov[i][3]
    print("{}: {}, {}, {}".format(network_name, avg_test_div, overall_div, num_singular))


  # sort by test and print top args.num_print
  chi2_divs_cov.sort(key = lambda x: x[0])
  chi2_divs_statecov.sort(key = lambda x: x[0])

  num_to_print = min(args.num_print, len(chi2_divs_cov))
  print("\nSort by test divergence")
  print("cov: Network, avg test div, overall div, num singular")
  for i in range(num_to_print):
    avg_test_div = chi2_divs_cov[i][0]
    overall_div = chi2_divs_cov[i][1]
    network_name = chi2_divs_cov[i][2]
    num_singular = chi2_divs_cov[i][3]
    print("{}: {}, {}, {}".format(network_name, avg_test_div, overall_div, num_singular))

  num_to_print = min(args.num_print, len(chi2_divs_cov))
  print("statecov: Network, avg test div, overall div, num singular")
  for i in range(num_to_print):
    avg_test_div = chi2_divs_statecov[i][0]
    overall_div = chi2_divs_statecov[i][1]
    network_name = chi2_divs_statecov[i][2]
    num_singular = chi2_divs_statecov[i][3]
    print("{}: {}, {}, {}".format(network_name, avg_test_div, overall_div, num_singular))


  # sort by number of singular indices and print top args.num_print
  print("\nNow by singular indices:")
  chi2_divs_cov.sort(key = lambda x: x[3])
  chi2_divs_statecov.sort(key = lambda x: x[3])

  num_to_print = min(args.num_print, len(chi2_divs_cov))
  print("cov: Network, avg test div, overall div, num singular")
  for i in range(num_to_print):
    avg_test_div = chi2_divs_cov[i][0]
    overall_div = chi2_divs_cov[i][1]
    network_name = chi2_divs_cov[i][2]
    num_singular = chi2_divs_cov[i][3]
    print("{}: {}, {}, {}".format(network_name, avg_test_div, overall_div, num_singular))

  num_to_print = min(args.num_print, len(chi2_divs_cov))
  print("statecov: Network, avg test div, overall div")
  for i in range(num_to_print):
    avg_test_div = chi2_divs_statecov[i][0]
    overall_div = chi2_divs_statecov[i][1]
    network_name = chi2_divs_statecov[i][2]
    num_singular = chi2_divs_statecov[i][3]
    print("{}: {}, {}, {}".format(network_name, avg_test_div, overall_div, num_singular))

