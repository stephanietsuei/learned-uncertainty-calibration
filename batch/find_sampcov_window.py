import sys, os
import numpy as np
import multiprocessing
import pickle
import matplotlib.pyplot as plt
import copy

sys.path.append(os.path.join(os.getcwd(), "analysis"))
from eval_cov_calibration import CovarianceCalibration
from constants import EVILOCARINA_DUMP, KIF_DUMP

sys.path.append(os.path.join(os.getcwd(), "utils"))
from utils import chi2_divergence


# compute mode or collect mode
mode = sys.argv[1]

if sys.argv[2] == "auto":
  nbins = "auto"
  ntestbins = "auto"
else:
  nbins = int(sys.argv[2])
  ntestbins = int(nbins/11)


def loop_fcn(i, cam_id, j, seq):

  estimator_data = os.path.join(root, 'tumvi_{}_cam{}'.format(seq, cam_id))
  gt_data = os.path.join(root, 'tumvi_{}_gt'.format(seq))

  all_sigmas_WTV = {}
  all_sigmas_W = {}
  all_sigmas_T = {}
  all_sigmas_V = {}
  all_sigmas_WT = {}

  for window_size in window_sizes:

    print("Camera {}, {}, Window {}".format(cam_id, seq, window_size))

    # Compute sample covariance
    calib = CovarianceCalibration(seq, gt_data, estimator_data,
      three_sigma=False, start_ind=0, end_ind=None,
      point_cloud_registration='horn',
      plot_timesteps=False,
      sample_cov_window_size=window_size,
      adjust_startend_to_samplecov=False)
    calib.align_gt_to_est()
    calib.compute_errors()
    calib.compute_sample_cov()
    calib.write_sample_cov_to_dataset(estimator_data)

    # Compute divergences
    calib = CovarianceCalibration(seq, gt_data, estimator_data,
      three_sigma=False, start_ind=0, end_ind=None,
      point_cloud_registration='horn',
      plot_timesteps=False,
      sample_cov_window_size=window_size,
      adjust_startend_to_samplecov=True)
    calib.align_gt_to_est()
    calib.compute_errors()
    calib.compute_sigmas(cov_source="sampled")

    (gsb_sigma, gsb_sigmaW, gsb_sigmaT, gsb_sigmaV, gsb_sigmaWT) = \
      calib.get_sigmas("all")

    all_sigmas_WTV[window_size] = gsb_sigma
    all_sigmas_W[window_size] = gsb_sigmaW
    all_sigmas_T[window_size] = gsb_sigmaT
    all_sigmas_V[window_size] = gsb_sigmaV
    all_sigmas_WT[window_size] = gsb_sigmaWT

  with open("tumvi_sigmas_{}_cam{}.pkl".format(seq, cam_id), "wb") as fid:
    pickle.dump((
      all_sigmas_WTV, all_sigmas_W, all_sigmas_T, all_sigmas_V, all_sigmas_WT
    ), fid)





dataset = "tumvi"
cam_ids = [ 0, 1 ]
sequences = [ "room1", "room2", "room3", "room4", "room5", "room6" ]
#sequences = [ "room1", "room2" ]
root = sys.argv[3]
#root = EVILOCARINA_DUMP
window_sizes = [ i for i in range(27, 601, 2) ]
#window_sizes = [ 31, 33, 35 ]

# save divergence orders for each of {WTV, W, T, V, WT}
# indices are cam_id0/room1, etc
num_covtypes = 5 # WTV, W, T, V, WT


# Assemble inputs and create output queue
loop_inputs = []
for i,cam_id in enumerate(cam_ids):
  for j,seq in enumerate(sequences):
    loop_inputs.append((i, cam_id, j, seq))


# Do all the computation
if mode == "compute":
  num_cores = multiprocessing.cpu_count()
  with multiprocessing.Pool(processes=num_cores) as pool:
    pool.starmap(loop_fcn, loop_inputs)

elif mode == "collect":
  # Reassemble output from queue into chi2_divergence matrix.
  all_the_sigmas_WTV = {}
  all_the_sigmas_W = {}
  all_the_sigmas_T = {}
  all_the_sigmas_V = {}
  all_the_sigmas_WT = {}
  for window_size in window_sizes:
    # (test sigmas, everything else)
    all_the_sigmas_WTV[window_size] = [ np.zeros((0,1)), np.zeros((0,1)) ]
    all_the_sigmas_W[window_size] = [ np.zeros((0,1)), np.zeros((0,1)) ]
    all_the_sigmas_T[window_size] = [ np.zeros((0,1)), np.zeros((0,1)) ]
    all_the_sigmas_V[window_size] = [ np.zeros((0,1)), np.zeros((0,1)) ]
    all_the_sigmas_WT[window_size] = [ np.zeros((0,1)), np.zeros((0,1)) ]

  for loop_input in loop_inputs:

    (i, cam_id, j, seq) = loop_input
    with open("tumvi_sigmas_{}_cam{}.pkl".format(seq, cam_id), "rb") as fid:
      (all_sigmas_WTV, all_sigmas_W, all_sigmas_T, all_sigmas_V, all_sigmas_WT) \
        = pickle.load(fid)

    window_sizes_in_block = all_sigmas_WTV.keys()
    if (seq=="room6"):
      for window_size in window_sizes_in_block:
        all_the_sigmas_WTV[window_size][0] = np.concatenate((
          all_the_sigmas_WTV[window_size][0],
          all_sigmas_WTV[window_size]
        ), axis=0)
        all_the_sigmas_W[window_size][0] = np.concatenate((
          all_the_sigmas_W[window_size][0],
          all_sigmas_W[window_size]
        ), axis=0)
        all_the_sigmas_T[window_size][0] = np.concatenate((
          all_the_sigmas_T[window_size][0],
          all_sigmas_T[window_size]
        ), axis=0)
        all_the_sigmas_V[window_size][0] = np.concatenate((
          all_the_sigmas_V[window_size][0],
          all_sigmas_V[window_size]
        ), axis=0)
        all_the_sigmas_WT[window_size][0] = np.concatenate((
          all_the_sigmas_WT[window_size][0],
          all_sigmas_WT[window_size]
        ), axis=0)

    else:
      for window_size in window_sizes_in_block:
        all_the_sigmas_WTV[window_size][1] = np.concatenate((
          all_the_sigmas_WTV[window_size][1],
          all_sigmas_WTV[window_size]
        ), axis=0)
        all_the_sigmas_W[window_size][1] = np.concatenate((
          all_the_sigmas_W[window_size][1],
          all_sigmas_W[window_size]
        ), axis=0)
        all_the_sigmas_T[window_size][1] = np.concatenate((
          all_the_sigmas_T[window_size][1],
          all_sigmas_T[window_size]
        ), axis=0)
        all_the_sigmas_V[window_size][1] = np.concatenate((
          all_the_sigmas_V[window_size][1],
          all_sigmas_V[window_size]
        ), axis=0)
        all_the_sigmas_WT[window_size][1] = np.concatenate((
          all_the_sigmas_WT[window_size][1],
          all_sigmas_WT[window_size]
        ), axis=0)

  # Compute divergences
  test_divs = np.zeros((5, len(window_sizes)))
  training_divs = np.zeros((5, len(window_sizes)))
  chi2_div_WTV = []
  chi2_div_W = []
  chi2_div_T = []
  chi2_div_V = []
  chi2_div_WT = []
  for k,window_size in enumerate(window_sizes):
    nees_WTV_test = all_the_sigmas_WTV[window_size][0]**2
    nees_W_test = all_the_sigmas_W[window_size][0]**2
    nees_T_test = all_the_sigmas_T[window_size][0]**2
    nees_V_test = all_the_sigmas_V[window_size][0]**2
    nees_WT_test = all_the_sigmas_WT[window_size][0]**2
    test_div_WTV = chi2_divergence(nees_WTV_test, 9, hist_nbins=nbins)
    test_div_W = chi2_divergence(nees_W_test, 3, hist_nbins=nbins)
    test_div_T = chi2_divergence(nees_T_test, 3, hist_nbins=nbins)
    test_div_V = chi2_divergence(nees_V_test, 3, hist_nbins=nbins)
    test_div_WT = chi2_divergence(nees_WT_test, 6, hist_nbins=nbins)

    nees_WTV_overall = all_the_sigmas_WTV[window_size][1]**2
    nees_W_overall = all_the_sigmas_W[window_size][1]**2
    nees_T_overall = all_the_sigmas_T[window_size][1]**2
    nees_V_overall = all_the_sigmas_V[window_size][1]**2
    nees_WT_overall = all_the_sigmas_WT[window_size][1]**2
    overall_div_WTV = chi2_divergence(nees_WTV_overall, 9, hist_nbins=nbins)
    overall_div_W = chi2_divergence(nees_W_overall, 3, hist_nbins=nbins)
    overall_div_T = chi2_divergence(nees_T_overall, 3, hist_nbins=nbins)
    overall_div_V = chi2_divergence(nees_V_overall, 3, hist_nbins=nbins)
    overall_div_WT = chi2_divergence(nees_WT_overall, 6, hist_nbins=nbins)

    chi2_div_WTV.append((window_size, test_div_WTV, overall_div_WTV))
    chi2_div_W.append((window_size, test_div_W, overall_div_W))
    chi2_div_T.append((window_size, test_div_T, overall_div_T))
    chi2_div_V.append((window_size, test_div_V, overall_div_V))
    chi2_div_WT.append((window_size, test_div_WT, overall_div_WT))

    test_divs[0,k] = test_div_WTV
    test_divs[1,k] = test_div_W
    test_divs[2,k] = test_div_T
    test_divs[3,k] = test_div_V
    test_divs[4,k] = test_div_WT
    training_divs[0,k] = overall_div_WTV
    training_divs[1,k] = overall_div_W
    training_divs[2,k] = overall_div_T
    training_divs[3,k] = overall_div_V
    training_divs[4,k] = overall_div_WT


  # sort by sum of two
  chi2_div_WTV.sort(key = lambda x: x[1] + x[2])
  chi2_div_W.sort(key = lambda x: x[1] + x[2])
  chi2_div_T.sort(key = lambda x: x[1] + x[2])
  chi2_div_V.sort(key = lambda x: x[1] + x[2])
  chi2_div_WT.sort(key = lambda x: x[1] + x[2])

  # Print best window size
  print("Best window sizes sum: (window, test div, training div)")
  print("WTV: {}, {}, {}".format(chi2_div_WTV[0][0], chi2_div_WTV[0][1], chi2_div_WTV[0][2]))
  print("W: {}, {}, {}".format(chi2_div_W[0][0], chi2_div_W[0][1], chi2_div_W[0][2]))
  print("T: {}, {}, {}".format(chi2_div_T[0][0], chi2_div_T[0][1], chi2_div_T[0][2]))
  print("V: {}, {}, {}".format(chi2_div_V[0][0], chi2_div_V[0][1], chi2_div_V[0][2]))
  print("WT: {}, {}, {}".format(chi2_div_WT[0][0], chi2_div_WT[0][1], chi2_div_WT[0][2]))


  # Best test
  chi2_div_WTV.sort(key = lambda x: x[1])
  chi2_div_W.sort(key = lambda x: x[1])
  chi2_div_T.sort(key = lambda x: x[1])
  chi2_div_V.sort(key = lambda x: x[1])
  chi2_div_WT.sort(key = lambda x: x[1])

  print("Best window sizes test: (window, test div, training div)")
  print("WTV: {}, {}, {}".format(chi2_div_WTV[0][0], chi2_div_WTV[0][1], chi2_div_WTV[0][2]))
  print("W: {}, {}, {}".format(chi2_div_W[0][0], chi2_div_W[0][1], chi2_div_W[0][2]))
  print("T: {}, {}, {}".format(chi2_div_T[0][0], chi2_div_T[0][1], chi2_div_T[0][2]))
  print("V: {}, {}, {}".format(chi2_div_V[0][0], chi2_div_V[0][1], chi2_div_V[0][2]))
  print("WT: {}, {}, {}".format(chi2_div_WT[0][0], chi2_div_WT[0][1], chi2_div_WT[0][2]))


  # Best training
  chi2_div_WTV.sort(key = lambda x: x[2])
  chi2_div_W.sort(key = lambda x: x[2])
  chi2_div_T.sort(key = lambda x: x[2])
  chi2_div_V.sort(key = lambda x: x[2])
  chi2_div_WT.sort(key = lambda x: x[2])

  print("Best window sizes training: (window, test div, training div)")
  print("WTV: {}, {}, {}".format(chi2_div_WTV[0][0], chi2_div_WTV[0][1], chi2_div_WTV[0][2]))
  print("W: {}, {}, {}".format(chi2_div_W[0][0], chi2_div_W[0][1], chi2_div_W[0][2]))
  print("T: {}, {}, {}".format(chi2_div_T[0][0], chi2_div_T[0][1], chi2_div_T[0][2]))
  print("V: {}, {}, {}".format(chi2_div_V[0][0], chi2_div_V[0][1], chi2_div_V[0][2]))
  print("WT: {}, {}, {}".format(chi2_div_WT[0][0], chi2_div_WT[0][1], chi2_div_WT[0][2]))


  # make plots
  plt.figure()
  plt.suptitle("Test Divergence vs. Window Size")
  plt.subplot(5,1,1)
  plt.xticks([])
  plt.plot(np.array(window_sizes), test_divs[0,:])
  plt.ylabel("WTV")
  plt.subplot(5,1,2)
  plt.xticks([])
  plt.plot(np.array(window_sizes), test_divs[1,:])
  plt.ylabel("W")
  plt.subplot(5,1,3)
  plt.xticks([])
  plt.plot(np.array(window_sizes), test_divs[2,:])
  plt.ylabel("T")
  plt.subplot(5,1,4)
  plt.xticks([])
  plt.plot(np.array(window_sizes), test_divs[3,:])
  plt.ylabel("V")
  plt.subplot(5,1,5)
  plt.plot(np.array(window_sizes), test_divs[4,:])
  plt.ylabel("WT")
  plt.xlabel("Window Size")

  plt.figure()
  plt.suptitle("Overall Divergence vs. Window Size")
  plt.subplot(5,1,1)
  plt.xticks([])
  plt.plot(np.array(window_sizes), training_divs[0,:])
  plt.ylabel("WTV")
  plt.subplot(5,1,2)
  plt.xticks([])
  plt.plot(np.array(window_sizes), training_divs[1,:])
  plt.ylabel("W")
  plt.subplot(5,1,3)
  plt.xticks([])
  plt.plot(np.array(window_sizes), training_divs[2,:])
  plt.ylabel("T")
  plt.subplot(5,1,4)
  plt.xticks([])
  plt.plot(np.array(window_sizes), training_divs[3,:])
  plt.ylabel("V")
  plt.subplot(5,1,5)
  plt.plot(np.array(window_sizes), training_divs[4,:])
  plt.ylabel("WT")
  plt.xlabel("Window Size")


  plt.figure()
  plt.subplot(2,1,1)
  plt.ylabel("Training")
  plt.xticks([])
  plt.plot(np.array(window_sizes), training_divs[0,:])
  plt.subplot(2,1,2)
  plt.ylabel("Test")
  plt.plot(np.array(window_sizes), test_divs[0,:])
  plt.xlabel("Window Size")

  plt.show()