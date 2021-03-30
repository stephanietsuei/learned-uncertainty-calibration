import argparse
import sys, os
import copy
import numpy as np
import pickle
import glob

sys.path.append(os.path.join(os.getcwd(), "analysis"))
from eval_cov_calibration import CovarianceCalibration

from constants import EVILOCARINA_DUMP, KIF_DUMP, EVILOCARINA_DNNDUMP, \
  KIF_DNNDUMP
from utils import get_xivo_output_filename, get_xivo_gt_filename, \
  chi2_divergence, state_size


parser = argparse.ArgumentParser("Inferencing for all the neural networks")
parser.add_argument("-mode", help="compute | collect")
parser.add_argument("-gpu_id", type=int)
parser.add_argument("-dump", default=KIF_DUMP, help="covariance dump directory")
parser.add_argument("-dnndump", default=KIF_DNNDUMP, help="where neural networks are stored")
parser.add_argument("-cov_type", default="WTV", help="covariance type being predicted")
parser.add_argument("-nbins", default=2000, help="number of bins used in histogram for approximation")
parser.add_argument("-int_upper_lim", default=100.0, help="upper limit of chi2_divergence")
args = parser.parse_args()


# compute mode or collect mode
mode = args.mode

# histogram bins
if args.nbins == "auto":
  nbins = "auto"
  test_nbins = "auto"
else:
  nbins = int(args.nbins)
  test_nbins = int(nbins / 10)


dataset = "tumvi"
if mode == "compute":
  gpu_id = args.gpu_id
  os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
cam_ids = [ 0, 1 ]
sequences = [ "room1", "room2", "room3", "room4", "room5", "room6" ]
covdumpdir = args.dump
dnndumpdir = args.dnndump


processed_nn_pickles = glob.glob(os.path.join(dnndumpdir, '*.pkl'))
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
  chi2_div = np.zeros(len(processed_nn_names))

  all_divs = { 'cov': [], 'gsbcov': [], 'gsbvcov': [] }

  for k,neural_net in enumerate(processed_nn_names):
    sigmas = np.zeros(0)

    # Get neural network name, input mode, and covariance prediction type
    with open(neural_net+".pkl", "rb") as fid:
      net_params = pickle.load(fid)
      network_input_mode = net_params["input_mode"]
      network_cov_type = net_params["cov_type"]
      assert(network_cov_type==args.cov_type)
    neural_net_name = neural_net.split('/')[-1]

    # Collect tumvi data
    for i,cam_id in enumerate(cam_ids):
      for j,seq in enumerate(sequences):

        idx_1 = i*len(sequences) + j
        estimator_data = get_xivo_output_filename(covdumpdir, "tumvi", seq, cam_id)
        gt_data = get_xivo_gt_filename(covdumpdir, "tumvi", seq)

        calib = CovarianceCalibration(seq, gt_data, estimator_data,
          three_sigma=False, start_ind=0, end_ind=None,
          point_cloud_registration='horn',
          plot_timesteps=False, cov_type=network_cov_type,
          adjust_startend_to_samplecov=True)
        calib.align_gt_to_est()
        calib.compute_errors()

        print("Camera {}, {}, {}".format(cam_id, seq, neural_net))

        calib.compute_sigmas(cov_source="neural_net",
          network_model=os.path.join(dnndumpdir, neural_net))

        new_sigmas = calib.get_sigmas(network_cov_type)
        sigmas = np.concatenate((sigmas, new_sigmas.flatten()))

        # Store test divergence for each network. We will use this value to
        # sort the networks and the other value just for reference
        if (seq=="room6"):
          test_div = calib.compute_chi2_divergences(hist_nbins=test_nbins,
            int_upper_lim=args.int_upper_lim)[0]

    # Save divergences for the network.
    df = state_size(network_cov_type)
    all_divs[network_input_mode].append(
      [ test_div, chi2_divergence(sigmas**2, df, hist_nbins=nbins,
                                  int_upper_lim=args.int_upper_lim),
        neural_net_name ]
    )

  # Save data
  with open("chi2_div_{}_{}".format(args.cov_type, gpu_id), "wb") as fid:
    pickle.dump(all_divs, fid)


# compute average for each network per sequence
elif mode == "collect":

  # Load saved data from compute phase
  all_divs0 = pickle.load(open("chi2_div_{}_0".format(args.cov_type), "rb"))
  all_divs1 = pickle.load(open("chi2_div_{}_1".format(args.cov_type), "rb"))
  
  # CSV file for generating table
  csvfile = open("dnn_table_{}.csv".format(args.cov_type), "w")
  csvfile.write("Input Mode,Network Name,Hidden Layers,Test Divergence,Overall Divergence,L2 Reg,Epochs,Weight W,Weight T,Weight V,Weight WW,Weight TT,Weight VV,Weight WT,Weight WV,Weight TV\n")

  print("Best networks on test + overall set for {}: (test div, overall div, nn name)".format(args.cov_type))

  for input_mode in [ "cov", "gsbcov", "gsbvcov" ]:

    all_divs = all_divs0[input_mode] + all_divs1[input_mode]

    # Print best network by divergence on test set
    all_divs.sort(key = lambda x: x[0]+x[1])
    print("{}: {}, {}, {}".format(
      input_mode, all_divs[0][0], all_divs[0][1], all_divs[0][2]))

    # Write neural network information to CSV file
    for j in range(10):
      with open(os.path.join(args.dnndump, all_divs[j][2]+".pkl"), "rb") as fid:
        network_params = pickle.load(fid)
        csvfile.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
          input_mode,
          all_divs[j][2],
          " ".join([ str(vv) for vv in network_params["hidden_layers"] ]),
          all_divs[j][0],
          all_divs[j][1],
          network_params["l2_reg"],
          network_params["nepochs"],
          network_params["weight_W"],
          network_params["weight_T"],
          network_params["weight_V"],
          network_params["weight_WW"],
          network_params["weight_TT"],
          network_params["weight_VV"],
          network_params["weight_WT"],
          network_params["weight_WV"],
          network_params["weight_TV"]
        ))

    csvfile.write("\n")
  csvfile.close()
