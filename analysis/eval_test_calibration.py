import argparse, os, sys
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from utils import (get_xivo_output_filename, get_xivo_gt_filename,
                   get_stock_parser, chi2_divergence, state_size, chi2_div_draws)
from pltutils import chi2_overlay
from constants import EVILOCARINA_DNNDUMP, KIF_DNNDUMP

from eval_cov_calibration import CovarianceCalibration



parser = get_stock_parser(
  'Generates plots for evaluation of covariance quality.',
  'Dataset options')

analysis_args = parser.add_argument_group('Analysis Options')
analysis_args.add_argument('-cov_source', default='original',
  help='Covariance used to generate plots. Options are [ original | sampled | scalar | linear | neural_net ]')
analysis_args.add_argument('-network_model', help='Tensorflow model and config')
analysis_args.add_argument('-network_dump', default=EVILOCARINA_DNNDUMP,
  help='Directory where network is located')
analysis_args.add_argument('-sample_size', default=200, type=int, help="For statistical analysis portion, number of points in each sample")
analysis_args.add_argument('-num_draws', default=50, type=int, help="For statistical analysis portion, number of times to compute chi2_divergence of random draw")
analysis_args.add_argument('-hist_nbins', default=2000, type=int, help="For statistical analysis portion, number of bins in approximate histogram")
analysis_args.add_argument('-draws_hist_nbins', default=75, type=int, help="binning to use in much smaller sample sizes of div_draws")
analysis_args.add_argument('-cov_type', default='all')
analysis_args.add_argument('-int_upper_lim', default=100.0, type=float, help="upper limit to chi2 divergence intergral")
analysis_args.add_argument('-plot_nbins', default=40, type=int)

args = parser.parse_args()


# Covariance type to process
if (args.cov_type=="all") and (args.cov_source not in ["original", "sampled"]):
  raise ValueError("can only analyze all for original and sampled case")
if args.cov_type == "all":
  cov_list = [ "WTV", "W", "T", "V", "WT" ]
else:
  cov_list = [args.cov_type]


sigmas = [ np.zeros((0,1)) for cov_type in cov_list ]
df = [ state_size(cov_type) for cov_type in cov_list ]

num_timesteps = 0
num_1sigma = { cov_type:np.zeros(df[k]) for k,cov_type in enumerate(cov_list) }
num_2sigma = { cov_type:np.zeros(df[k]) for k,cov_type in enumerate(cov_list) }
num_3sigma = { cov_type:np.zeros(df[k]) for k,cov_type in enumerate(cov_list) }


for cam_id in [0,1]:
  for seq in ["room6"]:

    print("Cam {}, {}".format(cam_id, seq))

    estimator_data = get_xivo_output_filename(args.dump, "tumvi", seq,
                                              cam_id=cam_id)
    gt_data = get_xivo_gt_filename(args.dump, "tumvi", seq)

    if args.cov_source == "original":
      adjust = False
    else:
      adjust = True
    calib = CovarianceCalibration(seq, gt_data, estimator_data,
      three_sigma=False, start_ind=0, end_ind=None, cov_type=args.cov_type,
      point_cloud_registration="horn", adjust_startend_to_samplecov=adjust)
    calib.align_gt_to_est()
    calib.compute_errors()

    if args.cov_source=="neural_net":
      calib.compute_sigmas(cov_source="neural_net",
        network_model=os.path.join(args.network_dump, args.network_model))
    else:
      calib.compute_sigmas(cov_source=args.cov_source)
    calib.compute_sigma_percentages(print_stuff=False)

    for k,cov_type in enumerate(cov_list):
      new_sigmas = calib.get_sigmas(cov_type)
      sigmas[k] = np.concatenate((sigmas[k], new_sigmas), axis=0)
      num_1sigma[cov_type] += calib.num_1sigma[cov_type]
      num_2sigma[cov_type] += calib.num_2sigma[cov_type]
      num_3sigma[cov_type] += calib.num_3sigma[cov_type]
    
    num_timesteps += calib.est.nposes


for k,cov_type in enumerate(cov_list):

  print("\nCOV TYPE: {}".format(cov_type))

  # Get parameters for sample studies
  npts = sigmas[k].size
  if args.num_draws is None:
    ndraws = int(npts / args.sample_size)
  else:
    ndraws = args.num_draws
  if args.hist_nbins is None:
    nbins = int(args.sample_size / 10)
  else:
    nbins = args.hist_nbins
  sample_chi2_divs = np.zeros(ndraws)

  # Compute all divergences
  chi2_div = chi2_divergence(sigmas[k]**2, df[k], hist_nbins=nbins,
    int_upper_lim=args.int_upper_lim)
  print("{} Divergence: {}".format(cov_type, chi2_div))

  nees_array = (sigmas[k].flatten())**2
  (mean_chi2_divs, std_chi2_divs) = chi2_div_draws(nees_array, df[k],
    num_draws=ndraws, draw_size=args.sample_size,
    hist_nbins=args.draws_hist_nbins, int_upper_lim=args.int_upper_lim)

  print("Mean: {}, std: {}".format(mean_chi2_divs, std_chi2_divs))

  # Total sigma counting display
  num_1sigma_str = []
  num_2sigma_str = []
  num_3sigma_str = []
  for i in range(state_size(cov_type)):
    val1 = num_1sigma[cov_type][i] / num_timesteps * 100.0
    val2 = num_2sigma[cov_type][i] / num_timesteps * 100.0
    val3 = num_3sigma[cov_type][i] / num_timesteps * 100.0
    num_1sigma_str.append("{:4.1f}".format(val1))
    num_2sigma_str.append("{:4.1f}".format(val2))
    num_3sigma_str.append("{:4.1f}".format(val3))
  num_1sigma_str = ", ".join(num_1sigma_str)
  num_2sigma_str = ", ".join(num_2sigma_str)
  num_3sigma_str = ", ".join(num_3sigma_str)
  print("Percent <= 1 sigma: {}".format(num_1sigma_str))
  print("Percent <= 2 sigma: {}".format(num_2sigma_str))
  print("Percent <= 3 sigma: {}".format(num_3sigma_str))

  # Make plots of overall calibration
  plot_chi2_pdf = not (args.cov_source == "original")
  chi2_overlay(sigmas[k]**2, df[k], "", plot_nbins=args.plot_nbins,
    plt_chi2_pdf=plot_chi2_pdf)

plt.show()