import argparse, os, sys
import pickle
from copy import copy
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2, rv_histogram
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from interpolate_gt import groundtruth_interpolator
from estimator_data import EstimatorData
from evaluate_ate import align_horn
from utils import (from_upper_triangular_list, scale_covariance_matrix,
                   rigid_transform_3d, calc_avg_sampling_freq,
                   upper_triangular_list, state_indices,
                   compute_scaled_eigvecs, get_xivo_output_filename,
                   get_xivo_gt_filename, get_stock_parser, chi2_divergence,
                   chi2_div_draws, state_size)
from pltutils import (chi2_overlay, time_three_plots, time_plot,
                      error_three_plots, inlier_outlier_time_scatter,
                      inlier_outlier_time_three_plots,
                      inlier_outlier_error_three_plots, plot_3D_error_cloud,
                      inlier_outlier_3D_error_cloud, sigma_hist,
                      inlier_outlier_sigma_hist, psd_plot,
                      plot_3d_trajectories)
from constants import EVILOCARINA_DNNDUMP, KIF_DNNDUMP

# Suppress matplotlib too many figures warnings
matplotlib.rc("figure", max_open_warning=0)



parser = get_stock_parser(
  'Generates plots for evaluation of covariance quality.',
  'Dataset options')
parser.add_argument('-mode', default='analysis',
  help='[ analysis | compute_sample_cov ]')
parser.add_argument('-point_cloud_registration', default='horn',
  help='[ first_step | teaser | horn ] method for point cloud registration')

# Sample covariance window
sample_cov_args = parser.add_argument_group('Sample Covariance Options')
sample_cov_args.add_argument('-sample_cov_window_size', default=11, type=int)

# sigma computation and plotting options in analysis mode
analysis_args = parser.add_argument_group('Analysis Options')
analysis_args.add_argument('-make_plots', action='store_false',
  default=True, help='whether or not to display plots')
analysis_args.add_argument('-plot_timesteps', action='store_true',
  default=False,
  help='if true, plot a count of vision packets instead of time')
analysis_args.add_argument('-cov_source', default='original',
  help='Covariance used to generate plots. Options are [ original | sampled | scalar | linear | neural_net ]')
analysis_args.add_argument('-network_model', help='Tensorflow model and config')
analysis_args.add_argument('-network_dump', default=EVILOCARINA_DNNDUMP,
  help='Directory where network is located')
analysis_args.add_argument('-cov_type', default="all",
  help='[ WTV | W | T | V | WT | all ]')
analysis_args.add_argument('-hist_nbins', default=2000, type=int,
  help='Number of histogram bins for computing chi2_divergence')
analysis_args.add_argument('-plot_nbins', default=40, type=int,
  help='Number of bins to use in plotting chi2 overlays')
analysis_args.add_argument('-int_upper_lim', default=100.0, type=float,
  help='Upper limit of chi2 divergence integral')
analysis_args.add_argument('-num_draws', default=50, type=int,
  help='Number of sample groups in chi2_div_draws')
analysis_args.add_argument('-draw_size', default=200, type=int,
  help='Number of draws to average over in chi2_div_draws')
analysis_args.add_argument('-draws_hist_nbins', default=75, type=int,
  help='Number of histogram bins to use in sample drawing, where there are fewer points than the whole dataset')

# Arguments for dataset output
parser.add_argument('-start_ind', default=0, type=int,
  help='estimator timestep at which to start the analysis')
parser.add_argument('-end_ind', default=None, type=int,
  help='estimator timestep at which to finish the analysis')

# arguments for teaser alignment
teaser_args = parser.add_argument_group('Teaser Point Cloud Alignment')
teaser_args.add_argument('-teaser_noise_bound', default=0.1, type=float)
teaser_args.add_argument('-teaser_cbar2', default=2.0, type=float)
teaser_args.add_argument('-teaser_rotation_cost_threshold', default=1e-6,
  type=float)
teaser_args.add_argument('-teaser_rotation_estimation_algorithm',
  default='GNC_TLS')
teaser_args.add_argument('-teaser_rotation_gnc_factor', default=1.4,
  type=float)
teaser_args.add_argument('-teaser_rotation_max_iterations', default=100,
  type=float)
teaser_args.add_argument('-teaser_max_clique_time_limit', default=3600.0,
  type=float)
teaser_args.add_argument('-teaser_max_clique_exact_solution', default=True,
  type=bool)


class CovarianceCalibration:
  def __init__(self, seq, gt_file, estimator_results, three_sigma=True,
               start_ind=0, end_ind=None, point_cloud_registration='teaser',
               plot_timesteps=False, sample_cov_window_size=11,
               adjust_startend_to_samplecov=False, cov_type="all"):

    self.seq = seq
    self.three_sigma = three_sigma
    self.point_cloud_registration = point_cloud_registration
    self.plot_timesteps = plot_timesteps

    # Covariance type to analyze
    self.cov_type = cov_type
    if self.cov_type == "all":
      self.cov_list = [ "WTV", "W", "T", "V", "WT" ]
    else:
      self.cov_list = [ self.cov_type ]

    # Point cloud registration parameters
    if self.point_cloud_registration == "teaser":
      self.solver_params = tpp.RobustRegistrationSolver.Params()
      self.solver_params.estimate_scaling = False

    # Text for x-axis time plots
    if self.plot_timesteps:
      self.time_axis_label = 'Timesteps (#vision packets)'
    else:
      self.time_axis_label = 'Time (s)'

    # Load ground truth data
    self.gt_interpolator = groundtruth_interpolator(gt_file)

    # load estimator data
    self.est = EstimatorData(estimator_results, start_ind=start_ind,
      end_ind=end_ind,
      adjust_startend_to_samplecov=adjust_startend_to_samplecov)

    self.time_axis = np.zeros((self.est.nposes,))
    self.time_axis_orig = []
    self.inliers = []
    self.outliers = []
    self.inliers_orig = []
    self.outliers_orig = []

    # Ground truth and error trajectories
    self.Rsb_gt = []
    self.Tsb_gt = np.zeros((3,self.est.nposes))
    self.Vsb_gt = np.zeros((3,self.est.nposes))
    self.Wsb_gt = np.zeros((3,self.est.nposes)) # duplicate of self.Rsb_gt

    self.Rsb_error = []
    self.Wsb_error = np.zeros((3,self.est.nposes))
    self.Tsb_error = np.zeros((3,self.est.nposes))
    self.Vsb_error = np.zeros((3,self.est.nposes))

    # covariance magnitude and directions
    self.gsb_sigma = np.zeros((self.est.nposes,1))
    self.gsb_sigmaW = np.zeros((self.est.nposes,1))
    self.gsb_sigmaT = np.zeros((self.est.nposes,1))
    self.gsb_sigmaV = np.zeros((self.est.nposes,1))
    self.gsb_sigmaWT = np.zeros((self.est.nposes,1))

    self.evecWTV_scales = np.zeros((9,self.est.nposes))
    self.evecW_scales = np.zeros((3,self.est.nposes))
    self.evecT_scales = np.zeros((3,self.est.nposes))
    self.evecV_scales = np.zeros((3,self.est.nposes))
    self.evecWT_scales = np.zeros((6,self.est.nposes))

    self.eigvec_coord_WTV = np.zeros((9,self.est.nposes))
    self.eigvec_coord_W = np.zeros((3,self.est.nposes))
    self.eigvec_coord_T = np.zeros((3,self.est.nposes))
    self.eigvec_coord_V = np.zeros((3,self.est.nposes))
    self.eigvec_coord_WT = np.zeros((6,self.est.nposes))

    # sigma counts
    self.num_1sigma = { c: [] for c in self.cov_list }
    self.num_2sigma = { c: [] for c in self.cov_list }
    self.num_3sigma = { c: [] for c in self.cov_list }

    # parameters for computing sample covariance windows
    assert(sample_cov_window_size % 2 == 1)
    half_window_size = int(sample_cov_window_size / 2)
    self.sample_cov_window_size = sample_cov_window_size
    self.half_window_size = half_window_size
    self.sample_covWTV = np.zeros((9,9,self.est.nposes))

    self.net_output = [] # will become a np.array later
    self.nn_cov_type = None

    # keep track of which indices don't cause problems with singularities,
    # mostly important for neural network case
    self.singular_indices = set()
    self.nonsingular_indices = []


  def set_teaser_params(self, noise_bound=0.01, cbar2=1.0,
    max_clique_exact_solution=True, max_clique_time_limit=3600.0,
    rotation_cost_threshold=1e-06, rotation_gnc_factor=1.4,
    rotation_max_iterations=100, rotation_estimation_algorithm='GNC_TLS'):
    self.solver_params.noise_bound = noise_bound
    self.solver_params.cbar2 = cbar2
    self.solver_params.max_clique_exact_solution = max_clique_exact_solution
    self.solver_params.max_clique_time_limit = max_clique_time_limit
    self.solver_params.rotation_cost_threshold = rotation_cost_threshold
    self.solver_params.rotation_gnc_factor = rotation_gnc_factor
    self.solver_params.rotation_max_iterations = rotation_max_iterations

    if rotation_estimation_algorithm == 'GNC_TLS':
      self.solver_params.rotation_estimation_algorithm = \
        tpp.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    elif rotation_estimation_algorithm == 'FGR':
      self.solver_params.rotation_estimation_algorithm = \
        tpp.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.FGR


  def align_gt_to_est(self):
    for i in range(self.est.nposes):
      timestamp = self.est.time_axis[i]

      R_sb_gt, T_sb_gt, V_sb_gt = self.gt_interpolator.get_gt_gsb(timestamp)
      self.Rsb_gt.append(R_sb_gt)
      self.Tsb_gt[:,i] = T_sb_gt
      self.Vsb_gt[:,i] = V_sb_gt
      self.Wsb_gt[:,i] = R_sb_gt.as_rotvec()

      self.time_axis[i] = timestamp

    if self.plot_timesteps:
      self.time_axis = np.linspace(0,self.est.nposes-1,self.est.nposes)
    else:
      # subtract first number from time axis so that it starts from 0,
      # convert from ns to s
      self.time_axis = self.est.time_axis - self.est.time_axis[0]
      self.time_axis = self.time_axis / 1e9

    # copy time axis before we mess with it
    self.time_axis_orig = self.time_axis

    # Align the ground truth and estimator data
    if self.point_cloud_registration == 'horn':
      rot,trans = rigid_transform_3d(np.transpose(self.Tsb_gt),
                                     np.transpose(self.est.Tsb))
      rotObj = Rotation.from_matrix(rot)
      for i in range(self.est.nposes):
        self.Rsb_gt[i] = rotObj * self.Rsb_gt[i]
        self.Tsb_gt[:,i] = rot.dot(self.Tsb_gt[:,i]) + trans
        self.Vsb_gt[:,i] = rot.dot(self.Vsb_gt[:,i])
      self.inliers = [ i for i in range(self.est.nposes) ]
      self.outliers = []

    # move ground truth data to same frame as estimator data.
    # Assume first frame is the ground truth measurement
    elif self.point_cloud_registration == 'first_step':
      timestamp, _, _, _, _, _, _, _, _ = self.est.get_estimator_point(0)
      rotObj, trans, _ = self.gt_interpolator.get_gt_gsb(timestamp)
      rotObj = rotObj.inv()
      rot = rotObj.as_dcm()
      for i in range(self.est.nposes):
        self.Rsb_gt[i] = rotObj * self.Rsb_gt[i]
        self.Tsb_gt[:,i] = rot.dot(self.Tsb_gt[:,i] - trans)
        self.Vsb_gt[:,i] = rot.dot(self.Vsb_gt[:,i])
      self.inliers = [ i for i in range(self.est.nposes) ]
      self.outliers = []

    # Use teaser
    elif self.point_cloud_registration == 'teaser':
      print("Teaser parameters:")
      print(self.solver_params)

      solver = tpp.RobustRegistrationSolver(self.solver_params)
      solver.solve(self.Tsb_gt, self.est.Tsb)
      solution = solver.getSolution()

      allinds = [ i for i in range(self.est.nposes) ]
      rotation_inliers = solver.getRotationInliers()
      translation_inliers = solver.getTranslationInliers()
      self.inliers = list(
        set(rotation_inliers).intersection(set(translation_inliers)))
      self.outliers = list(set(allinds).difference(set(self.inliers)))
      print("Total: {} inliers, {} outliers".format(len(self.inliers),
        len(self.outliers)))

      rot = solution.rotation
      rotObj = Rotation.from_matrix(rot)
      trans = solution.translation
      for i in range(self.est.nposes):
        self.Rsb_gt[i] = rotObj * self.Rsb_gt[i]
        self.Tsb_gt[:,i] = rot.dot(self.Tsb_gt[:,i]) + trans
        self.Vsb_gt[:,i] = rot.dot(self.Vsb_gt[:,i])

    # make copies of inliers and outliers arrays
    self.inliers_orig = self.inliers
    self.outliers_orig = self.outliers


  def compute_evec_axes(self, state_portion, t_ind, cov_source):
    # Compute sigma values for a subset of the motion state.

    (min_state_ind, max_state_ind) = state_indices(state_portion)
    cov_orig = self.est.P[min_state_ind:max_state_ind,
                          min_state_ind:max_state_ind, t_ind]
    cov_dim = max_state_ind - min_state_ind

    if cov_source=="original":
      cov = cov_orig
    elif cov_source=="sampled":
      assert(max_state_ind <= 9)
      assert(min_state_ind < max_state_ind)
      cov = self.est.sample_covWTV[min_state_ind:max_state_ind,
                                   min_state_ind:max_state_ind,
                                   t_ind]
    elif cov_source=="scalar":
      attr_name = "cov_scale_" + state_portion
      scale_factor = getattr(self.est, attr_name)
      cov = cov_orig*scale_factor
    elif cov_source=="linear":
      attr_name = "cov_A_" + state_portion
      A = getattr(self.est, attr_name)
      cov = A @ cov_orig @ A.T
    elif (cov_source=="neural_net") or (cov_source=="net_output"):
      Q = np.reshape(self.net_output[t_ind,:], (cov_dim, cov_dim))
      cov = Q @ Q.T
      tri_idx = 0
      # Scale output
      for i in range(cov_dim):
        for j in range(i,cov_dim):
          cov[i,j] *= self.upper_tri_maxes[tri_idx]
          if j != i:
            cov[j,i] *= self.upper_tri_maxes[tri_idx]
          tri_idx += 1
    else:
      raise ValueError("Invalid value given for option cov_source")

    (X, scales) = compute_scaled_eigvecs(cov, return_scales=True)
    return (X, scales)


  def compute_sigma(self, state_portion, error_vec, t_ind, cov_source):
    (X, evec_scales) = \
      self.compute_evec_axes(state_portion, t_ind, cov_source)
    try:
      CoordInEigvecs = np.linalg.solve(X, error_vec)
      sigma = np.linalg.norm(CoordInEigvecs)
    except np.linalg.LinAlgError:
      sigma = -1
      self.singular_indices.add(t_ind)
      CoordInEigvecs = np.zeros((state_size(state_portion),))
    return (sigma, evec_scales, CoordInEigvecs)


  def innovation_plot(self):
    """Compute, mean, variance of innovation signals and plot them."""

    mean_inn_Wsb = np.mean(self.est.inn_Wsb, axis=1)
    mean_inn_Tsb = np.mean(self.est.inn_Tsb, axis=1)
    mean_inn_Vsb = np.mean(self.est.inn_Vsb, axis=1)

    var_inn_Wsb = np.var(self.est.inn_Wsb, axis=1)
    var_inn_Tsb = np.var(self.est.inn_Tsb, axis=1)
    var_inn_Vsb = np.var(self.est.inn_Vsb, axis=1)

    plot_titles_Wsb = [
      "mean/var: {0:10.3g}, {1:10.3g}".format(mean_inn_Wsb[i], var_inn_Wsb[i])
      for i in range(3)
    ]
    plot_titles_Tsb = [
      "mean/var: {0:10.3g}, {1:10.3g}".format(mean_inn_Tsb[i], var_inn_Tsb[i])
      for i in range(3)
    ]
    plot_titles_Vsb = [
      "mean/var: {0:10.3g}, {1:10.3g}".format(mean_inn_Vsb[i], var_inn_Vsb[i])
      for i in range(3)
    ]
    time_three_plots(self.time_axis, self.est.inn_Tsb, r"$T_{sb}$ innovation",
      titles=plot_titles_Tsb)
    time_three_plots(self.time_axis, self.est.inn_Wsb, r"$W_{sb}$ innovation",
      titles=plot_titles_Wsb)
    time_three_plots(self.time_axis, self.est.inn_Vsb, r"$V_{wb}$ innovation",
      titles=plot_titles_Vsb)


  def innovation_psd_plot(self):
    freq = calc_avg_sampling_freq(self.est.time_axis / 1000000000)
    print("\nAverage sampling frequency: {} Hz".format(freq))

    psd_plot(self.est.inn_Wsb, freq, r"$W_{sb}$ PSD")
    psd_plot(self.est.inn_Tsb, freq, r"$T_{sb}$ PSD")
    psd_plot(self.est.inn_Vsb, freq, r"$V_{sb}$ PSD")


  def compute_errors(self):
    for ind in range(self.est.nposes):
      self.Tsb_error[:,ind] = self.est.Tsb[:,ind] - self.Tsb_gt[:,ind]
      self.Vsb_error[:,ind] = self.est.Vsb[:,ind] - self.Vsb_gt[:,ind]
      self.Rsb_error.append(self.est.Rsb[ind] * self.Rsb_gt[ind].inv())
      self.Wsb_error[:,ind] = self.Rsb_error[-1].as_rotvec().flatten()


  def compute_sigma_percentages(self, print_stuff=True):
    for cov_type in self.cov_list:
      dim = state_size(cov_type)
      evec_scales = self.get_evec_coord(cov_type)

      num_1sigma = np.zeros(dim)
      num_2sigma = np.zeros(dim)
      num_3sigma = np.zeros(dim)

      for ind in range(self.est.nposes):

        # by-axis, evec_scales calculation
        for k in range(dim):
          val = np.abs(evec_scales[k,ind])
          if val <= 3.0:
            num_3sigma[k] += 1
          if val <= 2.0:
            num_2sigma[k] += 1
          if val <= 1.0:
            num_1sigma[k] += 1

      # Display percentage of key in each spot
      if print_stuff:
        print("\nSigma-percentages for individual axes")
        for k in range(dim):
          print("\n{}, state #{} sigma percentages:".format(cov_type, k))
          print("1-sigma: {}".format(num_1sigma[k] / self.est.nposes))
          print("2-sigma: {}".format(num_2sigma[k] / self.est.nposes))
          print("3-sigma: {}".format(num_3sigma[k] / self.est.nposes))
      
      self.num_1sigma[cov_type] = num_1sigma
      self.num_2sigma[cov_type] = num_2sigma
      self.num_3sigma[cov_type] = num_3sigma


  def compute_sigmas(self, cov_source="original", network_model=None,
    net_output=None, output_maxes=None, cov_type=None):

    if cov_source=="neural_net":
      import tensorflow as tf

      # load network parameters
      net_params = pickle.load(open(network_model + ".pkl", "rb"))
      
      # load the network
      network = tf.keras.Sequential.from_config(net_params["config"])
      network.load_weights(network_model)
      
      # get the input
      net_input = self.est.get_network_input(net_params["input_mode"],
        net_params["input_maxes"], net_params["cov_type"])
      
      # make predictions
      self.net_output = np.float64(network.predict(net_input))
      self.upper_tri_maxes = net_params["output_maxes"]
      self.nn_cov_type = net_params["cov_type"]
      assert(self.nn_cov_type == self.cov_type)

      # clear memory
      tf.keras.backend.clear_session()

    elif cov_source=="net_output":
      if (net_output is None) or (output_maxes is None) or (cov_type is None):
        raise ValueError("you forgot to put in the network output and scales")
      self.net_output = net_output
      self.upper_tri_maxes = output_maxes
      self.nn_cov_type = cov_type
    
      assert(self.nn_cov_type == self.cov_type)

    self.compute_sigmas_hlpr(cov_source)



  def compute_sigmas_hlpr(self, cov_source="original"):
    self.singular_indices = set()

    for ind in range(self.est.nposes):
      # Create point
      Point = np.hstack((self.Wsb_error[:,ind],
                         self.Tsb_error[:,ind],
                         self.Vsb_error[:,ind]))
      
      for cov_type in self.cov_list:
        ind_l, ind_u = state_indices(cov_type)
        p = Point[ind_l:ind_u]
        (sigma, evec_scales, CoordInEigvecs) = \
          self.compute_sigma(cov_type, p, ind, cov_source)

        if cov_type=="WTV":
          self.gsb_sigma[ind] = sigma
          self.evecWTV_scales[:,ind] = evec_scales
          self.eigvec_coord_WTV[:,ind] = CoordInEigvecs
        elif cov_type=="W":
          self.gsb_sigmaW[ind] = sigma
          self.evecW_scales[:,ind] = evec_scales
          self.eigvec_coord_W[:,ind] = CoordInEigvecs
        elif cov_type=="T":
          self.gsb_sigmaT[ind] = sigma
          self.evecT_scales[:,ind] = evec_scales
          self.eigvec_coord_T[:,ind] = CoordInEigvecs
        elif cov_type=="V":
          self.gsb_sigmaV[ind] = sigma
          self.evecV_scales[:,ind] = evec_scales
          self.eigvec_coord_V[:,ind] = CoordInEigvecs
        elif cov_type=="WT":
          self.gsb_sigmaWT[ind] = sigma
          self.evecWT_scales[:,ind] = evec_scales
          self.eigvec_coord_WT[:,ind] = CoordInEigvecs

    # Delete the indices that had problems
    if len(self.singular_indices) > 0:
      self.nonsingular_indices = \
        list(set(range(self.est.nposes)) - self.singular_indices)
      self.time_axis = np.delete(self.time_axis_orig, list(self.singular_indices))
      if self.point_cloud_registration=="teaser":
        self.inliers = list(set(self.inliers_orig) - self.singular_indices)
        self.outliers = list(set(self.outliers_orig) - self.singular_indices)
      print("Warning: {} indices had singular covariances\n {}".format(
        len(self.singular_indices), self.singular_indices))
    else:
      self.nonsingular_indices = range(self.est.nposes)


  def compute_sample_cov(self, unbiased=True):
    # loop over all timesteps where we can compute the sample range
    for ind in range(self.half_window_size,
                     self.est.nposes-self.half_window_size):
      samp_covWTV = np.zeros((9,9))

      # sum up errors
      for j in range(ind-self.half_window_size, ind+self.half_window_size):
        WTV_error = np.hstack((self.Wsb_error[:,j],
                               self.Tsb_error[:,j],
                               self.Vsb_error[:,j]))
        samp_covWTV += np.outer(WTV_error, WTV_error)

      if unbiased:
        samp_covWTV = samp_covWTV / (self.sample_cov_window_size - 1)
      else:
        samp_covWTV = samp_covWTV / self.sample_cov_window_size

      self.sample_covWTV[:,:,ind] = samp_covWTV


  def write_sample_cov_to_dataset(self, output_filename):

    first_ind = self.half_window_size
    last_ind = self.est.nposes - self.half_window_size

    for ind in range(self.est.nposes):
      if (ind >= first_ind) and (ind < last_ind):
        self.est.assign_val(ind, "has_sample_cov", True)
        self.est.assign_val(ind, "sample_cov_WTV",
          upper_triangular_list(self.sample_covWTV[:,:,ind]))
      else:
        self.est.assign_val(ind, "has_sample_cov", False)
        self.est.assign_val(ind, "sample_cov_WTV", [])

    self.est.add_param("sample_cov_window", self.sample_cov_window_size)

    self.est.write_json(output_filename)


  def plot_trajectories(self, npts=-1):
    plot_3d_trajectories(self.est.Tsb, self.Tsb_gt)


  def get_sigmas(self, cov_type, use_outliers=False):
    assert(self.cov_type=="all" or (cov_type in self.cov_list))
    
    if self.point_cloud_registration == "teaser":
      if use_outliers:
        indices = self.outliers
      else:
        indices = self.inliers
    else:
      indices = self.nonsingular_indices

    if cov_type == "all":
      return (self.gsb_sigma[indices],
              self.gsb_sigmaW[indices],
              self.gsb_sigmaT[indices],
              self.gsb_sigmaV[indices],
              self.gsb_sigmaWT[indices])
    elif cov_type == "WTV":
      return self.gsb_sigma[indices]
    elif cov_type == "W":
      return self.gsb_sigmaW[indices]
    elif cov_type == "T":
      return self.gsb_sigmaT[indices]
    elif cov_type == "V":
      return self.gsb_sigmaV[indices]
    elif cov_type == "WT":
      return self.gsb_sigmaWT[indices]


  def get_evec_coord(self, cov_type, use_outliers=False):
    assert(self.cov_type=="all" or (cov_type in self.cov_list))
    
    if self.point_cloud_registration == "teaser":
      if use_outliers:
        indices = self.outliers
      else:
        indices = self.inliers
    else:
      indices = self.nonsingular_indices

    if cov_type == "all":
      return (self.eigvec_coord_WTV[:,indices],
              self.eigvec_coord_W[:,indices],
              self.eigvec_coord_T[:,indices],
              self.eigvec_coord_V[:,indices],
              self.eigvec_coord_WT[:,indices])
    elif cov_type == "WTV":
      return self.eigvec_coord_WTV[:,indices]
    elif cov_type == "W":
      return self.eigvec_coord_W[:,indices]
    elif cov_type == "T":
      return self.eigvec_coord_T[:,indices]
    elif cov_type == "V":
      return self.eigvec_coord_V[:,indices]
    elif cov_type == "WT":
      return self.eigvec_coord_WT[:,indices]


  def chi2_divergence_draws(self, num_draws=30, draw_size=100, hist_nbins=40,
                            int_upper_lim=-1):
    print("\nChi2 Divergence Draws")
    for cov_type in self.cov_list:
      sigmas = self.get_sigmas(cov_type)
      (mean, std) = chi2_div_draws(sigmas.flatten()**2, state_size(cov_type),
        num_draws=num_draws, draw_size=draw_size, hist_nbins=hist_nbins,
        int_upper_lim=int_upper_lim)
      print("{}: mean={}, std={}".format(cov_type, mean, std))


  def compute_chi2_divergences(self, hist_nbins=100, int_upper_lim=-1):
    print("\nChi2 Divergence")
    if self.point_cloud_registration=="teaser":
      inlier_divergences = []
      outlier_divergences = []
      for cov_type in self.cov_list:
        nees_in = self.get_sigmas(cov_type)**2
        nees_out = self.get_sigmas(cov_type, use_outliers=True)**2
        df = state_size(cov_type)

        div_in = chi2_divergence(nees_in, df, hist_nbins=hist_nbins,
                                 int_upper_lim=int_upper_lim)
        div_out = chi2_divergence(nees_out, df, hist_nbins=hist_nbins,
                                  int_upper_lim=int_upper_lim)
        print("{}: inliers {}, outliers {}".format(cov_type, div_in, div_out))
        inlier_divergences.append(div_in)
        outlier_divergences.append(div_out)
    else:
      divergences = []
      for cov_type in self.cov_list:
        nees = self.get_sigmas(cov_type)**2
        div = chi2_divergence(nees,
                              state_size(cov_type),
                              hist_nbins=hist_nbins,
                              int_upper_lim=int_upper_lim)
        print("{}: {}".format(cov_type, div))
        divergences.append(div)
      return divergences


  def chi2_overlays(self, plot_nbins=40, num_line_pts=2000):
    for cov_type in self.cov_list:
      sigmas = self.get_sigmas(cov_type)
      title = r"{} {} $\chi^2$ comparison".format(self.seq, cov_type)
      chi2_overlay(sigmas**2, state_size(cov_type), title,
                   num_line_pts=num_line_pts,
                   plot_nbins=plot_nbins)


  def sigma_histogram(self, plot_nbins=10):
    for cov_type in self.cov_list:
      sigmas = self.get_sigmas(cov_type)
      title = "{} {} sigma histogram".format(self.seq, cov_type)
      if self.point_cloud_registration == "teaser":
        inlier_outlier_sigma_hist(sigmas, title, self.inliers, self.outliers,
          plot_nbins=plot_nbins)
      else:
        sigma_hist(sigmas, title, plot_nbins=plot_nbins)

  def sigma_over_time(self):
    ylabel = r"# $\sigma$"
    for cov_type in self.cov_list:
      sigmas = self.get_sigmas(cov_type)
      title = "{} {} uncertainty estimates".format(self.seq, cov_type)
      if self.point_cloud_registration == "teaser":
        inlier_outlier_time_scatter(self.time_axis, sigmas, self.inliers,
          self.outliers, title=title, ylabel=ylabel,
          xlabel=self.time_axis_label)
      else:
        time_plot(self.time_axis, sigmas, title=title, ylabel=ylabel,
          xlabel=self.time_axis_label)


  def show_state_errors(self):
    print("\nState Errors Display:")
    Tsb_error = self.est.Tsb - self.Tsb_gt
    vel_error = self.est.Vsb - self.Vsb_gt
    rot_error = np.zeros((3,self.est.nposes))
    for ind in range(self.est.nposes):
      R_sb_diff = self.est.Rsb[ind] * self.Rsb_gt[ind].inv()
      rot_error[:,ind] = R_sb_diff.as_rotvec().flatten()

    if self.point_cloud_registration == "teaser":
      inlier_outlier_error_three_plots(self.time_axis, rot_error,
        self.inliers, self.outliers, self.seq, "rotation", "rad?")
      inlier_outlier_error_three_plots(self.time_axis, Tsb_error,
        self.inliers, self.outliers, self.seq, "translation", "m")
      inlier_outlier_error_three_plots(self.time_axis, vel_error,
        self.inliers, self.outliers, self.seq, "velocity", "m/s")
      inlier_outlier_3D_error_cloud(rot_error, self.inliers, self.outliers,
        "Rotation Error Cloud (rad?)")
      inlier_outlier_3D_error_cloud(Tsb_error, self.inliers, self.outliers,
        "Translation Error Cloud (m)")
      inlier_outlier_3D_error_cloud(vel_error, self.inliers, self.outliers,
        "Velocity Error Cloud (m/s)")
    else:
      error_three_plots(self.time_axis, rot_error, self.seq, "rotation", "rad?")
      error_three_plots(self.time_axis, Tsb_error, self.seq, "translation", "m")
      error_three_plots(self.time_axis, vel_error, self.seq, "velocity", "m/s")
      plot_3D_error_cloud(rot_error, "Rotation Error Cloud (rad?)")
      plot_3D_error_cloud(Tsb_error, "Translation Error Cloud (m)")
      plot_3D_error_cloud(vel_error, "Velocity Error Cloud (m/s)")


  def plot_gauge_group(self):
    time_plot(self.time_axis_orig, self.est.gauge_group,
      title="Reference Group ID", xlabel=self.time_axis_label)



if __name__=="__main__":
  args = parser.parse_args()

  # get data files
  estimator_data = get_xivo_output_filename(args.dump, args.dataset, args.seq,
    cam_id=args.cam_id, sen=args.sen)
  gt_data = get_xivo_gt_filename(args.dump, args.dataset, args.seq,
    sen=args.sen)

  if args.point_cloud_registration=="teaser":
    import teaserpp_python as tpp


  # avoid analysis for covariance matrices for which we haven't computed a
  # sample covariance
  if (args.cov_source != "original") and (args.start_ind==0) \
    and (args.end_ind is None):
    adjust_startend_to_samplecov = True
  else:
    adjust_startend_to_samplecov = False
  
  # cov_type can only be "all" for original or sampled
  if args.cov_type == "all":
    if args.cov_source not in [ "original", "sampled" ]:
      raise ValueError("Covariance type can only be 'all' when cov_source" +
        " is 'original' or 'sampled'.")

  # Check that we have required neural network inputs
  if (args.mode=="analysis" and args.cov_source=="neural_net"):
    if args.network_model is None:
      parser.error("specify a neural network file")

  # Setup object
  calib = CovarianceCalibration(args.seq, gt_data, estimator_data,
    three_sigma=False, start_ind=args.start_ind, end_ind=args.end_ind,
    point_cloud_registration=args.point_cloud_registration,
    plot_timesteps=args.plot_timesteps,
    sample_cov_window_size=args.sample_cov_window_size, cov_type=args.cov_type,
    adjust_startend_to_samplecov=adjust_startend_to_samplecov)

  if args.point_cloud_registration == "teaser":
    calib.set_teaser_params(
      cbar2=args.teaser_cbar2,
      noise_bound=args.teaser_noise_bound,
      rotation_cost_threshold=args.teaser_rotation_cost_threshold,
      max_clique_exact_solution=args.teaser_max_clique_exact_solution,
      max_clique_time_limit=args.teaser_max_clique_time_limit,
      rotation_gnc_factor=args.teaser_rotation_gnc_factor,
      rotation_max_iterations=args.teaser_rotation_max_iterations,
      rotation_estimation_algorithm=args.teaser_rotation_estimation_algorithm
    )

  # ground truth alignment
  calib.align_gt_to_est()

  # Compute key quantities
  calib.compute_errors()
  if args.mode == "analysis" and args.cov_source=="neural_net":
    calib.compute_sigmas(cov_source="neural_net",
      network_model=os.path.join(args.network_dump, args.network_model))
  elif args.mode == "analysis":
    calib.compute_sigmas(cov_source=args.cov_source)

  if args.mode == "compute_sample_cov":
    calib.compute_sample_cov()
    calib.write_sample_cov_to_dataset(estimator_data)

  # Plots
  if (args.mode=="analysis") and args.make_plots:
    #calib.sigma_histogram(nbins=args.hist_nbins)
    #calib.sigma_over_time()
    calib.show_state_errors()
    calib.plot_trajectories()
    calib.innovation_plot()
    calib.innovation_psd_plot()
    calib.plot_gauge_group()
    calib.compute_sigma_percentages()
    calib.compute_chi2_divergences(hist_nbins=args.hist_nbins,
                                   int_upper_lim=args.int_upper_lim)
    calib.chi2_divergence_draws(hist_nbins=args.draws_hist_nbins,
                                int_upper_lim=args.int_upper_lim,
                                num_draws=args.num_draws,
                                draw_size=args.draw_size)
    calib.chi2_overlays(plot_nbins=args.plot_nbins)
    plt.show()

