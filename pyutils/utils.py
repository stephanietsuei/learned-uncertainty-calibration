import re
import os
import json
import numpy as np
import scipy
from scipy.stats import chi2, rv_histogram
import copy
import argparse

from constants import EVILOCARINA_ROOT, EVILOCARINA_DUMP


def get_stock_parser(description, group_desc=None):
  # Parser with default options for dataset
  if group_desc is not None:
    Pparser = argparse.ArgumentParser(description)
    parser = Pparser.add_argument_group(group_desc)
  else:
    parser = argparse.ArgumentParser(description)

  parser.add_argument("-root", default=EVILOCARINA_ROOT,
    help="location of VIO dataset")
  parser.add_argument("-dump", default=EVILOCARINA_DUMP,
    help="location of xivo's output data from a dataset")
  parser.add_argument("-dataset", default="tumvi",
    help="name of a (supported) VIO dataset [tumvi|cosyvio|alphred|xivo]")
  parser.add_argument("-seq", default="room1",
    help="short tag for sequence name")
  parser.add_argument("-cam_id", default=0, type=int,
    help="camera from stereo camera pair (only used for tumvi dataset)")
  parser.add_argument("-sen", default="tango_top",
    help="sensor from which images were captured (only used for cosyvio dataset)")

  if group_desc is not None:
    return copy.copy(Pparser)
  else:
    return copy.copy(parser)


def get_xivo_output_filename(dumpdir, dataset, seq, cam_id=0, sen='tango_top'):
  if dataset=="tumvi":
    estimator_datafile = os.path.join(dumpdir,
      "tumvi_{}_cam{}".format(seq, cam_id))
  elif dataset=="cosyvio":
    estimator_datafile = os.path.join(dumpdir,
      "cosyvio_{}_{}".format(sen, seq))
  else:
    estimator_datafile = os.path.join(dumpdir, "{}_{}".format(dataset, seq))
  return estimator_datafile


def get_xivo_gt_filename(dumpdir, dataset, seq, sen='tango_top'):
  if dataset=="cosyvio":
    gt_data = os.path.join(dumpdir, "cosyvio_{}_{}_gt".format(sen, seq))
  else:
    gt_data = os.path.join(dumpdir, "{}_{}_gt".format(dataset, seq))
  return gt_data


def cleanup_and_load_json(results_filename):
  with open(results_filename, 'r') as fp:
    filestr = ""
    for line in fp:
      sub1 = re.sub('nan', 'NaN', line)
      sub2 = re.sub('inf', 'Infinity', sub1)
      filestr = filestr + sub2
    return json.loads(filestr)


def state_indices(state_portion):
  if state_portion=="W":
    ind_l = 0
    ind_u = 3
  elif state_portion=="T":
    ind_l = 3
    ind_u = 6
  elif state_portion=="V":
    ind_l = 6
    ind_u = 9
  elif state_portion=="WT":
    ind_l = 0
    ind_u = 6
  elif state_portion=="WTV":
    ind_l = 0
    ind_u = 9
  else:
    raise ValueError("invalid state portion")
  return (ind_l, ind_u)


def state_size(state_portion):
  (idx_l, idx_u) = state_indices(state_portion)
  return idx_u - idx_l


def idx_to_state(idx):
  if 0 <= idx <= 2:
    return "W"
  elif 3 <= idx <= 5:
    return "T"
  elif 6 <= idx <= 8:
    return "V"
  else:
    raise ValueError("prolly not looking for this in the analysis")


def upper_triangular_list(square_matrix, return_numpy=False, ret_dim=True):
    """Used on covariance matrices so we're not printing out as many digits. Save the
    upper-triangle of a square matrix as a list of numbers, row-major."""
    dim = np.shape(square_matrix)[0]
    l = []
    for i in range(dim):
        for j in range(i,dim):
            l.append(square_matrix[i][j])
    if return_numpy:
      l = np.array(l)
    if ret_dim:
      return [dim, l]
    else:
      return l


def from_upper_triangular_list(dim, l):
  """reverses function `upper_triangular_list` in file savers.py"""
  mat = np.zeros((dim,dim))
  ind = 0
  for i in range(dim):
    for j in range(i,dim):
      mat[i,j] = l[ind]
      mat[j,i] = mat[i,j]
      ind += 1
  return mat


def scale_covariance_matrix(cov, factor):
  U,S,Vh = np.linalg.svd(cov)
  scaled_S = S / factor
  return U.dot(np.diag(scaled_S).dot(Vh))


def rigid_transform_3d(A, B):
  # Input: expects Nx3 matrix of points
  # Returns R,t
  # R = 3x3 rotation matrix
  # t = 3x1 column vector
  assert len(A) == len(B)

  N = A.shape[0]  # total points
  centroid_A = np.mean(A, axis=0)
  centroid_B = np.mean(B, axis=0)
  # centre the points
  AA = A - np.tile(centroid_A, (N, 1))
  BB = B - np.tile(centroid_B, (N, 1))
  # dot is matrix multiplication for array
  H = np.matmul(np.transpose(AA), BB)
  U, _, Vt = np.linalg.svd(H)
  R = np.matmul(Vt.T, U.T)

  # special reflection case
  if np.linalg.det(R) < 0:
    Vt[2,:] *= -1
    R = np.matmul(Vt.T, U.T)

  t = np.matmul(-R, centroid_A.T) + centroid_B.T

  return R, t


def calc_avg_sampling_freq(time_axis):
  num_pts = len(time_axis)
  dt = np.zeros(num_pts-1)
  for i in range(num_pts-1):
    dt[i] = time_axis[i+1] - time_axis[i]
  avg_dt = np.mean(dt)
  avg_freq = 1 / avg_dt
  return avg_freq


def get_psd_eig(matrix):
  Eigvals,Eigvecs = np.linalg.eig(matrix)
  if (np.sum(np.sign(Eigvals)) == Eigvals.size):
    return (Eigvals,Eigvecs)
  else:
    print("Warning: Covariance matrix is not quite positive definite!")

    new_matrix = scipy.linalg.sqrtm(matrix @ matrix.T)
    (Eigvals2,Eigvecs2) = np.linalg.eig(new_matrix)

    # tweak eigenvalues so that we have positive-definiteness
    for i,eigval in enumerate(Eigvals2):
      if eigval < 1e-6:
        Eigvals2[i] = 1e-6
    return (Eigvals2,Eigvecs2)


def compute_scaled_eigvecs(P, return_scales=False):
  eigvals, eigvecs = np.linalg.eig(P)
  for eigval in eigvals:
    if eigval < 0:
      (eigvals, eigvecs) = get_psd_eig(P)
      break

  singular_vals = np.sqrt(eigvals)
  scaled_eigvecs = eigvecs @ np.diag(singular_vals)
  if return_scales:
    return (scaled_eigvecs, singular_vals)
  else:
    return scaled_eigvecs


def chi2_divergence(nees, df, hist_nbins=2000, dx=0.01, int_upper_lim=-1,
                    chi2_percentile=0.9999, int_max_nsteps=None, BIG=10):


  # get max
  if int_upper_lim < 0:
    int_upper_lim = np.max(nees)

  # Get axis to compute integral
  try:
    # Chi2 vars, then normalize so that area under the 
    hist = np.histogram(nees, bins=hist_nbins)
    hist_dist = rv_histogram(hist)

    xmax = np.max([int_upper_lim, chi2.ppf(chi2_percentile, df)])
    if int_max_nsteps is None:
      x = np.linspace(0, xmax, int(xmax/dx))
    else:
      nsteps = min(int(xmax/dx), int_max_nsteps)
      x = np.linspace(0, xmax, nsteps)
  except:
    print("Warning: Out of Memory error in chi2_divergence. " +
          "Returning default large value.")
    return BIG

  # Get approximate pdf points of histogram
  hist_pdf = hist_dist.pdf(x)

  # Get the chi-2 pdf
  rv = chi2(df)
  chi2_pdf = rv.pdf(x)

  # Compute squared difference
  diff = chi2_pdf - hist_pdf
  diff_sq = diff**2

  # Integrate the squared difference and return
  divergence = np.sqrt(np.trapz(diff_sq, x))
  return divergence


def chi2_div_draws(nees, df, num_draws=30, draw_size=100, hist_nbins=100,
  int_upper_lim=-1):
  sample_divs = np.zeros(num_draws)
  for i in range(num_draws):
    nees_sample = np.random.choice(nees, draw_size, replace=False)
    sample_divs[i] = chi2_divergence(nees_sample, df, hist_nbins=hist_nbins,
      int_upper_lim=int_upper_lim)
  return (np.mean(sample_divs), np.std(sample_divs))
