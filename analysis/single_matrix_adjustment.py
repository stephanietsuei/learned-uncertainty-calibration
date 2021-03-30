import sys, os, argparse
import numpy as np
import scipy

from ipopt import minimize_ipopt

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from estimator_data import EstimatorData
from utils import from_upper_triangular_list, state_indices, \
  compute_scaled_eigvecs, get_stock_parser, get_xivo_output_filename, \
  idx_to_state


def check_grad(f, df, x0, delta=1e-6, thresh=1e-4):
  f0 = f(x0)
  gradient = df(x0)

  for i in range(cov_dim):
    for j in range(cov_dim):
      ind = i*cov_dim + j
      x1 = x0
      x1[ind] += delta
      f1 = f(x1)
      gradient_approx = (f1 - f0) / delta
      error = np.abs(gradient_approx-gradient[ind])
      if error > thresh:
        print("Gradient {},{} error is {}".format(i,j, error))




def cost_fcn(x):
  A = np.reshape(x, (cov_dim,cov_dim))
  obj = 0
  for est in estimators:
    for k in range(est.nposes):
      Pest = est.P[ind_l:ind_u, ind_l:ind_u, k]
      P = est.sample_covWTV[ind_l:ind_u, ind_l:ind_u, k]
      diff = (A @ Pest @ A.T) - P
      diff_sq = diff * diff

      for i in range(cov_dim):
        for j in range(i,cov_dim):
          if i==j:
            letter = idx_to_state(i+ind_l)
            weightij = weights["weight"+letter]
          else:
            letter1 = idx_to_state(i+ind_l)
            letter2 = idx_to_state(j+ind_l)
            weightij = weights["weight"+letter1+letter2]
          obj += weightij * diff_sq[i,j]
    
  return obj


def gradient_fcn(x):
  A = np.reshape(x, (cov_dim,cov_dim))
  grad = np.zeros((cov_dim,cov_dim))

  # This function is coded to match my notes as closely as possible, and not
  # necessarily in the most efficient way
  for est in estimators:
    for k in range(est.nposes):
      Pest = est.P[ind_l:ind_u, ind_l:ind_u, k]
      P = est.sample_covWTV[ind_l:ind_u, ind_l:ind_u, k]
      
      P_adj = A @ Pest @ A.T

      for i in range(cov_dim):
        for j in range(i,cov_dim):
          gijk = P_adj[i,j] - P[i,j]

          if j > i:
            # grad w.r.t a_id
            for d in range(cov_dim):
              #dgijk_daid = Pest[d,:].dot(A[j,:])
              dgijk_daid = 0
              for m in range(cov_dim):
                dgijk_daid += Pest[d,m]*A[j,m]
              grad[i,d] += 2 * gijk * dgijk_daid
            # grad w.r.t a_jd
            for d in range(cov_dim):
              #dgijk_dajd = Pest[:,d].dot(A[i,:])
              dgijk_dajd = 0
              for l in range(cov_dim):
                dgijk_dajd += Pest[l,d]*A[i,l]
              grad[j,d] += 2 * gijk * dgijk_dajd
          elif j==i:
            # grad w.r.t a_ii
            dgii_daii = 2*Pest[i,i]*A[i,i]
            for m in range(cov_dim):
              if m != i:
                dgii_daii += Pest[i,m]*A[i,m]
            for l in range(cov_dim):
              if l != i:
                dgii_daii += Pest[l,i]*A[i,l]
            grad[i,i] += 2 * gijk * dgii_daii

            # grad w.r.t a_id
            for d in range(cov_dim):
              dgii_daid = Pest[i,d]*A[i,i] + Pest[d,i]*A[i,i]
              for m in range(cov_dim):
                if (m!=i and m!=d):
                  dgii_daid += Pest[d,m]*A[i,m]
              for l in range(cov_dim):
                if (l!=i and l!=d):
                  dgii_daid += Pest[l,d]*A[i,l]
              dgii_daid += 2*Pest[d,d]*A[i,d]
              grad[i,d] += 2 * gijk * dgii_daid
          else:
            raise ValueError("We're not supposed to be here!")

  grad = np.reshape(grad, (cov_dim*cov_dim,))
  return grad


# Command line arguments
parser = get_stock_parser("Compute a linear adjust to the estimated covariances for a given sequence by solving a NLP")
parser.add_argument("-check_grad", default=False, action="store_true")
parser.add_argument("-algorithm", default="ipopt")
parser.add_argument("-estimate_jac", default=False, action="store_true")
parser.add_argument("-cov_type", default="W")
parser.add_argument("-use_weights", default=False, action="store_true")
args = parser.parse_args()


weights = {}
if args.use_weights:
  weights["weightW"] = 10.
  weights["weightT"] = 10.
  weights["weightV"] = 10.
  weights["weightWW"] = 2.5
  weights["weightTT"] = 2.5
  weights["weightVV"] = 2.5
  weights["weightWT"] = 0.5
  weights["weightWV"] = 0.5
  weights["weightTV"] = 0.5
else:
  weights["weightW"] = 1.
  weights["weightT"] = 1.
  weights["weightV"] = 1.
  weights["weightWW"] = 1.
  weights["weightTT"] = 1.
  weights["weightVV"] = 1.
  weights["weightWT"] = 1.
  weights["weightWV"] = 1.
  weights["weightTV"] = 1.



# Load data
estimator_datafiles = []
estimators = []
test_estimators = []
test_datafiles = []

for cam_id in [0,1]:
  for seq in [ "room1", "room2", "room3", "room4", "room5", "room6" ]:
    estimator_datafile = get_xivo_output_filename(args.dump, args.dataset,
      seq, cam_id=cam_id)
    seq_est = EstimatorData(estimator_datafile, adjust_startend_to_samplecov=True)

    if seq=="room6":
      test_datafiles.append(estimator_datafile)
      test_estimators.append(seq_est)
    else:
      estimator_datafiles.append(estimator_datafile)
      estimators.append(seq_est)



# check gradients
if args.check_grad:
  ind_l = 0
  ind_u = 9
  cov_dim = ind_u - ind_l
  check_grad(cost_fcn, gradient_fcn, np.random.randn(cov_dim*cov_dim))
  input("Press any key to continue")
  """
  err = scipy.optimize.check_grad(cost_fcn, gradient_fcn,
    np.random.randn(cov_dim*cov_dim))
  if err > 1e-6:
    raise ValueError("gradient not happy. Error is {}".format(err))
  else:
    print("Gradient check passed :)")
  """


#for cov_type in cov_types:
print("COV TYPE: {}".format(args.cov_type))

# Parse covariance type
(ind_l, ind_u) = state_indices(args.cov_type)
cov_dim = ind_u - ind_l

# Initial guess
x0 = np.random.randn(cov_dim*cov_dim)

# Solve optimization problem
if args.estimate_jac:
  res = minimize_ipopt(cost_fcn, x0, jac=None)
else:
  res = minimize_ipopt(cost_fcn, x0, jac=gradient_fcn)

# grab results
A = np.reshape(res.x, (cov_dim,cov_dim))
print("A: \n{}".format(A))

# Save the result
for est in estimators:
  est.add_param("linear_cov_scale_{}".format(args.cov_type), A)
for est in test_estimators:
  est.add_param("linear_cov_scale_{}".format(args.cov_type), A)
print("")

# add computed matrices to output file
for i,est in enumerate(estimators):
  est.write_json(estimator_datafiles[i])
for i,est in enumerate(test_estimators):
  est.write_json(test_datafiles[i])