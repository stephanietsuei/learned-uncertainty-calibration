import sys, os, argparse
import numpy as np
from scipy.spatial.transform import Rotation
import cvxpy as cvx

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from estimator_data import EstimatorData
from utils import from_upper_triangular_list, state_indices, get_stock_parser, \
  get_xivo_output_filename, idx_to_state


# Command line arguments
parser = get_stock_parser("Solves an optimization problem to compute a single scalar adjustment to the estimated covariances for a single sequence.")
parser.add_argument("-use_weights", default=False, action="store_true")
parser.add_argument("-cov_type", default="WTV")
args = parser.parse_args()

estimator_datafiles = []
estimators = []
test_estimators = []
test_datafiles = []

# Load data
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




# For every combination of covariance, solve an optimization problem for the
# (global) scale factor
cov_type = args.cov_type
print("COV TYPE: {}".format(cov_type))

# Parse covariance type
(ind_l, ind_u) = state_indices(cov_type)
cov_dim = ind_u - ind_l

# Declare decision varianble
scale = cvx.Variable(1)

# Construct quadratic objective function
quad_coef = 0
linear_coef = 0
for est in estimators:
  for ind in range(est.nposes):
    estimated_cov = est.P[ind_l:ind_u, ind_l:ind_u, ind]
    sample_cov = est.sample_covWTV[ind_l:ind_u, ind_l:ind_u, ind]

    for i in range(cov_dim):
      for j in range(i, cov_dim):
        if i==j:
          letter = idx_to_state(i+ind_l)
          weightij = weights["weight"+letter]
        else:
          letter1 = idx_to_state(i+ind_l)
          letter2 = idx_to_state(j+ind_l)
          weightij = weights["weight"+letter1+letter2]
        quad_coef += estimated_cov[i,j]*estimated_cov[i,j] * weightij
        linear_coef += -2*estimated_cov[i,j]*sample_cov[i,j] * weightij

quad_coef = np.array([[quad_coef]])
linear_coef = np.array([linear_coef])

obj_func = cvx.quad_form(scale, quad_coef) + (linear_coef.T @ scale)
obj = cvx.Minimize(obj_func)

# Constraint
constr = [ scale >= 0 ]

# Setup cvxpy problem
prob = cvx.Problem(obj, constr)
obj_value = prob.solve(solver=cvx.GUROBI)

print("Optimal value {}".format(obj_value))
print("Scale: {}".format(scale.value))

# Save the result
for est in estimators:
  est.add_param("scalar_cov_scale_{}".format(cov_type), scale[0].value)
for est in test_estimators:
  est.add_param("scalar_cov_scale_{}".format(cov_type), scale[0].value)

print("")


# Write a new output file
for i,est in enumerate(estimators):
  est.write_json(estimator_datafiles[i])
for i,est in enumerate(test_estimators):
  est.write_json(test_datafiles[i])