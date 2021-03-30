import argparse
import copy
import numpy as np
import control


def compute_steadystate_cov(A, C, state_noise_cov, meas_noise_cov):
  (X,_,_) = control.dare(A, C.T, state_noise_cov, meas_noise_cov)
  return X


def generate_covariance(dim, diag_multiplier):
  cov_sqrt = np.random.randn(dim, dim)
  cov = cov_sqrt @ cov_sqrt.T
  for i in range(dim):
    cov[i,i] *= diag_multiplier
  return cov



parser = argparse.ArgumentParser()
parser.add_argument("-state_dim", default=4, type=int)
parser.add_argument("-dynamics", default="random_stable",
  help="[random_stable|integrator_chain]")
parser.add_argument("-meas_cov_diag_mult", default=15, type=float)
parser.add_argument("-dyn_cov_diag_mult", default=20, type=float)
parser.add_argument("-perturbation", default="dyn_cov",
  help="[dyn_cov|meas_cov|A|C|all]")

args = parser.parse_args()
state_dim = args.state_dim


# Generate random LQR-stabilized dynamics matrix - generate until we are both
# controllable and observable
if args.dynamics == "random_stable":
  input_dim = 2
  B = np.zeros((state_dim, input_dim))
  B[0,0] = 1
  B[1,1] = 1
  C = np.transpose(B)

  found = False
  while not found:
    A_random = np.random.rand(state_dim, state_dim)
    ctrb_mat = control.ctrb(A_random, B)
    obsv_mat = control.obsv(A_random, C)

    rank_ctrb = np.linalg.matrix_rank(ctrb_mat)
    rank_obsv = np.linalg.matrix_rank(obsv_mat)
    if rank_ctrb==state_dim and rank_obsv==state_dim:
      found = True

  Qstate = 10*np.eye(state_dim)
  Qinput = 1000*np.eye(input_dim)
  (_, L, Klqr) = control.dare(A_random, B, Qstate, Qinput)
  for eigval in L:
    assert(np.abs(eigval) < 1)

  A = A_random - B @ Klqr

# or just use the standard integrator chain
elif args.dynamics == "integrator_chain":
  dt = 0.05
  A = np.eye(state_dim)
  for i in range(state_dim-1):
    A[i,i+1] = dt
  B = np.zeros((state_dim,1))
  B[-1,0] = 1
  C = np.transpose(B)

else:
  print("unrecognized dynamics")


# Generate random noise covariances - assume we observe the whole state
# make the diagonals 15x bigger than the
meas_dim = C.shape[0]
meas_noise_cov = generate_covariance(meas_dim, args.meas_cov_diag_mult)
dyn_noise_cov = generate_covariance(state_dim, args.dyn_cov_diag_mult)


# Make minor adjustments to things
A1 = copy.copy(A)
C1 = copy.copy(C)
meas_noise_cov1 = copy.copy(meas_noise_cov)
dyn_noise_cov1 = copy.copy(dyn_noise_cov)

if (args.perturbation=="dyn_cov") or (args.perturbation=="all"):
  #perturb = 0.1 * generate_covariance(state_dim, 1)
  #dyn_noise_cov1 += perturb
  dyn_noise_cov1 *= 2.0
if (args.perturbation=="meas_cov") or (args.perturbation=="all"):
  #perturb = 0.1 * generate_covariance(meas_dim, 1)
  #meas_noise_cov1 += perturb
  meas_noise_cov1 *= 2.0
if (args.perturbation=="A") or (args.perturbation=="all"):
  perturb = 0.1 * np.random.randn(state_dim, state_dim)
  A1 += perturb
if (args.perturbation=="C") or (args.perturbation=="all"):
  perturb = 0.1 * np.random.randn(meas_dim, state_dim)
  C1 += perturb


# Compute two steady-state covariances
Sigma = compute_steadystate_cov(A, C, dyn_noise_cov, meas_noise_cov) # the actual
Sigma1 = compute_steadystate_cov(A1, C1, dyn_noise_cov1, meas_noise_cov1) # estimated

print("Where KF converges to (nominal values): ")
print(Sigma)
print("")
print("Where KF should converge to (perturbed values): ")
print(Sigma1)

