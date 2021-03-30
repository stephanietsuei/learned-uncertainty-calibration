import os, sys, argparse
from datetime import datetime
import pickle

import numpy as np
from scipy.linalg import block_diag
import tensorflow as tf

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from estimator_data import EstimatorData
from utils import state_indices, upper_triangular_list, idx_to_state, \
  get_xivo_gt_filename, get_xivo_output_filename
from constants import EVILOCARINA_DUMP, EVILOCARINA_DNNDUMP


parser = argparse.ArgumentParser()
parser.add_argument("-data", default="ekf_data.pkl")
parser.add_argument("-dnndump", default="/home/stephanie/data/ekf_dnns")
parser.add_argument("-input_mode", default="cov", help="cov | statecov ")
parser.add_argument("-hidden_layers_width", default=[128, 128, 128], nargs="+",
  type=int)
parser.add_argument("-nepochs", default=25, type=int)
parser.add_argument("-l2_reg", default=0.0, type=float)
parser.add_argument("-gpu_id", default=0, type=int)
parser.add_argument("-use_weights", default=False, action="store_true")
args = parser.parse_args()


def pd_upper_triangle_loss(y_true, y_pred):
  """y_true is the upper triangle of the true covariance. y_pred is the nxn
  square root of the adjusted covariance. Also, we know that cov_dim=9."""

  # construct adjusted covariance
  Q = tf.reshape(y_pred, [-1, cov_dim, cov_dim])
  Qt = tf.transpose(Q, perm=[0,2,1])
  P_adj = Q @ Qt

  # Flatten each matrix, then use multiplication to take upper triangle
  P_adj = tf.reshape(P_adj, [-1, cov_dim*cov_dim,])     # (N, 81)
  P_adj_tri = tf.matmul(TriangleOpTen, tf.transpose(P_adj))     # (45, N)
  y_pred_tri = tf.transpose(P_adj_tri)                          # (N, 45)
  diff_sq = tf.math.square(y_pred_tri - y_true)                 # (N, 45)

  diff_sq_weighted = tf.matmul(diff_sq, StateWeightTen)         # (N, 1)

  # Weight different parts of the state
  return tf.reshape(diff_sq_weighted, [-1,])


# Make output directory
if not os.path.exists(args.dnndump):
  os.makedirs(args.dnndump)

# Load data
datafile = args.data
with open(datafile, "rb") as fid:
  data_dict = pickle.load(fid)

if args.input_mode == "cov":
  x_train = data_dict['x_train_cov']
  x_test = data_dict['x_test_cov']
elif args.input_mode == "statecov":
  x_train = data_dict['x_train_statecov']
  x_test = data_dict['x_test_statecov']
y_train = data_dict['y_train']
y_test = data_dict['y_test']

n_train_pts = x_train.shape[0]
n_test_pts = x_test.shape[0]


# dimensions
state_dim = 4 
cov_dim = state_dim
cov_tri_len = int(cov_dim*(cov_dim+1) / 2)
output_dim = cov_dim*cov_dim
if args.input_mode == "cov":
  input_dim = cov_tri_len
elif args.input_mode == "statecov":
  input_dim = state_dim + cov_tri_len



# Scale inputs and outputs
train_input_maxes = np.max(np.abs(x_train), axis=0)
train_output_maxes = np.max(np.abs(y_train), axis=0)
x_train = x_train / np.tile(train_input_maxes, (n_train_pts,1))
y_train = y_train / np.tile(train_output_maxes, (n_train_pts,1))
x_test = x_test / np.tile(train_input_maxes, (n_test_pts,1))
y_test = y_test / np.tile(train_output_maxes, (n_test_pts,1))


# Operation for getting right matrix for triangular fill
blks = []
for i in range(cov_dim):
  zeros_blk = np.zeros((cov_dim-i,i))
  eye_blk = np.eye(cov_dim-i)
  blk = np.hstack((zeros_blk, eye_blk))
  blks.append(blk)
TriangleOp = block_diag(*blks)
TriangleOpTen = tf.constant(TriangleOp, dtype=tf.float32)


# Operation for mapping a upper-triangular vector to a symmetric matrix
FromTriangleOp = np.zeros((cov_dim*cov_dim, cov_tri_len))
idx=0
for i in range(cov_dim):
  for j in range(i,cov_dim):
    idx_triu = i*cov_dim + j
    idx_tril = j*cov_dim + i
    FromTriangleOp[idx_triu,idx] = 1.0
    FromTriangleOp[idx_tril,idx] = 1.0
    idx += 1
FromTriangleOp = FromTriangleOp.T  # (45, 81)
FromTriangleOpTen = tf.constant(FromTriangleOp, dtype=tf.float32)

# (Equal weights on all the states)
state_weights = np.ones((cov_tri_len,1))
if args.use_weights:
  idx = 0
  for i in range(cov_dim):
    for j in range(i,cov_dim):
      if i == j:
        state_weights[idx] = 5
      idx += 1
StateWeightTen = tf.constant(state_weights, dtype=tf.float32)

# loss function
loss_fcn = pd_upper_triangle_loss

# define the model
print("Defining model...")
layers = [ tf.keras.layers.Dense(
  args.hidden_layers_width[0],
  activation='relu',
  input_dim=input_dim,
  kernel_regularizer=tf.keras.regularizers.l2(args.l2_reg),
  bias_regularizer=tf.keras.regularizers.l2(args.l2_reg)
  ) ]
for width in args.hidden_layers_width[1:]:
  layers.append(tf.keras.layers.Dense(
    width,
    activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(args.l2_reg),
    bias_regularizer=tf.keras.regularizers.l2(args.l2_reg),
  ))
layers.append(tf.keras.layers.Dense(
  output_dim,
  activation='relu',
  kernel_regularizer=tf.keras.regularizers.l2(args.l2_reg),
  bias_regularizer=tf.keras.regularizers.l2(args.l2_reg)
))

model = tf.keras.models.Sequential(layers)
model.summary()
model.compile(optimizer='adam', loss=loss_fcn)


# Train and evaluate
history = model.fit(x_train, y_train, epochs=args.nepochs)
print("\nTraining done. Evaluating on test...")
results = model.evaluate(x_test, y_test)

# Save the model
print("\nSaving data...")
curr_datetime = datetime.now()
time_str = "{}-{}-{}-{}-{}".format(curr_datetime.year, curr_datetime.month,
  curr_datetime.day, curr_datetime.hour, curr_datetime.minute)
model_filename = "Dense_ekf_{}_".format(args.input_mode)
model_filename = model_filename + time_str + "_gpu" + str(args.gpu_id)
model.save_weights(os.path.join(args.dnndump, model_filename))


# Save the parameters
parameters_filename = os.path.join(args.dnndump, model_filename+".pkl")
params = {
  'hidden_layers': args.hidden_layers_width,
  'input_mode': args.input_mode,
  'nepochs': args.nepochs,
  'training_history': history.history,
  'test_results': results,
  'input_maxes': train_input_maxes,
  'output_maxes': train_output_maxes,
  'input_dim': input_dim,
  'output_dim': output_dim,
  'l2_reg': args.l2_reg,
  'loss_fcn': "pd_upper_triangle",
  'config': model.get_config(),
  'cov_loss_weights': state_weights
}
with open(parameters_filename, "wb") as fid:
  pickle.dump(params, fid)

