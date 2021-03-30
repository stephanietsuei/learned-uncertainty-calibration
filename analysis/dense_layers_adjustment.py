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

from eval_cov_calibration import CovarianceCalibration


parser = argparse.ArgumentParser()
parser.add_argument("-process_text", default=False, action='store_true')
parser.add_argument("-dataset", default="tumvi")
parser.add_argument("-dump", default=EVILOCARINA_DUMP)
parser.add_argument("-dnndump", default=EVILOCARINA_DNNDUMP)
parser.add_argument("-input_mode", default="gsbcov")
parser.add_argument("-hidden_layers_width", default=[128, 128, 128], nargs='+',
  type=int)
parser.add_argument("-nepochs", default=25, type=int)
parser.add_argument("-weightW", default=1.0, type=float)
parser.add_argument("-weightT", default=1.0, type=float)
parser.add_argument("-weightV", default=1.0, type=float)
parser.add_argument("-weightWW", default=1.0, type=float)
parser.add_argument("-weightTT", default=1.0, type=float)
parser.add_argument("-weightVV", default=1.0, type=float)
parser.add_argument("-weightWT", default=1.0, type=float)
parser.add_argument("-weightWV", default=1.0, type=float)
parser.add_argument("-weightTV", default=1.0, type=float)
parser.add_argument("-l2_reg", default=0.0, type=float)
parser.add_argument("-gpu_id", default=0, type=int)
parser.add_argument("-loss_function", default="pd_upper_triangle",
  help="loss function to use [pd_upper_triangle | BuresWasserstein | RiccatiContraction ]")
parser.add_argument("-val_metric", default=None,
  help="metric function [ MeanLossMetric | CovarianceCalibrationMetric ]")
parser.add_argument("-use_validation", default=False, action="store_true")
parser.add_argument("-cov_type", default="WTV", help="WTV|W|T|V|WT")
args = parser.parse_args()



class MeanLossMetric(tf.keras.callbacks.Callback):
  def __init__(self, x_val, y_val, loss):
    super(MeanLossMetric, self).__init__()
    self.x_val = x_val
    self.y_val = y_val
    self.loss_fcn = loss
    self.validation_metrics = []
  
  def on_epoch_end(self, epoch, logs={}):
    y_pred = self.model.predict(self.x_val)
    losses = self.loss_fcn(y_val, y_pred)
    mean_loss = tf.math.reduce_mean(losses)
    self.validation_metrics.append(mean_loss.numpy())
    print("\nEpoch {} mean loss: {}\n".format(epoch+1, mean_loss))



class CovarianceCalibrationMetric(tf.keras.callbacks.Callback):
  """Fun! Use covariance calibration values as a metric. But in order to do
  this, we need to split off a specific sequence from the SLAM dataset as a
  validation set."""

  def __init__(self, seq, dataset, gt_data, estimator_data, x_val):
    super(CovarianceCalibrationMetric, self).__init__()

    self.calib = CovarianceCalibration(seq, gt_data, estimator_data,
      three_sigma=False, start_ind=0, end_ind=None,
      point_cloud_registration='horn', plot_timesteps=False,
      adjust_startend_to_samplecov=True)
    self.calib.align_gt_to_est()
    self.calib.compute_errors()

    self.x_val = x_val
    self.validation_metrics = []


  def on_epoch_end(self, epoch, logs={}):
    print("")
    y_pred = self.model.predict(self.x_val)
    self.calib.compute_sigmas(cov_source="net_output",
                              net_output=np.float64(y_pred),
                              output_maxes=train_output_maxes)
    res = self.calib.compute_chi2_divergences(nbins=2000)
    print("\n")

    self.validation_metrics.append(list(res))


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


def BuresWasserstein_loss(y_true, y_pred):    #  both (N, 45)
  """Bures-Wasserstein distance between two positive definite matrices. This is
  currently not working because the variable l is coming out negative and the
  square root of a negative number is nan. I don't understand why this is
  happening, but think that it is probably because of geometrical reasons that
  I don't understand..."""

  # construct adjuseted covariance
  Q = tf.reshape(y_pred, [-1, cov_dim, cov_dim])
  Qt = tf.transpose(Q, perm=[0,2,1])
  Q_pred = Q @ Qt

  # Constructed true covariance
  Q_true = tf.matmul(y_true, FromTriangleOpTen)      # (N, 81)

  # Reshape them into 3D matrices
  Q_pred = tf.reshape(Q_pred, [-1, cov_dim, cov_dim]) # (N, 9, 9)
  Q_true = tf.reshape(Q_true, [-1, cov_dim, cov_dim]) # (N, 9, 9)

  evals, evecs = tf.linalg.eigh(Q_pred)
  evals_D = tf.linalg.diag(tf.math.sqrt(evals))
  Q_pred_sqrt = evecs @ evals_D @ tf.transpose(evecs, perm=[0,2,1])
  C = Q_pred_sqrt @ Q_true @ Q_pred_sqrt

  l = tf.linalg.trace(Q_pred) + tf.linalg.trace(Q_true) - \
    2*tf.math.sqrt(tf.linalg.trace(C))
  return tf.math.sqrt(l)


def RicattiContraction_loss(y_true, y_pred):
  """I'm fairly certain that this is coded correctly. At the moment, errors
  are coming up at the matrix inversion step, because the network may predict
  something that is positive semidefinite instead of positive definite."""

  # construct adjuseted covariance
  Q = tf.reshape(y_pred, [-1, cov_dim, cov_dim])
  Qt = tf.transpose(Q, perm=[0,2,1])
  Q_pred = Q @ Qt

  # Constructed true covariance
  Q_true = tf.matmul(y_true, FromTriangleOpTen)      # (N, 81)
  Q_true = tf.reshape(Q_true, [-1, cov_dim, cov_dim])

  # Compute eigenvalues and sum
  C = tf.linalg.inv(Q_pred) @ Q_true
  evals, _ = tf.linalg.eigh(C)   # (N, 9)
  tt = tf.math.log(evals)        # (N, 9)
  tt2 = tf.math.square(tt)       # (N, 9)
  ll = tf.math.reduce_sum(tt2, 1) # (N,)
  return tf.math.sqrt(ll)



class XivoTFCovData:
  def __init__(self, root, dataset="tumvi", input_mode="cov",
               use_validation=False):
    self.root = root
    self.dataset = dataset
    self.use_validation = use_validation

    self.train_seqs = []
    self.val_seqs = []
    self.test_seqs = []

    # get training and test sequences
    if self.dataset=="tumvi":
      all_seqs = [ "tumvi_room{}_cam{}".format(i,j) \
        for j in range(2) for i in range(1,7) ]
      if self.use_validation:
        self.train_seqs = all_seqs[:-2]
        self.val_seqs = [all_seqs[-2]]
        self.test_seqs = [all_seqs[-1]]
      else:
        self.train_seqs = all_seqs[:-2]
        self.test_seqs = all_seqs[-2:]
    
    # Fields to save for later
    self.train_input_maxes = []
    self.train_output_maxes = []


  def load_seqs(self, seq_list):
    x = np.zeros((0, input_dim))
    y = np.zeros((0, cov_tri_len))
    for seq in seq_list:
      results_file = os.path.join(self.root, seq)
      (seq_x, seq_y) = self.load_xivo_data(results_file)
      x = np.concatenate((x, seq_x))
      y = np.concatenate((y, seq_y))
    return (x, y)


  def load_data(self):
    (x_train, y_train) = self.load_seqs(self.train_seqs)
    (x_test, y_test) = self.load_seqs(self.test_seqs)
    if self.use_validation:
      (x_val, y_val) = self.load_seqs(self.val_seqs)

    # Scale each dimension of input and output
    if state_dim > 0:
      for i in range(state_dim):
        train_input_abs_max = np.max(np.abs(x_train[:,i]))
        x_train[:,i] = x_train[:,i] / train_input_abs_max
        x_test[:,i] = x_test[:,i] / train_input_abs_max
        if self.use_validation:
          x_val[:,i] = x_val[:,i] / train_input_abs_max
        self.train_input_maxes.append(train_input_abs_max)

    for i in range(cov_tri_len):
      train_input_abs_max = np.max(np.abs(x_train[:,state_dim+i]))
      x_train[:,state_dim+i] = \
        x_train[:,state_dim+i] / train_input_abs_max
      x_test[:,state_dim+i] = \
        x_test[:,state_dim+i] / train_input_abs_max
      if self.use_validation:
        x_val[:,state_dim+i] = \
          x_val[:,state_dim+i] / train_input_abs_max
      self.train_input_maxes.append(train_input_abs_max)

      train_output_abs_max = np.max(np.abs(y_train[:,i]))
      y_train[:,i] = y_train[:,i] / train_output_abs_max
      y_test[:,i] = y_test[:,i] / train_output_abs_max
      if self.use_validation:
        y_val[:,i] = y_val[:,i] / train_output_abs_max
      self.train_output_maxes.append(train_output_abs_max)

    # Convert to single-precision to make GPUs happy
    x_train = np.float32(x_train)
    y_train = np.float32(y_train)
    x_test = np.float32(x_test)
    y_test = np.float32(y_test) 
    if self.use_validation:
      x_val = np.float32(x_val)
      y_val = np.float32(y_val)
    else:
      x_val = None
      y_val = None

    return ((x_train, y_train), (x_val, y_val), (x_test, y_test))


  def load_xivo_data(self, results_file, return_est=False):
    est = EstimatorData(results_file, adjust_startend_to_samplecov=True)

    all_x = np.zeros((est.nposes, input_dim))
    all_y = np.zeros((est.nposes, cov_tri_len))

    for i in range(est.nposes):
      P = est.sample_covWTV[ind_l:ind_u, ind_l:ind_u, i]
      P_triangle = upper_triangular_list(P, return_numpy=True, ret_dim=False)
      all_y[i,:] = P_triangle

      Pest = est.P[ind_l:ind_u, ind_l:ind_u, i]
      Pest_triangle = upper_triangular_list(Pest, return_numpy=True, ret_dim=False)

      if input_mode=="cov":
        all_x[i,:] = Pest_triangle
      elif input_mode=="gsbcov":
        x = np.concatenate((
          est.Rsb[i].as_rotvec(),
          est.Tsb[:,i].tolist(),
          Pest_triangle
        ))
        all_x[i,:] = x
      elif input_mode=="gsbvcov":
        x = np.concatenate((
          est.Rsb[i].as_rotvec(),
          est.Tsb[:,i].tolist(),
          est.Vsb[:,i].tolist(),
          Pest_triangle
        ))
        all_x[i,:] = x
      else:
        raise ValueError("Invalid input mode")

    if return_est:
      return (all_x, all_y, est)
    else:
      return (all_x, all_y)



# Parse loss function
if args.loss_function == "pd_upper_triangle":
  loss_fcn = pd_upper_triangle_loss
elif args.loss_function == "BuresWasserstein":
  loss_fcn = BuresWasserstein_loss
elif args.loss_function == "RiccatiContraction":
  loss_fcn = RicattiContraction_loss
else:
  raise ValueError("Invalid loss function")


# Are we using validation data?
if args.val_metric is not None:
  use_validation = True
else:
  use_validation = args.use_validation


# make output directory
if not os.path.exists(args.dnndump):
  os.makedirs(args.dnndump)


# Some global variables involving sizes
if args.cov_type == "WTV":
  cov_dim = 9
  ind_l = 0
  ind_u = 9
elif args.cov_type == "W":
  cov_dim = 3
  ind_l = 0
  ind_u = 3
elif args.cov_type == "T":
  cov_dim = 3
  ind_l = 3
  ind_u = 6
elif args.cov_type == "V":
  cov_dim = 3
  ind_l = 6
  ind_u = 9
elif args.cov_type == "WT":
  cov_dim = 6
  ind_l = 0
  ind_u = 6
else:
  raise ValueError("Invalid type of covariance to predict")
output_dim = cov_dim*cov_dim
input_mode = args.input_mode
cov_tri_len = int(cov_dim*(cov_dim+1) / 2)
if input_mode=="cov":
  state_dim = 0
elif input_mode=="gsbcov":
  state_dim = 6
elif input_mode=="gsbvcov":
  state_dim = 9
else:
  raise ValueError("Invalid input mode")
input_dim = state_dim + cov_tri_len


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

# Weights
state_weights = np.zeros((cov_tri_len,1))
idx=0
for i in range(cov_dim):
  for j in range(i,cov_dim):
    if i==j:
      letter = idx_to_state(i+ind_l)
      state_weights[idx,0] = getattr(args, "weight" + letter)
    else:
      letter1 = idx_to_state(i+ind_l)
      letter2 = idx_to_state(j+ind_l)
      state_weights[idx,0] = getattr(args, "weight" + letter1 + letter2)
    idx += 1
StateWeightTen = tf.constant(state_weights, dtype=tf.float32)


# Load the data
print("Loading data...")
if use_validation:
  pklname = os.path.join(args.dump, 'xivo_{}_{}_{}_val.pkl'.format(
    args.dataset, args.input_mode, args.cov_type))
else:
  pklname = os.path.join(args.dump, 'xivo_{}_{}_{}.pkl'.format(
    args.dataset, args.input_mode, args.cov_type))

if args.process_text:
  xivo_data = XivoTFCovData(args.dump, dataset=args.dataset, 
    input_mode=args.input_mode, use_validation=use_validation)
  ((x_train,y_train), (x_val,y_val), (x_test,y_test)) = xivo_data.load_data()
  train_input_maxes = xivo_data.train_input_maxes
  train_output_maxes = xivo_data.train_output_maxes

  if use_validation:
    val_camid = int(xivo_data.val_seqs[0][-1])
    val_seq = xivo_data.val_seqs[0][6:11]
  else:
    val_camid = None
    val_seq = None

  with open(pklname, 'wb') as fid:
    pickle.dump({
      'x_train': x_train,
      'y_train': y_train,
      'x_val': x_val,
      'y_val': y_val,
      'x_test': x_test,
      'y_test': y_test,
      'train_input_maxes': xivo_data.train_input_maxes,
      'train_output_maxes': xivo_data.train_output_maxes,
      'val_camid': val_camid,
      'val_seq': val_seq
    }, fid)
else:
  with open(pklname, 'rb') as fid:
    data = pickle.load(fid)
    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    y_val = data['y_val']
    x_test = data['x_test']
    y_test = data['y_test']
    train_input_maxes = data['train_input_maxes']
    train_output_maxes = data['train_output_maxes']
    val_camid = data['val_camid']
    val_seq = data['val_seq']
if args.process_text:
  sys.exit()

# parse metric
if args.val_metric == "MeanLossMetric":
  validation_metric = MeanLossMetric(x_val, y_val, loss_fcn)
elif args.val_metric == "CovarianceCalibrationMetric":
  gt_data = get_xivo_gt_filename(args.dump, "tumvi", val_seq)
  estimator_data = get_xivo_output_filename(args.dump, "tumvi", val_seq,
                                            cam_id=val_camid)
  validation_metric = CovarianceCalibrationMetric(val_seq, "tumvi",
    gt_data, estimator_data, x_val)



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
model.compile(optimizer='adam',
              loss=loss_fcn)


# Train and evaluate
print("\nTraining model...")
if use_validation and (args.val_metric is not None):
  history = model.fit(x_train, y_train,
                      epochs=args.nepochs,
                      validation_data=(x_val,y_val),
                      callbacks=[validation_metric])
elif use_validation:
  history = model.fit(x_train, y_train,
                      epochs=args.nepochs,
                      validation_data=(x_val,y_val))
else:
  history = model.fit(x_train, y_train, epochs=args.nepochs)

print("\nTraining done. Evaluating on test...")
results = model.evaluate(x_test, y_test)


# Save the model
print("\nSaving data...")
curr_datetime = datetime.now()
time_str = "{}-{}-{}-{}-{}".format(curr_datetime.year, curr_datetime.month,
  curr_datetime.day, curr_datetime.hour, curr_datetime.minute)
model_filename = "Dense_{}_{}_".format(args.dataset, args.input_mode)
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
  'loss_fcn': args.loss_function,
  'weight_W': args.weightW,
  'weight_T': args.weightT,
  'weight_V': args.weightV,
  'weight_WW': args.weightWW,
  'weight_TT': args.weightTT,
  'weight_VV': args.weightVV,
  'weight_WT': args.weightWT,
  'weight_WV': args.weightWV,
  'weight_TV': args.weightTV,
  'cov_type': args.cov_type,
  'config': model.get_config(),
}
if args.val_metric is not None:
  params['val_metric_name'] = args.val_metric
  params['val_metrics'] = validation_metric.validation_metrics
with open(parameters_filename, "wb") as fid:
  pickle.dump(params, fid)

