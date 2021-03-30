import sys
import os
import itertools
import argparse

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from constants import KIF_DUMP, KIF_DNNDUMP


parser = argparse.ArgumentParser("neural network batch script")
parser.add_argument("-gpu_id", type=int)
parser.add_argument("-dump", default=KIF_DUMP)
parser.add_argument("-dnndump", default=KIF_DNNDUMP)
parser.add_argument("-cov_type", default="WTV")
args = parser.parse_args()


# Select GPU to use
#gpu_id = sys.argv[1]
gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Covariance type
cov_type = args.cov_type

# Combinations to experiment with
#l2_regs = [ 0.0, 0.001, 0.01, 0.1, 0.5, 1.0 ]
l2_regs = [ 0.001 ]

nepochs = [ 25, 50, 100, 150, 200 ]

if int(gpu_id) == 1:
  hidden_layers_list = [
    [ 512, 512, 256, 256, 128, 64 ],
    [ 128, 128, 128 ],
    [ 1024, 512, 256, 128, 64 ],
    [ 256, 256, 256, 128, 128, 64 ],
  ]
else:
  hidden_layers_list = [
    [ 64, 64, 64 ],
    [ 256, 256, 256, 128, 128 ],
    [ 128, 128, 128, 128, 128 ],
    [ 1024, 1024, 512, 512, 256, 256, 128, 128, 64, 64 ]
  ]

input_modes = [ "cov", "gsbcov", "gsbvcov" ]


weightW = 10
weightT = 10
weightV = 10
weightWW = 2.5
weightTT = 2.5
weightVV = 2.5
weightWT = 0.5
weightWV = 0.5
weightTV = 0.5


# The training
allcombs = itertools.product(l2_regs, nepochs, hidden_layers_list, input_modes)
for item in allcombs:

  (l2reg, nepoch, hidden_layers, input_mode) = item

  hidden_layers = [ str(x) for x in hidden_layers ]

  cmd = "python analysis/dense_layers_adjustment.py \
-dataset tumvi \
-dump {} \
-dnndump {} \
-input_mode {} \
-hidden_layers_width {} \
-nepochs {} \
-l2_reg {} \
-gpu_id {} \
-weightW {} \
-weightT {} \
-weightV {} \
-weightWW {} \
-weightTT {} \
-weightVV {} \
-weightWT {} \
-weightWV {} \
-weightTV {} \
-cov_type {}".format(
    args.dump,
    args.dnndump,
    input_mode,
    " ".join(hidden_layers),
    nepoch,
    l2reg,
    gpu_id,
    weightW,
    weightT,
    weightV,
    weightWW,
    weightTT,
    weightVV,
    weightWT,
    weightWV,
    weightTV,
    cov_type
  )

  os.system(cmd)
