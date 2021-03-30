import sys
import os
import itertools
import argparse

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from constants import KIF_DUMP, KIF_DNNDUMP


parser = argparse.ArgumentParser("neural network batch script for vanilla ekf experiment")
parser.add_argument("-gpu_id", type=int)
parser.add_argument("-data", default="ekf_data.pkl")
parser.add_argument("-dnndump", default=KIF_DNNDUMP)
args = parser.parse_args()


# Select GPU to use
#gpu_id = sys.argv[1]
gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Combinations to experiment with
l2_regs = [ 0, 0.0001, 0.001, 0.01, 0.1  ]
#l2_regs = [ 0.001 ]

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

input_modes = [ "cov", "statecov" ]


# The training
allcombs = itertools.product(l2_regs, nepochs, hidden_layers_list, input_modes)
for item in allcombs:

  (l2reg, nepoch, hidden_layers, input_mode) = item

  hidden_layers = [ str(x) for x in hidden_layers ]

  cmd = "python kfexperiments/ekf_dense_layers.py \
-use_weights \
-data {} \
-dnndump {} \
-input_mode {} \
-hidden_layers_width {} \
-nepochs {} \
-l2_reg {} \
-gpu_id {}".format(
    args.data,
    args.dnndump,
    input_mode,
    " ".join(hidden_layers),
    nepoch,
    l2reg,
    gpu_id,
  )

  os.system(cmd)
