import pickle
import glob
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from constants import EVILOCARINA_DNNDUMP, KIF_DNNDUMP

# hack hack... Edit file as needed when updating dense_layers_adjustment.py


#dnndumpdir = EVILOCARINA_DNNDUMP
dnndumpdir = sys.argv[1]
processed_nn_pickles = glob.glob(os.path.join(dnndumpdir, "*.pkl"))

for pklfile in processed_nn_pickles:
  with open(pklfile, "rb") as fid:
    data = pickle.load(fid)

  #data["loss_fcn"] = "pd_upper_triangle"
  data["cov_type"] = "WTV"
  
  with open(pklfile, "wb") as fid:
    pickle.dump(data, fid)

