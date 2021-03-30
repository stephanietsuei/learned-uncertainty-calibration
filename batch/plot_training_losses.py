import os, sys
import pickle
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from constants import EVILOCARINA_DNNDUMP, KIF_DNNDUMP

cov_types = [ "WTV", "W", "T", "V", "WT" ]
input_modes = [ "cov", "gsbcov", "gsbvcov" ]


# ICRA 2021 Submission (Only works for WTV)
best_networks = {
  "WTV": {
    "cov": "Dense_tumvi_cov_2020-10-20-22-41_gpu1",
    "gsbcov": "Dense_tumvi_gsbcov_2020-10-20-22-57_gpu0",
    "gsbvcov": "Dense_tumvi_gsbvcov_2020-10-20-22-59_gpu0",
  },
}

window_sizes = {
  "WTV": 275,
}



# NeuRIPS 2020 submission
# For nbins=2000, with a test split
#best_networks = {
#  "WTV": {
#    "cov": "Dense_tumvi_cov_2020-5-25-20-15_gpu0",
#    "gsbcov": "Dense_tumvi_gsbcov_2020-5-25-20-58_gpu1",
#    "gsbvcov": "Dense_tumvi_gsbvcov_2020-5-25-23-29_gpu1",
#  },
#  "W": {
#    "cov": "Dense_tumvi_cov_2020-5-26-3-59_gpu0",
#    "gsbcov": "Dense_tumvi_gsbcov_2020-5-26-5-31_gpu1",
#    "gsbvcov": "Dense_tumvi_gsbvcov_2020-5-26-5-3_gpu1"
#  },
#  "T": {
#    "cov": "Dense_tumvi_cov_2020-5-26-10-53_gpu1",
#    "gsbcov": "Dense_tumvi_gsbcov_2020-5-26-10-10_gpu0",
#    "gsbvcov": "Dense_tumvi_gsbvcov_2020-5-26-9-22_gpu1"
#  },
#  "V": {
#    "cov": "Dense_tumvi_cov_2020-5-26-15-54_gpu0",
#    "gsbcov": "Dense_tumvi_gsbcov_2020-5-26-16-3_gpu0",
#    "gsbvcov": "Dense_tumvi_gsbvcov_2020-5-26-20-49_gpu0"
#  },
#  "WT": {
#    "cov": "Dense_tumvi_cov_2020-5-26-22-29_gpu0",
#    "gsbcov": "Dense_tumvi_gsbcov_2020-5-26-21-11_gpu1",
#    "gsbvcov": "Dense_tumvi_gsbvcov_2020-5-27-0-51_gpu0"
#  }
#}
#
#window_sizes = {
#  "WTV": 283,
#  "W": 213,
#  "T": 597,
#  "V": 105,
#  "WT": 489
#}
#

# Figure with training losses
plt.figure()
max_epochs = 0

for i,cov_type in enumerate(cov_types):

  plt.subplot(len(cov_types),1,i+1)

  for j,input_mode in enumerate(input_modes):
    network_name = best_networks[cov_type][input_mode]
    window_size = window_sizes[cov_type]
    network_folder = os.path.join(os.environ["HOME"], "data", 
      "xivo_tumvi_dnndump_{}_{}".format(window_size, cov_type))

    pklfilename = os.path.join(network_folder, network_name+".pkl")
    with open(pklfilename, "rb") as fid:
      network_params = pickle.load(fid)
    
    if network_params["nepochs"] > max_epochs:
      max_epochs = network_params["nepochs"]

    # plot loss 
    plt.plot(range(network_params["nepochs"]),
             network_params["training_history"]["loss"])
    plt.ylabel(cov_type)


# embellish the plot
for i in range(len(cov_types)):
  plt.subplot(len(cov_types), 1, i+1)
  plt.xlim(0, max_epochs)

  if i < len(cov_types)-1:
    plt.xticks([])
  
  if i==0:
    plt.legend(input_modes)

plt.xlabel("epochs")
plt.suptitle("Training Loss")
plt.show()
