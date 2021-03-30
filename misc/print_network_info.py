import argparse, os, sys
import pickle
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from constants import EVILOCARINA_DNNDUMP, KIF_DNNDUMP

parser = argparse.ArgumentParser()
parser.add_argument("-network_dump", default=EVILOCARINA_DNNDUMP)
parser.add_argument("-network_models", nargs="+",
  help="Tensorflow models and config")
args = parser.parse_args()

weight_types = [ "W", "T", "V", "WW", "TT", "VV", "WT", "WV", "TV" ]


# Figure with training losses
plt.figure()

max_epochs = 0

for i,model in enumerate(args.network_models):

  pklfilename = os.path.join(args.network_dump, model + ".pkl")
  with open(pklfilename, "rb") as fid:
    network_params = pickle.load(fid)

    print("Hidden Layers: {}".format(network_params["hidden_layers"]))
    print("Input Mode: {}".format(network_params["input_mode"]))
    print("Output Type: {}".format(network_params["cov_type"]))
    print("Epochs: {}".format(network_params["nepochs"]))
    print("L2 Factor: {}".format(network_params["l2_reg"]))

    if network_params["nepochs"] > max_epochs:
      max_epochs = network_params["nepochs"]

    for weight_type in weight_types:
      name = "weight_{}".format(weight_type)
      print("{}: {}".format(name, network_params[name]))

    # Plot loss
    plt.subplot(len(args.network_models),1,i+1)
    plt.plot(range(network_params["nepochs"]),
            network_params["training_history"]["loss"])
    plt.ylabel("Hypothesis {}".format(i+3))

# Set epochs
for i in range(len(args.network_models)):
  plt.subplot(len(args.network_models), 1, i+1)
  plt.xlim(0, max_epochs)

plt.xlabel("epochs")
plt.suptitle("Training Loss")
plt.show()
