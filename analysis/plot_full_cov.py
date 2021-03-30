import argparse, os, sys
import json
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from estimator_data import EstimatorData
from utils import from_upper_triangular_list, get_stock_parser, \
  get_xivo_output_filename, get_xivo_gt_filename


parser = get_stock_parser("Plots covariance matrix as colormap.")


class CovariancePlot:
  def __init__(self, estimator_results, seq):
    self.seq = seq
    self.est = EstimatorData(estimator_results)

  def plot_cov(self, ind):
    plt.figure()
    plt.title("{} covariance for timestep {}".format(self.seq, ind))
    ax = plt.gca()
    im = ax.imshow(self.est.P[:,:,ind], cmap='RdBu')
    ax.figure.colorbar(im, ax=ax)

  def user_select_plot(self):
    while True:
      print("Index range: {} to {}".format(self.est.start_ind, self.est.end_ind))
      index = int(input("Select a timestep: "))
      self.plot_cov(index)
      plt.show()


if __name__=="__main__":
  args = parser.parse_args()

  estimator_data = get_xivo_output_filename(args.dump, args.dataset,
    args.seq, cam_id=args.cam_id, sen=args.sen)

  CP = CovariancePlot(estimator_data, args.seq)
  CP.user_select_plot()
