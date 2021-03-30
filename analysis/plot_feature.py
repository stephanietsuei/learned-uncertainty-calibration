import sys
import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from interpolate_gt import groundtruth_interpolator
from utils import from_upper_triangular_list, cleanup_and_load_json, \
  get_stock_parser, get_xivo_output_filename, get_xivo_gt_filename


parser = get_stock_parser("Plot point cloud of features")
parser.add_argument('-plot_deviations', default=True, type=bool,
  help='whether to plot deviations from the sample mean or the exact positions')
parser.add_argument('-max_suggestion', default=25, type=int,
  help='how many feature IDs to suggest for plotting')


class FeatureDetections:
  def __init__(self, estimator_results, seq):
    self.seq = seq

    # store data from files
    self.estimator_results = cleanup_and_load_json(estimator_results)

    self.n_timestaps = len(self.estimator_results["data"])

    # data structures for storing results
    self.feature_counts = {}
    # map from ID to list of positions
    self.feature_positions = {}
    # map from ID to covariance matrix
    self.feature_covariance = {}
    # first col = ID, second col = sample covaraince of position measurements
    self.feature_samplecov = {}
    # sample covariance volume
    self.feature_samplecov_vol = {}
    # errors from the mean
    self.feature_deviation_from_mean = {}

    self.load_data()
    self.compute_sample_mean_and_covariance()


  def load_data(self):
    for data in self.estimator_results["data"]:
      for idx,feature_id in enumerate(data["feature_ids"]):
        feature_cov = \
          from_upper_triangular_list(3, data["feature_covs"][6*idx:6*(idx+1)])
        feature_pos = np.array(data["feature_positions"][3*idx:3*(idx+1)])

        if feature_id not in self.feature_counts:
          self.feature_counts[feature_id] = 1
          self.feature_positions[feature_id] = [feature_pos]
          self.feature_covariance[feature_id] = [feature_cov]
        else:
          self.feature_counts[feature_id] += 1
          self.feature_positions[feature_id].append(feature_pos)
          self.feature_covariance[feature_id].append(feature_cov)

    # Convert positions list into one big Nx3 numpy array
    for feature_id, positions_list in self.feature_positions.items():
      self.feature_positions[feature_id] = np.array(positions_list)


  def plot_feature_life_distribution(self):
    feature_lifetimes = [ self.feature_counts[feature_id] for feature_id
      in self.feature_counts.keys() ]
    feature_lifetimes = np.array(feature_lifetimes)
    mean_lifetime = np.mean(feature_lifetimes)

    plt.figure()
    plt.hist(feature_lifetimes, 100)
    plt.xlabel('Length of Feature Lifetime')
    plt.ylabel('# Detected Features')
    plt.title('Feature Lifetime Distribution: mean={:4.3e}'.format(mean_lifetime))


  def compute_sample_mean_and_covariance(self):
    # compute sample covariance and volume
    for (feature_id, positions_list) in self.feature_positions.items():
      try:
        num_positions = positions_list.shape[0]
        mean_pos = np.mean(positions_list, axis=0)
        deviations = positions_list - np.tile(mean_pos, (num_positions, 1))

        sample_cov = np.zeros((3,3))
        for deviation in deviations:
          sample_cov += np.outer(deviation, deviation)
        sample_cov = sample_cov / num_positions

        (_, Svals, _) = np.linalg.svd(sample_cov)
        sample_cov_volume = 4/3*np.pi*np.prod(Svals)

        self.feature_deviation_from_mean[feature_id] = deviations
        self.feature_samplecov[feature_id] = sample_cov
        self.feature_samplecov_vol[feature_id] = sample_cov_volume
      except np.linalg.LinAlgError:
        print("LinAlgError: Skipping feature {}".format(feature_id))



  def plot_pts(self, feature_id, plot_deviations=True):

    if plot_deviations:
      values = self.feature_deviation_from_mean[feature_id]
    else:
      values = self.feature_positions[feature_id]

    fig = plt.figure()
    plt.suptitle("Scatter plot of feature {} in sequence {}: {} pts, {:4.3e} sample cov vol".format(
      feature_id, self.seq, self.feature_counts[feature_id],
      self.feature_samplecov_vol[feature_id]))

    plt.subplot(2,2,1)
    plt.scatter(values[:,0],
                values[:,1])
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(2,2,2)
    plt.scatter(values[:,1],
                values[:,2])
    plt.xlabel("y")
    plt.ylabel("z")

    plt.subplot(2,2,3)
    plt.scatter(values[:,0],
                values[:,2])
    plt.xlabel("x")
    plt.ylabel("z")

    ax = fig.add_subplot(224,projection='3d')
    ax.scatter(values[:,0],
               values[:,1],
               values[:,2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


  def user_select_plot(self, max_features, reverse_lists=False,
    plot_deviations=True):

    # sort feature IDs by sample covariance volume  (ascending)
    sorted_IDs_covvolume = [ feature_id for feature_id, v in
      sorted(self.feature_samplecov_vol.items(),
        key=lambda item: item[1],
        reverse=reverse_lists) ]

    # sort feature IDs by number of instances (descending)
    sorted_IDs_numbers = [ feature_id for feature_id, v in
      sorted(self.feature_counts.items(),
        key=lambda item: item[1],
        reverse=(not reverse_lists))]

    # print first max_features
    print("IDs sorted by covariance volume: {}".format(
      sorted_IDs_covvolume[:max_features]))
    print("IDs sorted by numbers: {}".format(
      sorted_IDs_numbers[:max_features]))

    # Prompt user to select one to plot
    while True:
      feature_id = input("Select an ID: ")
      feature_id = int(feature_id)
      self.plot_pts(feature_id, plot_deviations=plot_deviations)
      plt.show()


if __name__=="__main__":
  args = parser.parse_args()

  estimator_data = get_xivo_output_filename(args.dump, args.dataset, args.seq,
    cam_id=args.cam_id, sen=args.sen)

  FD = FeatureDetections(estimator_data, args.seq)
  FD.plot_feature_life_distribution()
  FD.user_select_plot(args.max_suggestion,
    plot_deviations=args.plot_deviations)
