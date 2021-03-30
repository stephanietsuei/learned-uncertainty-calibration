import sys
import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from scipy.special import gamma

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from interpolate_gt import groundtruth_interpolator
from utils import from_upper_triangular_list, cleanup_and_load_json, \
  get_stock_parser, get_xivo_output_filename, get_xivo_gt_filename


parser = get_stock_parser("Plot point cloud of estimated group locations")
parser.add_argument('-plot_deviations', default=True, type=bool,
  help='whether to plot deviations from the sample mean or the exact positions')
parser.add_argument('-max_suggestion', default=25, type=int,
  help='how many feature IDs to suggest for plotting')


class GroupDetections:
  def __init__(self, estimator_results, seq):
    self.seq = seq

    # store data from files
    self.estimator_results = cleanup_and_load_json(estimator_results)

    self.n_timestaps = len(self.estimator_results["data"])

    # data structures for storing results
    self.group_counts = {}
    # map from ID to list of positions
    self.group_Tsb = {}
    # map from ID to list of orientations
    self.group_Rsb = {}
    # map from ID to covariance matrix
    self.group_covariance = {}
    # first col = ID, second col = sample covaraince of position measurements
    self.group_samplecov = {}
    # sample covariance volume
    self.group_samplecov_vol = {}
    # errors from the mean
    self.group_deviation_from_mean = {}

    self.load_data()
    self.compute_sample_mean_and_covariance()


  def load_data(self):
    for data in self.estimator_results["data"]:
      for idx,group_id in enumerate(data["group_ids"]):
        group_cov = \
          from_upper_triangular_list(3, data["group_covs"][21*idx:21*(idx+1)])
        pose_data = np.array(data["group_poses"][7*idx:7*(idx+1)])
        Qsb_xyzw = pose_data[0:4]
        Tsb = pose_data[4:]
        Rsb = Rotation.from_quat(Qsb_xyzw)

        if group_id not in self.group_counts:
          self.group_counts[group_id] = 1
          self.group_Tsb[group_id] = [Tsb]
          self.group_Rsb[group_id] = [Rsb.as_rotvec()]
          self.group_covariance[group_id] = [group_cov]
        else:
          self.group_counts[group_id] += 1
          self.group_Tsb[group_id].append(Tsb)
          self.group_Rsb[group_id].append(Rsb.as_rotvec())
          self.group_covariance[group_id].append(group_cov)

    # Convert positions list into one big Nx3 numpy array for convenience
    for group_id, Tsb_list in self.group_Tsb.items():
      self.group_Tsb[group_id] = np.array(Tsb_list)
    for group_id, Rsb_list in self.group_Rsb.items():
      self.group_Rsb[group_id] = np.array(Rsb_list)
    


  def plot_group_life_distribution(self):
    group_lifetimes = [ self.group_counts[group_id] for group_id
      in self.group_counts.keys() ]
    group_lifetimes = np.array(group_lifetimes)
    mean_lifetime = np.mean(group_lifetimes)

    plt.figure()
    plt.hist(group_lifetimes, 100)
    plt.xlabel('Length of Group Lifetime')
    plt.ylabel('# Groups')
    plt.title('Group Lifetime Distribution: mean={:4.3e}'.format(mean_lifetime))


  def compute_sample_mean_and_covariance(self):
    # compute sample covariance and volume
    for (group_id, Tsb_list) in self.group_Tsb.items():
      try:
        # Get mean position and deviations
        num_positions = Tsb_list.shape[0]
        mean_Tsb = np.mean(Tsb_list, axis=0)
        Tsb_deviations = Tsb_list - np.tile(mean_Tsb, (num_positions, 1))

        Rsb_nparrays = self.group_Rsb[group_id]
        rot = Rotation.from_rotvec(Rsb_nparrays)
        mean_Rsb = rot.mean()
        Rsb_deviations = rot * mean_Rsb.inv()
        Rsb_deviations = Rsb_deviations.as_rotvec()

        deviations = np.hstack((Rsb_deviations, Tsb_deviations))

        sample_cov = np.zeros((6,6))
        for deviation in deviations:
          sample_cov += np.outer(deviation, deviation)
        sample_cov = sample_cov / num_positions

        (_, Svals, _) = np.linalg.svd(sample_cov)
        sample_cov_volume =  2/6 * np.power(np.pi,3) / gamma(3) * np.prod(Svals)

        self.group_deviation_from_mean[group_id] = deviations
        self.group_samplecov[group_id] = sample_cov
        self.group_samplecov_vol[group_id] = sample_cov_volume
      except np.linalg.LinAlgError:
        print("LinAlgError: Skipping group {}".format(group_id))




  def plot_pts(self, group_id, plot_deviations=True):

    if plot_deviations:
      values = self.group_deviation_from_mean[group_id]
    else:
      values = self.group_Tsb[group_id]

    fig = plt.figure()
    plt.suptitle("Scatter plot of group {} in sequence {}: {} pts, {:4.3e} sample cov vol".format(
      group_id, self.seq, self.group_counts[group_id],
      self.group_samplecov_vol[group_id]))

    # Rsb values
    plt.subplot(2,4,1)
    plt.scatter(values[:,0],
                values[:,1])
    plt.xlabel("Rsb x")
    plt.ylabel("Rsb y")

    plt.subplot(2,4,2)
    plt.scatter(values[:,1],
                values[:,2])
    plt.xlabel("Rsb y")
    plt.ylabel("Rsb z")

    plt.subplot(2,4,3)
    plt.scatter(values[:,0],
                values[:,2])
    plt.xlabel("Rsb x")
    plt.ylabel("Rsb z")

    ax = fig.add_subplot(244,projection='3d')
    ax.scatter(values[:,0],
               values[:,1],
               values[:,2])
    ax.set_xlabel("Rsb x")
    ax.set_ylabel("Rsb y")
    ax.set_zlabel("Rsb z")

    # Tsb values
    plt.subplot(2,4,5)
    plt.scatter(values[:,3],
                values[:,4])
    plt.xlabel("Tsb x")
    plt.ylabel("Tsb y")

    plt.subplot(2,4,6)
    plt.scatter(values[:,4],
                values[:,5])
    plt.xlabel("Tsb y")
    plt.ylabel("Tsb z")

    plt.subplot(2,4,7)
    plt.scatter(values[:,3],
                values[:,5])
    plt.xlabel("Tsb x")
    plt.ylabel("Tsb z")

    ax = fig.add_subplot(248,projection='3d')
    ax.scatter(values[:,3],
               values[:,4],
               values[:,5])
    ax.set_xlabel("Tsb x")
    ax.set_ylabel("Tsb y")
    ax.set_zlabel("Tsb z")



  def user_select_plot(self, max_groups, reverse_lists=False,
    plot_deviations=True):

    # sort group IDs by sample covariance volume  (ascending)
    sorted_IDs_covvolume = [ group_id for group_id, v in
      sorted(self.group_samplecov_vol.items(),
        key=lambda item: item[1],
        reverse=reverse_lists) ]

    # sort group IDs by number of instances (descending)
    sorted_IDs_numbers = [ group_id for group_id, v in
      sorted(self.group_counts.items(),
        key=lambda item: item[1],
        reverse=(not reverse_lists))]

    # print first max_groups
    print("IDs sorted by covariance volume: {}".format(
      sorted_IDs_covvolume[:max_groups]))
    print("IDs sorted by numbers: {}".format(
      sorted_IDs_numbers[:max_groups]))

    # Prompt user to select one to plot
    while True:
      group_id = input("Select an ID: ")
      group_id = int(group_id)
      self.plot_pts(group_id, plot_deviations=plot_deviations)
      plt.show()


if __name__=="__main__":
  args = parser.parse_args()

  estimator_data = get_xivo_output_filename(args.dump, args.dataset, args.seq,
    cam_id=args.cam_id, sen=args.sen)
  
  GD = GroupDetections(estimator_data, args.seq)
  GD.plot_group_life_distribution()
  GD.user_select_plot(args.max_suggestion,
    plot_deviations=args.plot_deviations)
