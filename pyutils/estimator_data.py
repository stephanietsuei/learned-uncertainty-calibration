import sys, os
import json
import numpy as np
from scipy.spatial.transform import Rotation

from utils import (from_upper_triangular_list, scale_covariance_matrix,
                   state_indices, upper_triangular_list,
                   cleanup_and_load_json)
from to_json import to_json


class EstimatorData:
  def __init__(self, results_file, start_ind=0, end_ind=None,
    adjust_startend_to_samplecov=False):

    self.start_ind = start_ind
    self.end_ind = end_ind

    # open file to do nan/NaN and inf/Infinity replacements
    self.estimator_results = cleanup_and_load_json(results_file)
    
    if adjust_startend_to_samplecov:
      self.set_startend_to_samplecov()

    if self.end_ind is None:
      self.nposes = len(self.estimator_results["data"]) - self.start_ind
      self.end_ind = self.nposes
    else:
      self.nposes = self.end_ind - self.start_ind

    (total_dim,_) = self.estimator_results["data"][0]["Pstate"]
    self.time_axis = np.zeros((self.nposes,1))
    self.Rsb = []
    self.Tsb = np.zeros((3,self.nposes))
    self.Vsb = np.zeros((3,self.nposes))
    self.P = np.zeros((total_dim, total_dim, self.nposes))

    self.MeasurementUpdateInitialized = []
    self.inn_Wsb = np.zeros((3,self.nposes))
    self.inn_Tsb = np.zeros((3,self.nposes))
    self.inn_Vsb = np.zeros((3,self.nposes))

    # lists of lists for the map.
    self.feature_ids = []
    self.feature_positions = []
    self.feature_covariances = []
    self.group_ids = []
    self.group_Tsb = []
    self.group_Rsb = []
    self.group_covariances = []

    self.gauge_group = []

    # sample covariance data
    self.has_sample_cov = []
    self.sample_covWTV = np.zeros((9,9,self.nposes))

    # scalar factors on covariance
    self.cov_scale_W = 1
    self.cov_scale_T = 1
    self.cov_scale_V = 1
    self.cov_scale_WT = 1
    self.cov_scale_WTV = 1

    # Linear adjustment on convariances
    self.cov_A_W = np.eye(3)
    self.cov_A_T = np.eye(3)
    self.cov_A_V = np.eye(3)
    self.cov_A_WT = np.eye(6)
    self.cov_A_WTV = np.eye(9)

    # collect the pose data
    self.collect_data()


  def scale_covariance_matries(self, factor):
    for i in range(self.nposes):
      self.P[:,:,i] = scale_covariance_matrix(self.P[:,:,i], factor)


  def collect_data(self):
    for i in range(self.nposes):
      timestamp, R_sb, T_sb, V_sb, P, inn_Wsb, inn_Tsb, inn_Vsb, gauge_group = \
          self.get_estimator_point(self.start_ind+i)
      self.Rsb.append(R_sb)
      self.Tsb[:,i] = T_sb
      self.Vsb[:,i] = V_sb
      self.P[:,:,i] = P
      self.inn_Wsb[:,i] = inn_Wsb
      self.inn_Tsb[:,i] = inn_Tsb
      self.inn_Vsb[:,i] = inn_Vsb
      self.time_axis[i] = timestamp
      self.gauge_group.append(gauge_group)

      feature_ids, feature_positions, feature_covs = \
          self.collect_feature_data(self.start_ind+i)
      self.feature_ids.append(feature_ids)
      self.feature_positions.append(feature_positions)
      self.feature_covariances.append(feature_covs)

      group_ids, group_Rsb, group_Tsb, group_covs = \
          self.collect_group_data(self.start_ind+i)
      self.group_ids.append(group_ids)
      self.group_Rsb.append(group_Rsb)
      self.group_Tsb.append(group_Tsb)
      self.group_covariances.append(group_covs)

      # read sample covariance
      cov = self.get_sample_cov(self.start_ind+i)
      if cov is not None:
        self.has_sample_cov.append(True)
        self.sample_covWTV[:,:,i] = cov
      else:
        self.has_sample_cov.append(False)

      # scaled covariance values
      if "scalar_cov_scale_W" in self.estimator_results:
        self.cov_scale_W = self.estimator_results["scalar_cov_scale_W"]
      if "scalar_cov_scale_T" in self.estimator_results:
        self.cov_scale_T = self.estimator_results["scalar_cov_scale_T"]
      if "scalar_cov_scale_V" in self.estimator_results:
        self.cov_scale_V = self.estimator_results["scalar_cov_scale_V"]
      if "scalar_cov_scale_WT" in self.estimator_results:
        self.cov_scale_WT = self.estimator_results["scalar_cov_scale_WT"]
      if "scalar_cov_scale_WTV" in self.estimator_results:
        self.cov_scale_WTV = self.estimator_results["scalar_cov_scale_WTV"]

      # linear adjustment to covariance
      if "linear_cov_scale_W" in self.estimator_results:
        self.cov_A_W = np.reshape(
          np.array(self.estimator_results["linear_cov_scale_W"]), (3,3))
      if "linear_cov_scale_T" in self.estimator_results:
        self.cov_A_T = np.reshape(
          np.array(self.estimator_results["linear_cov_scale_T"]), (3,3))
      if "linear_cov_scale_V" in self.estimator_results:
        self.cov_A_V = np.reshape(
          np.array(self.estimator_results["linear_cov_scale_V"]), (3,3))
      if "linear_cov_scale_WT" in self.estimator_results:
        self.cov_A_WT = np.reshape(
          np.array(self.estimator_results["linear_cov_scale_WT"]), (6,6))
      if "linear_cov_scale_WTV" in self.estimator_results:
        self.cov_A_WTV = np.reshape(
          np.array(self.estimator_results["linear_cov_scale_WTV"]), (9,9))


  def get_sample_cov(self, ind):
    data = self.estimator_results["data"][ind]
    if "has_sample_cov" in data:
      if data["has_sample_cov"]:
        (cov_dimWTV, sample_covWTV) = data["sample_cov_WTV"]
        return from_upper_triangular_list(cov_dimWTV, sample_covWTV)
      else:
        return None


  def get_estimator_point(self, ind):
    data = self.estimator_results["data"][ind]

    timestamp = data["Timestamp"]

    # Rotation, translation, velocity
    T_sb = np.array(data["Tsb_XYZ"])
    V_sb = np.array(data["Vsb_XYZ"])
    Q_sb_wxyz = np.array(data["qsb_WXYZ"])
    Q_sb_xyzw = np.array([Q_sb_wxyz[1], Q_sb_wxyz[2], Q_sb_wxyz[3],
      Q_sb_wxyz[0]])
    R_sb = Rotation.from_quat(Q_sb_xyzw)

    # Covariance
    (Pdim, Plist) = data["Pstate"]
    P = from_upper_triangular_list(Pdim, Plist)

    # innovation
    inn_Wsb = np.array(data["inn_Wsb"])
    inn_Tsb = np.array(data["inn_Tsb"])
    inn_Vsb = np.array(data["inn_Vsb"])

    # gauge group
    group = data["group"]

    return (timestamp, R_sb, T_sb, V_sb, P, inn_Wsb, inn_Tsb, inn_Vsb, group)


  def collect_feature_data(self, ind):
    data = self.estimator_results["data"][ind]

    feature_ids = np.array(data["feature_ids"])
    feature_covs = []
    feature_positions = []

    for idx,_ in enumerate(feature_ids):
      cov_data = from_upper_triangular_list(3,
        data["feature_covs"][6*idx:6*(idx+1)])
      feature_pos = np.array(data["feature_positions"][3*idx:3*(idx+1)])

      feature_covs.append(cov_data)
      feature_positions.append(feature_pos)

    return (feature_ids, feature_positions, feature_covs)


  def collect_group_data(self, ind):
    data = self.estimator_results["data"][ind]

    group_ids = np.array(data["group_ids"])
    group_covs = []
    group_Rsb = []
    group_Tsb = []

    for idx,_ in enumerate(group_ids):
      cov_data = from_upper_triangular_list(6,
        data["group_covs"][21*idx:21*(idx+1)])
      pose_data = np.array(data["group_poses"][7*idx:7*(idx+1)])
      Qsb_xyzw = pose_data[0:4]
      Tsb = pose_data[4:]
      Rsb = Rotation.from_quat(Qsb_xyzw)

      group_covs.append(cov_data)
      group_Rsb.append(Rsb)
      group_Tsb.append(Tsb)
    
    return (group_ids, group_Rsb, group_Tsb, group_covs)


  def assign_val(self, ind, fieldname, value):
    self.estimator_results["data"][self.start_ind+ind][fieldname] = value


  def add_param(self, fieldname, value):
    self.estimator_results[fieldname] = value


  def write_json(self, filename):
    json_string = to_json(self.estimator_results)
    with open(filename, 'w') as fid:
      fid.write(json_string)


  def set_startend_to_samplecov(self):
    for i,data in enumerate(self.estimator_results["data"]):
      if "has_sample_cov" in data:
        if data["has_sample_cov"]:
          self.start_ind = i
          break
    n_pts = len(self.estimator_results["data"])
    for i in range(1, n_pts):
      idx = n_pts - i
      if "has_sample_cov" in self.estimator_results["data"][idx]:
        if self.estimator_results["data"][idx]["has_sample_cov"]:
          self.end_ind = idx
          break
  

  def get_network_input(self, input_mode, input_scales, cov_type="WTV"):

    ind_l,ind_u = state_indices(cov_type)

    cov_input = np.array([
      upper_triangular_list(self.P[ind_l:ind_u, ind_l:ind_u, i], ret_dim=False)
      for i in range(self.nposes)
    ])
    Wsb_input = np.vstack([rot.as_rotvec() for rot in self.Rsb])

    if input_mode=="cov":
      raw_net_input = cov_input
    elif input_mode=="gsbcov":
      raw_net_input = np.hstack((Wsb_input, self.Tsb.T, cov_input))
    elif input_mode=="gsbvcov":
      raw_net_input = np.hstack((Wsb_input, self.Tsb.T, self.Vsb.T, cov_input))
    
    # Scale the input
    assert(raw_net_input.shape[1] == len(input_scales))
    scale_matrix = np.tile(np.array(input_scales), (self.nposes,1))

    net_input = raw_net_input / scale_matrix

    return net_input
