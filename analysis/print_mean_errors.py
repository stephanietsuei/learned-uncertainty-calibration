import os, sys
import numpy as np
import scipy


sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from utils import get_stock_parser, get_xivo_gt_filename, \
  get_xivo_output_filename
from eval_cov_calibration import CovarianceCalibration

parser = get_stock_parser("Prints mean state and state errors across the" +
  " whole TUM VI dataset.")
args = parser.parse_args()


# Data to collect
translation_errors = np.zeros((0,3))
rotation_errors = np.zeros((0,3))
velocity_errors = np.zeros((0,3))
translations = np.zeros((0,3))
rotations = np.zeros((0,3))
velocities = np.zeros((0,3))

for cam_id in [0,1]:
  for seq in ["room1", "room2", "room3", "room4", "room5", "room6"]:
    print("Cam {}, {}".format(cam_id, seq))

    estimator_data = get_xivo_output_filename(args.dump, "tumvi", seq,
                                              cam_id=cam_id)
    gt_data = get_xivo_gt_filename(args.dump, "tumvi", seq)

    calib = CovarianceCalibration(seq, gt_data, estimator_data,
      three_sigma=False, start_ind=0, end_ind=None,
      point_cloud_registration="horn", adjust_startend_to_samplecov=False)
    calib.align_gt_to_est()
    calib.compute_errors()

    translation_errors = np.concatenate((translation_errors, calib.Tsb_error.T))
    rotation_errors = np.concatenate((rotation_errors, calib.Wsb_error.T))
    velocity_errors = np.concatenate((velocity_errors, calib.Vsb_error.T))
    translations = np.concatenate((translations, calib.Tsb_gt.T))
    rotations = np.concatenate((rotations, calib.Wsb_gt.T))
    velocities = np.concatenate((velocities, calib.Vsb_gt.T))


print("total points: {}".format(translation_errors.shape[0]))

# Compute and print means
rot = scipy.spatial.transform.Rotation.from_rotvec(rotation_errors)
mean_Tsb_norm = np.mean(np.linalg.norm(translations, axis=1))
mean_Vsb_norm = np.mean(np.linalg.norm(velocities, axis=1))
mean_Wsb_norm = np.mean(np.linalg.norm(rotations, axis=1))
mean_Tsb_error = np.linalg.norm(np.mean(translation_errors, axis=0))
mean_Wsb_error = np.linalg.norm(rot.mean().as_rotvec())
mean_Vsb_error = np.linalg.norm(np.mean(velocity_errors, axis=0))
print("Mean Translation Errors: {} m".format(mean_Tsb_error))
print("Mean Rotation Error: {} rad".format(np.linalg.norm(mean_Wsb_error)))
print("Mean Velocity Errors: {} m/s".format(mean_Vsb_error))
print("Mean Translation Norm: {} m".format(mean_Tsb_norm))
print("Mean Rotation Norm: {} rad".format(mean_Wsb_norm))
print("Mean Velocity Norm: {} m/s".format(mean_Vsb_norm))
