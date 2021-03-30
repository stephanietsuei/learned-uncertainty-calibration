import sys, os
import argparse

import numpy as np
import rosbag

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
import to_json


parser = argparse.ArgumentParser()
parser.add_argument('-bagfile', required=True)
parser.add_argument('-motiontopic', default='/xivo/fullstate')
parser.add_argument('-maptopic', default='/xivo/map')
parser.add_argument('-dataset', default='tumvi')
parser.add_argument('-seq', default='room1')
parser.add_argument('-sen', default='tango_top')
parser.add_argument('-cam_id', default=0, type=int)



def rosvec_to_list(rosvec):
  return [ rosvec.x, rosvec.y, rosvec.z ]

def rosquat_to_list(rosquat):
  return [ rosquat.w, rosquat.x, rosquat.y, rosquat.z ]


def roslist_to_list(roslist, length):
  out = []
  for i in range(length):
    out.append(roslist[i])
  return out


def main(bagfilename, motiontopic, maptopic, output_file):
  bagfile = rosbag.Bag(bagfilename, mode='r')

  # data structure that maps timestamp to a dictionary
  alltimesteps = {}

  # parse fullstate topic
  for _,msg,_ in bagfile.read_messages(topics=motiontopic):
    entry = {}

    ts = msg.header.stamp.secs*1000000000 + msg.header.stamp.nsecs

    entry['group'] = msg.group

    entry['ImagePath'] = ''
    entry['Timestamp'] = ts
    entry['Tsb_XYZ'] = rosvec_to_list(msg.gsb.translation)
    entry['qsb_WXYZ'] = rosquat_to_list(msg.gsb.rotation)
    entry['Tbc_XYZ'] = rosvec_to_list(msg.gbc.translation)
    entry['qbc_WXYZ'] = rosquat_to_list(msg.gbc.rotation)
    entry['Tsc_XYZ'] = rosvec_to_list(msg.gsc.translation)
    entry['qsc_WXYZ'] = rosquat_to_list(msg.gsc.rotation)
    entry['Vsb_XYZ'] = rosvec_to_list(msg.Vsb)

    entry['Pstate'] = [ msg.MotionStateSize, list(msg.covariance) ]

    entry['MeasurementUpdateInitialized'] = bool(msg.MeasurementUpdateInitialized)
    entry['inn_Tsb'] = rosvec_to_list(msg.inn_Tsb)
    entry['inn_Wsb'] = rosvec_to_list(msg.inn_Wsb)
    entry['inn_Vsb'] = rosvec_to_list(msg.inn_Vsb)

    entry['bg'] = rosvec_to_list(msg.bg)
    entry['ba'] = rosvec_to_list(msg.ba)
    entry['qg_WXYZ'] = rosquat_to_list(msg.qg)
    entry['td'] = msg.td
    entry['Ca'] = list(msg.Ca)
    entry['Cg'] = list(msg.Cg)

    alltimesteps[ts] = entry

  # Parse map topic
  for _,msg,_ in bagfile.read_messages(topics=maptopic):

    ts = msg.header.stamp.secs*1000000000 + msg.header.stamp.nsecs

    entry = alltimesteps[ts]

    feature_positions = []
    feature_ids = []
    feature_covariances = []

    for feature in msg.features:
      feature_ids.append(feature.id)
      feature_positions.extend(rosvec_to_list(feature.Xs))

      cov = roslist_to_list(feature.covariance, 9)
      feature_covariances.extend([cov[0], cov[1], cov[2], cov[4], cov[5], cov[8]])

    entry['num_instate_features'] = msg.num_features
    entry['feature_positions'] = feature_positions
    entry['feature_covs'] = feature_covariances
    entry['feature_ids'] = feature_ids

    alltimesteps[ts] = entry

  # sort timestamps
  timestamps = alltimesteps.keys()
  timestamps.sort()

  # create list of dictionaries
  final_list = []
  for ts in timestamps:
    final_list.append(alltimesteps[ts])

  json_string = to_json.to_json(final_list)
  with open(output_file, 'w') as fid:
    fid.write(json_string)



if __name__=="__main__":
  args = parser.parse_args()

  if args.dataset=="tumvi":
    output_filename = "tumvi_{}_cam{}".format(args.seq, args.cam_id)
  elif args.dataset=="cosyvio":
    output_filename = "cosyvio_{}_{}".format(args.sen, args.seq)
  else:
    output_filename = "{}_{}".format(args.dataset, args.seq)

  main(args.bagfile, args.motiontopic, args.maptopic, output_filename)
