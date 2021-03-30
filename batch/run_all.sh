#!/bin/bash



for cam_id in {0..1}
do
  for room_id in {1..6}
  do
    python3 run/xivo/pyxivo.py -mode dumpCov -dataset tumvi -cfg cfg/tumvi_cam${cam_id}.json -cam_id ${cam_id} -seq room${room_id} 

  done
done