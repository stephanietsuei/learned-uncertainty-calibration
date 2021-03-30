#!/bin/bash

# After finding a sample covariance window, run this to compute all adjustments
# to the covariances in the dataset


# NeurIPS 2020 numbers, test split = room6, cam1
# with test split means ranking = training + test, otherwise overall
#window_sizes=(233 191 583 93 537)     # nbins=2000, no test split
#window_sizes=(283 213 597 105 489)    # (final!) nbins=2000, with test split
#window_sizes=(235 187 595 95 509)     # nbins=auto, no test split
#window_sizes=(247 193 591 89 527)     # nbins=auto, with test split

# ICRA 2021 numbers, test split = room6, cam0 and cam1, nbins=2000
window_sizes=(275 221 581 93 529)      # rank=training+test
#window_sizes=(275 285 581 125 529)    # rank=test only
#window_sizes=(222 191 569 93 519)     # rank=training only


cov_types=("WTV" "W" "T" "V" "WT")

for idx in {0..4}
do

  window=${window_sizes[${idx}]}
  cov_type=${cov_types[${idx}]}
  echo $window
  echo $cov_type

  cp -r "$HOME/data/covdump" "$HOME/data/xivo_tumvi_cov_dump_${window}"

  echo "Computing sample covariance"
  for cam_id in {0..1}
  do
    for room_id in {1..6}
    do
      #python3 run/xivo/pyxivo.py -mode dumpCov -dataset tumvi -cfg cfg/tumvi_cam${cam_id}.json -cam_id ${cam_id} -seq room${room_id}
      echo "room$room_id cam$cam_id"

      python analysis/eval_cov_calibration.py -mode compute_sample_cov -dataset tumvi -cam_id ${cam_id} -seq room${room_id} -sample_cov_window_size ${window} -dump $HOME/data/xivo_tumvi_cov_dump_${window}

    done
  done


  # scalar adjustment
  echo "unweighted scalar adjustment"
  python analysis/single_scalar_adjustment.py -dataset tumvi -dump $HOME/data/xivo_tumvi_cov_dump_$window -cov_type $cov_type

  # Get data/validation sets
  echo "getting validation data"
  python analysis/dense_layers_adjustment.py -dump $HOME/data/xivo_tumvi_cov_dump_$window -process_text -cov_type $cov_type -input_mode cov
  python analysis/dense_layers_adjustment.py -dump $HOME/data/xivo_tumvi_cov_dump_$window -process_text -cov_type $cov_type -input_mode gsbcov
  python analysis/dense_layers_adjustment.py -dump $HOME/data/xivo_tumvi_cov_dump_$window -process_text -cov_type $cov_type -input_mode gsbvcov

  python analysis/dense_layers_adjustment.py -dump $HOME/data/xivo_tumvi_cov_dump_$window -process_text -use_validation -cov_type $cov_type -input_mode cov
  python analysis/dense_layers_adjustment.py -dump $HOME/data/xivo_tumvi_cov_dump_$window -process_text -use_validation -cov_type $cov_type -input_mode gsbcov
  python analysis/dense_layers_adjustment.py -dump $HOME/data/xivo_tumvi_cov_dump_$window -process_text -use_validation -cov_type $cov_type -input_mode gsbvcov


  # Weighted scalar adjustment
  echo "weighted scalar adjustment"
  cp -r $HOME/data/xivo_tumvi_cov_dump_${window}  $HOME/data/covdump_${window}_weighted

  python analysis/single_scalar_adjustment.py -dataset tumvi -dump $HOME/data/covdump_${window}_weighted -cov_type $cov_type
done
