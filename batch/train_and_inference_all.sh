#!/bin/bash

# training script for each gpu


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

  echo "Yahoo! Training time!"
  window=${window_sizes[${idx}]}
  cov_type=${cov_types[${idx}]}
  echo $window
  echo $cov_type

  mkdir -p /local2/stsuei/xivo_tumvi_dnndump_${window}_${cov_type}

  python batch/train_several.py -dump /local2/stsuei/xivo_tumvi_cov_dump_${window} -dnndump /local2/stsuei/xivo_tumvi_dnndump_${window}_${cov_type} -cov_type ${cov_type} -gpu_id $1

done


for idx in {0..4}
do
  echo "Yahoo! Inference time!"
  window=${window_sizes[${idx}]}
  cov_type=${cov_types[${idx}]}
  echo $window
  echo $cov_type

  python batch/process_dense_networks.py -mode compute -dump /local2/stsuei/xivo_tumvi_cov_dump_${window} -dnndump /local2/stsuei/xivo_tumvi_dnndump_${window}_${cov_type} -cov_type ${cov_type} -nbins 2000 -gpu_id $1   &> chi2_${cov_type}_gpu$1.txt
done