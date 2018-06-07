#!/bin/bash

set -e # exit on error
. ./path.sh

stage=0

. parse_options.sh  # e.g. this parses the --stage option if supplied.


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

local/check_dependencies.sh

# download data
if [ $stage -le 0 ]; then
  local/prepare_data.sh
fi


epochs=120
depth=5
dir=exp/unet_${depth}_${epochs}_sgd
if [ $stage -le 1 ]; then
  # training
  local/run_unet.sh --dir $dir --epochs $epochs --depth $depth
fi

logdir=$dir/segment/log
nj=10
if [ $stage -le 2 ]; then
    echo "doing segmentation...."
  $cmd JOB=1:$nj $logdir/segment.JOB.log local/segment.py \
       --test-img data/download/val2017 \
       --test-ann data/download/annotations/instances_val2017.json \
       --dir $dir/segment \
       --train-image-size 128 \
       --model model_best.pth.tar \
       --job JOB --num-jobs $nj
fi

if [ $stage -le 3 ]; then
  echo "doing evaluation..."
  local/evaluate.py \
    --segment-dir $dir/segment \
    --val-ann data/download/annotations/instances_val2017.json
fi
