#!/bin/bash

. ./path.sh

stage=0

. parse_options.sh  # e.g. this parses the --stage option if supplied.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

dir=exp/all/fcn8s_resnet101_bce


segdir=segment_val/256
logdir=$dir/$segdir/log
nj=2
if [ $stage -le 2 ]; then
  echo "doing segmentation...."
    $cmd --mem 10G JOB=1:$nj $logdir/segment.JOB.log local/segment.py \
       --limits 2 \
       --train-image-size 256 \
       --seg-size 256 \
       --model model_best.pth.tar \
       --arch fcn8_resnet101 \
       --mode val \
       --segment $segdir \
       --job JOB --num-jobs $nj \
       --dir $dir \
       --img data/train \
       --ann data/annotations/instancesonly_filtered_gtFine_train.json
fi

if [ $stage -le 3 ]; then
  echo "doing evaluation..."
  local/evaluate.py \
    --segment-dir $dir/$segdir \
    --val-ann data/annotations/instancesonly_filtered_gtFine_train.json
fi
