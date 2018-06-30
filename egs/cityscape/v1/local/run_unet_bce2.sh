#!/bin/bash
stage=0

num_colors=3
num_classes=9
train_image_size=256

epochs=25
batch_size=16
lr=0.01
momentum=0.9
dir=exp/unet_bce2

. ./cmd.sh
. ./path.sh
. ./scripts/parse_options.sh




if [ $stage -le 1 ]; then
  mkdir -p $dir/configs
  echo "$0: Creating core configuration and unet configuration"

  cat <<EOF > $dir/configs/core.config
  num_classes $num_classes
  num_colors $num_colors
EOF

fi


if [ $stage -le 2 ]; then
  echo "$0: Training the network....."
  $cmd --gpu 1 --mem 20G $dir/train.log limit_num_gpus.sh local/train_new.py \
       --batch-size $batch_size \
       --momentum $momentum \
       --train-image-size $train_image_size \
       --arch unet \
       --epochs $epochs \
       --lr $lr \
       --milestones 15 20 \
       --core-config $dir/configs/core.config \
       --visualize \
       --tensorboard \
       $dir
fi
