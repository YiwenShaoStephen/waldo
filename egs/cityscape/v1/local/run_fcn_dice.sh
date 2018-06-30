#!/bin/bash
stage=0

num_colors=3
num_classes=9
train_image_size=1024

epochs=10
batch_size=2
lr=0.02
momentum=0.9
dir=exp/fcn_vgg16_dice

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
  $cmd --gpu 1 --mem 30G $dir/train.log limit_num_gpus.sh local/train_new.py \
       --batch-size $batch_size \
       --momentum $momentum \
       --train-image-size $train_image_size \
       --epochs $epochs \
       --lr $lr \
       --loss dice \
       --core-config $dir/configs/core.config \
       --visualize \
       --tensorboard \
       $dir
fi
