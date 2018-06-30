#!/usr/bin/env python3

# Copyright 2018 Yiwen Shao

# Apache 2.0

import torch
import os
import argparse
from waldo.core_config import CoreConfig
from waldo.train_utils import runningScore, soft_dice_loss
from models.Unet import UNet
from dataset import COCODataset

parser = argparse.ArgumentParser(
    description='Pytorch COCO instance segmentation setup')
parser.add_argument('dir', type=str,
                    help='directory of output models and logs')
parser.add_argument('--val-img', default='data/val', type=str,
                    help='Directory of validation images')
parser.add_argument('--val-ann',
                    default='data/annotations/instancesonly_filtered_gtFine_val.json',
                    help='Path to validation set annotations')
parser.add_argument('--class-name-file', default=None, type=str,
                    help="If given, is the subclass we are going to detect/segment")
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 16)')
parser.add_argument('--train-image-size', default=None, type=int,
                    help='The size of the parts of training images that we'
                    'train on (in order to form a fixed minibatch size).'
                    'These are derived from the input images'
                    ' by padding and then random cropping.')
parser.add_argument('--model', type=str, default='model_best.pth.tar',
                    help='Name of the model file to use for segmenting.')
parser.add_argument('--loss', default='bce', type=str, choices=['bce', 'dice'],
                    help='loss function')


def main():
    global args
    args = parser.parse_args()
    core_config_path = os.path.join(args.dir, 'configs/core.config')

    core_config = CoreConfig()
    core_config.read(core_config_path)
    print('Using core configuration from {}'.format(core_config_path))

    offset_list = core_config.offsets
    print("offsets are: {}".format(offset_list))

    # model configurations from core config
    num_classes = core_config.num_classes
    num_colors = core_config.num_colors
    num_offsets = len(core_config.offsets)

    if args.class_name_file:
        with open(args.class_name_file, 'r') as fh:
            class_nms = fh.readline().split()
            print('Training on {} classes: {}'.format(
                len(class_nms), class_nms))
    else:
        class_nms = None
        print('Training on all classes.')

    model = UNet(num_classes, num_offsets)
    model_path = os.path.join(args.dir, args.model)
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("loaded.")

        valset = COCODataset(args.val_img, args.val_ann, core_config,
                             args.train_image_size, class_nms=class_nms)
        valloader = torch.utils.data.DataLoader(
            valset, num_workers=4, batch_size=args.batch_size)

    score_metrics = runningScore(num_classes, num_offsets)
    validate(valloader, model, args.batch_size, score_metrics)


def validate(validateloader, model, batch_size, score_metrics):
    """Perform validation on the validation set"""

    # switch to evaluate mode
    model.eval()

    score_metrics.reset()

    for i, (input, class_label, bound) in enumerate(validateloader):

        with torch.no_grad():
            output = model(input)

            # TODO. Treat class label and bound label equally by now
            target = torch.cat((class_label, bound), 1)

            score_metrics.update(output, target)

    score, class_iou = score_metrics.get_scores()
    for k, v in score.items():
        print('{}:\t{}'.format(k, v))
    for k, v in class_iou.items():
        print('Class {}:\t{}'.format(k, v))


if __name__ == "__main__":
    main()
