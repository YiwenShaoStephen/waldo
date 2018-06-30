#!/usr/bin/env python3

# Copyright      2018  Yiwen Shao

# Apache 2.0

import torch
import torch.utils.data
import argparse
import os
import random
import numpy as np
from models.FCN import fcn16s
from waldo.core_config import CoreConfig
from dataset import COCODataset
from train import sample

parser = argparse.ArgumentParser(
    description='Pytorch COCO sampling script')
parser.add_argument('--test-img', type=str, required=True,
                    help='Directory of test images')
parser.add_argument('--test-ann', type=str, required=True,
                    help='Path to test annotation or info file')
parser.add_argument('--dir', type=str, required=True,
                    help='Experiment directory which contains config, model'
                    'and the output result of this script')
parser.add_argument('--model', type=str, default='model_best.pth.tar',
                    help='Name of the model file to use for segmenting.')
parser.add_argument('--is-val', type=bool, default=True,
                    help='If true, test-ann should contain the ground truth of testset,'
                    'otherwise only test info (image id and size) is required')
parser.add_argument('--limits', default=None, type=int,
                    help="If given, is the size of subset we do segmenting on")
parser.add_argument('--train-image-size', default=384, type=int,
                    help='The size of the parts of training images that we'
                    'train on (in order to form a fixed minibatch size).')
parser.add_argument('--batch-size', default=16, type=int,
                    help='mini-batch size (default: 16)')
random.seed(0)
np.random.seed(0)


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
    num_offsets = len(core_config.offsets)

    model = fcn16s(num_classes + num_offsets)

    model_path = os.path.join(args.dir, args.model)
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print("loaded.")
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    class_nms_file = os.path.join(args.dir, 'configs/subclass.txt')
    if os.path.exists(class_nms_file):
        with open(class_nms_file, 'r') as fh:
            class_nms = fh.readline().split()
            print('Segmenting on {} classes: {}'.format(
                len(class_nms), class_nms))
    else:
        class_nms = None
        print('Segmenting on all classes.')

    dataset = COCODataset(args.test_img, args.test_ann, core_config,
                          mode='train', size=args.train_image_size,
                          class_nms=class_nms, limits=args.limits)
    print('Total samples in the test set: {0}'.format(len(dataset)))

    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=1, batch_size=args.batch_size)

    sample_dir = os.path.join(args.dir, 'sample')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    sample(model, dataloader, sample_dir, core_config)


if __name__ == '__main__':
    main()
