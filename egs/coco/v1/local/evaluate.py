#!/usr/bin/env python3

# Copyright      2018  Yiwen Shao

# Apache 2.0

import os
import pickle
import argparse
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

parser = argparse.ArgumentParser(description='scoring script for COCO dataset')
parser.add_argument('--segment-dir', type=str, required=True,
                    help='Directory of segmentation results')
parser.add_argument('--val-ann', type=str,
                    default='data/download/annotations/instances_val2017.json',
                    help='Path to validation annotations file')
parser.add_argument('--imgid', type=int, default=None,
                    help='If given, only do evaluation on that image')


def main():
    global args
    args = parser.parse_args()

    cocoGt = COCO(args.val_ann)
    class_nms = ['person', 'dog', 'skateboard']
    evaluate(cocoGt, args.segment_dir, class_nms, args.imgid)


def evaluate(coco, segment_dir, class_nms, imgid=None):
    pkl_dir = os.path.join(segment_dir, 'pkl')
    results = []
    if imgid:
        imgIds = [imgid]
    else:
        imgIds = []
    pkl_files = next(os.walk(pkl_dir))[2]
    for pkl_file in pkl_files:
        with open('{}/{}'.format(pkl_dir, pkl_file), 'rb') as fh:
            result = pickle.load(fh)
            results.extend(result)
            if not imgid:
                # imgId should be int instead of str
                imgId = int(pkl_file.split('.')[0])
                imgIds.append(imgId)
    print('Evaluating on {} classes: {}'.format(len(class_nms), class_nms))
    print('Evaluating on {} images: {}'.format(len(imgIds), imgIds))
    coco_results = coco.loadRes(results)
    cocoEval = COCOeval(coco, coco_results, 'segm')
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = coco.getCatIds(catNms=class_nms)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    main()
