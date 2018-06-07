# Copyright      2018  Yiwen Shao

# Apache 2.0

import os
import torch
import numpy as np
from PIL import Image
from skimage.transform import resize
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from waldo.data_manipulation import convert_to_combined_image
import warnings
warnings.filterwarnings("ignore")


class COCODataset:
    def __init__(self, img_dir, annfile, c_cfg, size,
                 is_val=False, class_nms=None, limits=None,
                 job=0, num_jobs=1):
        self.img_dir = img_dir
        self.coco = COCO(annfile)
        self.c_cfg = c_cfg
        self.size = size
        self.class_nms = class_nms  # if given, is a subset of all 81 categories
        # add a background class
        self.catIds = [0]
        self.is_val = is_val
        self.ids = []
        # if given, only classes included in class_nms will be considered as valid class
        # in training. And only images include at least one of these classes will be taken.
        if self.class_nms:
            cats = self.coco.loadCats(self.coco.getCatIds())
            all_class_nms = [cat['name'] for cat in cats]
            for class_nm in self.class_nms:
                if class_nm not in all_class_nms:
                    raise ValueError('the given class name {}'
                                     'should be included in the dataset'.format(class_nm))
            assert len(class_nms) + 1 == self.c_cfg.num_classes
            catIds = self.coco.getCatIds(catNms=self.class_nms)
            self.catIds.extend(catIds)
            for cat_id in catIds:
                self.ids.extend(self.coco.getImgIds(catIds=cat_id))
            self.ids = np.unique(self.ids).tolist()  # sort without duplicates
        else:
            self.ids = list(self.coco.imgs.keys())
            self.catIds.extend(self.coco.getCatIds())

        # if given, only a portion of dataset is used
        if limits:
            self.limits = limits
            self.ids = self.ids[:limits]

        # for parallelization with multiple jobs (threads)
        self.job = job
        self.num_jobs = num_jobs
        assert job <= num_jobs
        if self.job > 0:  # job id is 1-indexed
            id_array = np.array(self.ids)
            self.ids = np.array_split(id_array, self.num_jobs)[
                self.job - 1].tolist()

    def __getitem__(self, index):
        img_id = self.ids[index]
        # only annotations that has catIds included in self.catIds will be taken
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.catIds)
        anns = self.coco.loadAnns(ann_ids)
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = np.array(Image.open(os.path.join(
            self.img_dir, img_path)).convert('RGB'))
        height, width, _ = img.shape
        img = resize(img, (self.size, self.size),
                     preserve_range=True, mode='reflect')

        if self.is_val:  # only original image is needed for validation
            img = np.moveaxis(img, -1, 0)
            img = img.astype('float32') / 256.0
            return img_id, img, (height, width)

        mask = np.zeros((self.size, self.size), dtype='uint16')
        object_class = [0]  # the background class id is 0
        object_id = 1  # start with 1
        for ann in anns:
            category_id = ann['category_id']
            rle = self.annToRLE(ann, height, width)
            class_id = self.catIds.index(category_id)
            # get binary mask for each object and multiple it by object id
            m = maskUtils.decode(rle) * (object_id)
            m = resize(m, (self.size, self.size),
                       order=0, preserve_range=True, mode='reflect').astype(int)
            object_id += 1
            object_class.append(class_id)
            # merge it to a single mask. If overlap occurs, use the newest one
            mask = np.maximum(m, mask)
        n_classes = self.c_cfg.num_classes
        n_offsets = len(self.c_cfg.offsets)
        n_colors = self.c_cfg.num_colors
        image_with_mask = {}
        image_with_mask['img'] = img
        image_with_mask['mask'] = mask
        image_with_mask['object_class'] = object_class
        combined_img = convert_to_combined_image(image_with_mask, self.c_cfg)
        # combined_img = randomly_crop_combined_image(
        #     combined_img, self.c_cfg, self.size, self.size)
        img = torch.from_numpy(combined_img[:n_colors, :, :])
        class_label = torch.from_numpy(
            combined_img[n_colors:n_colors + n_classes, :, :])
        bound = torch.from_numpy(
            combined_img[n_colors + n_classes:n_colors + n_classes + n_offsets, :, :])

        return img, class_label, bound

    def __len__(self):
        return len(self.ids)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: RLE
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle


class COCOTestset:
    def __init__(self, img_dir, info_file, c_cfg, class_nms=None):
        self.img_dir = img_dir
        self.coco = COCO(info_file)
        self.c_cfg = c_cfg
        self.class_nms = class_nms  # if given, is a subset of all 81 categories
        # add a background class
        self.catIds = [0]
        if self.class_nms:
            cats = self.coco.loadCats(self.coco.getCatIds())
            all_class_nms = [cat['name'] for cat in cats]
            for class_nm in self.class_nms:
                if class_nm not in all_class_nms:
                    raise ValueError('the given class name {}'
                                     'should be included in the dataset'.format(class_nm))
            assert len(class_nms) + 1 == self.c_cfg.num_classes
            catIds = self.coco.getCatIds(catNms=self.class_nms)
            self.catIds.extend(catIds)
            self.ids = self.coco.getImgIds(catIds=catIds)
        else:
            self.ids = list(self.coco.imgs.keys())
            self.catIds.extend(self.coco.getCatIds())

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = np.array(Image.open(os.path.join(
            self.img_dir, img_path)).convert('RGB'))
        return img, img_id

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    from waldo.core_config import CoreConfig
    from pycocotools.cocoeval import COCOeval
    from waldo.segmenter import ObjectSegmenter, SegmenterOptions
    import torchvision
    c_config = CoreConfig()
    c_config.num_classes = 4
    c_config.num_colors = 3
    class_nms = ['person', 'dog', 'skateboard']
    trainset = COCODataset('data/download/val2017',
                           'data/download/annotations/instances_val2017.json', c_config,
                           128, class_nms)
    trainloader = torch.utils.data.DataLoader(
        trainset, num_workers=1, batch_size=1)
    data_iter = iter(trainloader)
    img, category, bound, img_id = data_iter.next()
    torchvision.utils.save_image(img, 'raw.png')
    for i in range(c_config.num_classes):
        torchvision.utils.save_image(
            category[:, i:i + 1, :, :], 'class_{}.png'.format(i))
    for i in range(len(c_config.offsets)):
        torchvision.utils.save_image(
            bound[:, i:i + 1, :, :], 'bound_{}.png'.format(i))

    offset_list = c_config.offsets
    num_classes = c_config.num_classes
    object_merge_factor = 1.0
    segmenter_opts = SegmenterOptions(same_different_bias=0,
                                      object_merge_factor=object_merge_factor)
    seg = ObjectSegmenter(category[0].detach().numpy(),
                          bound[0].detach().numpy(),
                          num_classes, offset_list,
                          segmenter_opts)
    mask_pred, object_class = seg.run_segmentation()
    print(object_class)
    image_with_mask = {}
    image_with_mask['mask'] = mask_pred
    image_with_mask['object_class'] = object_class
    results = []
    img_id = img_id[0].numpy().tolist()
    results.extend(convert_to_coco_result(
        image_with_mask, img_id, trainset.catIds))
    coco = trainset.coco
    coco_results = coco.loadRes(results)
    cocoEval = COCOeval(coco, coco_results, 'segm')
    cocoEval.params.imgIds = [img_id]
    cocoEval.params.catIds = coco.getCatIds(catNms=class_nms)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
