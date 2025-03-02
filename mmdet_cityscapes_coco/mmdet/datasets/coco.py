# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


import numpy as np
import pycocotools.mask as mask_utils
import cv2

import random
import torch

@DATASETS.register_module()
class CocoDataset(BaseDetDataset):
    """Dataset for COCO."""

    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def __init__(self, percentage=100, ratio=None, top_k=-1, *args, **kwargs):
        self.percentage = percentage
        self.ratio = ratio
        self.top_k = top_k
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            score_fields = ['forgetting_score', 'roi_boundary_score', 'roi_boundary_el2n_score',
                    'roi_el2n_score', 'roi_aum_score', 'p2a_ratio', 'roi_score']
            for field in score_fields:
                if field in ann and ann[field] != 0:
                    instance[field] = ann[field]

            instances.append(instance)
        data_info['instances'] = instances
        return data_info


    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list
        
        if self.percentage < 100:
            sample_size = int(len(self.data_list) * (self.percentage / 100))
            self.data_list = random.sample(self.data_list, sample_size)

            print('*******************************')
            print('Random Selected Number: ', len(self.data_list))
            print('*******************************')

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
    

##################
    def calculate_ratios(self):
        """Calculate the perimeter-to-area ratio for each mask in the dataset."""
        for data_info in self.data_list:
            instances = [inst for inst in data_info['instances'] if not inst['ignore_flag']]

            for instance in instances:
                if isinstance(instance['mask'], list):
                    mask = np.zeros((data_info['height'], data_info['width']), dtype=np.uint8)
                    polygons = instance['mask']
                    for polygon in polygons:
                        poly = np.array(polygon).reshape((len(polygon) // 2, 2))
                        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                else:
                    mask = mask_utils.decode(instance['mask'])
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour_max = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(contour_max)
                    if area > 0:
                        perimeter = cv2.arcLength(contour_max, True)
                        instance['ratio'] = perimeter / area
                    else:
                        instance['ratio'] = 0
                else:
                    instance['ratio'] = 0

    def calculate_norm_ratios(self):
        """Calculate the perimeter-to-area ratio for each mask in the dataset."""
        import math
        for data_info in self.data_list:
            instances = [inst for inst in data_info['instances'] if not inst['ignore_flag']]

            for instance in instances:
                if isinstance(instance['mask'], list):
                    mask = np.zeros((data_info['height'], data_info['width']), dtype=np.uint8)
                    polygons = instance['mask']
                    for polygon in polygons:
                        poly = np.array(polygon).reshape((len(polygon) // 2, 2))
                        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                else:
                    mask = mask_utils.decode(instance['mask'])
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour_max = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(contour_max)
                    if area > 0:
                        perimeter = cv2.arcLength(contour_max, True)

                        if area != 0:
                            circle_perimeter = 2 * math.sqrt(math.pi * area)
                            instance['ratio'] = perimeter / circle_perimeter
                        else:
                            instance['ratio'] = 0  # Avoid division by zero for shapes with no area
                    else:
                        instance['ratio'] = 0
                else:
                    instance['ratio'] = 0

    def calculate_ratios_norm(self):
        """Calculate the perimeter-to-area ratio for each mask in the dataset and normalize within each category."""
        import math
        category_scores = {}

        # Step 1: Calculate ratio for each instance and accumulate category-wise scores
        for data_info in self.data_list:
            instances = [inst for inst in data_info['instances'] if not inst['ignore_flag']]

            for instance in instances:
                if isinstance(instance['mask'], list):
                    mask = np.zeros((data_info['height'], data_info['width']), dtype=np.uint8)
                    polygons = instance['mask']
                    for polygon in polygons:
                        poly = np.array(polygon).reshape((len(polygon) // 2, 2))
                        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                else:
                    mask = mask_utils.decode(instance['mask'])

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour_max = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(contour_max)
                    if area > 0:
                        perimeter = cv2.arcLength(contour_max, True)
                        instance['org_ratio'] = perimeter / area
                    else:
                        instance['org_ratio'] = 0
                else:
                    instance['org_ratio'] = 0

                # Accumulate category-wise scores
                cat_id = instance['bbox_label']
                score = instance['org_ratio']
                if cat_id in category_scores:
                    category_scores[cat_id] += score
                else:
                    category_scores[cat_id] = score

        # Step 2: Normalize ratios within each category based on total category scores
        for data_info in self.data_list:
            instances = [inst for inst in data_info['instances'] if not inst['ignore_flag']]

            for instance in instances:
                cat_id = instance['bbox_label']
                score = instance['org_ratio']
                if category_scores[cat_id] > 0:
                    instance['ratio'] = score / category_scores[cat_id]
                else:
                    instance['ratio'] = 0

    def calculate_norm_ratios_norm(self):
        """Calculate the perimeter-to-area ratio for each mask in the dataset and normalize within each category."""
        import math
        category_scores = {}

        # Step 1: Calculate ratio for each instance and accumulate category-wise scores
        for data_info in self.data_list:
            instances = [inst for inst in data_info['instances'] if not inst['ignore_flag']]

            for instance in instances:
                if isinstance(instance['mask'], list):
                    mask = np.zeros((data_info['height'], data_info['width']), dtype=np.uint8)
                    polygons = instance['mask']
                    for polygon in polygons:
                        poly = np.array(polygon).reshape((len(polygon) // 2, 2))
                        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                else:
                    mask = mask_utils.decode(instance['mask'])

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour_max = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(contour_max)
                    if area > 0:
                        perimeter = cv2.arcLength(contour_max, True)
                        circle_perimeter = 2 * math.sqrt(math.pi * area)
                        instance['org_ratio'] = perimeter / circle_perimeter
                    else:
                        instance['org_ratio'] = 0
                else:
                    instance['org_ratio'] = 0

                # Accumulate category-wise scores
                cat_id = instance['bbox_label']
                score = instance['org_ratio']
                if cat_id in category_scores:
                    category_scores[cat_id] += score
                else:
                    category_scores[cat_id] = score

        # Step 2: Normalize ratios within each category based on total category scores
        # print_list = [23541, 177090, 309797, 22194, 184994, 392154]
        for data_info in self.data_list:
            instances = [inst for inst in data_info['instances'] if not inst['ignore_flag']]
            # if data_info['img_id'] in print_list:
                # print('Img ID: ', data_info['img_id'])

            for instance in instances:
                cat_id = instance['bbox_label']
                score = instance['org_ratio']
                if category_scores[cat_id] > 0:
                    instance['ratio'] = score / category_scores[cat_id]
                else:
                    instance['ratio'] = 0

                    # print(str(instance['bbox_label'])+' '+str(instance['ratio']))

    def select_images_by_store_score_stratified(self, top_k, pruning_hard_rate=0.1):
        stratas = 50
        print('Using stratified sampling for image selection...')

        image_scores = {}
        # Collect scores for each image
        for data_info in self.data_list:
            img_id = data_info['img_id']
            instances = data_info['instances']
            
            # if self.ratio == 'store_ratio':
            #     txt_score = 'p2a_ratio'
            # elif self.ratio == 'store_aum_score':
            #     txt_score = 'roi_aum_score'
            # elif self.ratio == 'store_el2n_score':
            #     txt_score = 'roi_el2n_score'
            # elif self.ratio == 'store_entropy_score':
            #     txt_score = 'roi_score'
            if self.ratio == 'ccs':
                txt_score = 'roi_aum_score'
            
            total_score = sum(instance.get(txt_score, 0) for instance in instances if instance.get('ignore_flag', 0) == 0)
            image_scores[img_id] = total_score

        scores = torch.tensor(list(image_scores.values()), dtype=torch.float32)
        image_ids = np.array(list(image_scores.keys()))

        if pruning_hard_rate > 0:
            sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=False)
            cutoff_index = int(len(sorted_images) * pruning_hard_rate)
            pruned_sorted_images = sorted_images[cutoff_index:]

            pruned_image_ids, pruned_scores = zip(*pruned_sorted_images) if pruned_sorted_images else ([], [])
            scores = torch.tensor(pruned_scores, dtype=torch.float32)
            image_ids = np.array(pruned_image_ids)

        min_score = torch.min(scores)
        max_score = torch.max(scores) * 1.0001
        step = (max_score - min_score) / stratas

        def bin_range(k):
            return min_score + k * step, min_score + (k + 1) * step

        strata_num = [torch.logical_and(scores >= bin_range(i)[0], scores < bin_range(i)[1]).sum() for i in range(stratas)]
        strata_num = torch.tensor(strata_num)

        def bin_allocate(num, bins):
            sorted_index = torch.argsort(bins)
            sort_bins = bins[sorted_index]

            num_bin = bins.shape[0]
            rest_exp_num = num
            budgets = []
            for i in range(num_bin):
                rest_bins = num_bin - i
                avg = rest_exp_num // rest_bins
                cur_num = min(sort_bins[i].item(), avg)
                budgets.append(cur_num)
                rest_exp_num -= cur_num

            rst = torch.zeros((num_bin,)).type(torch.int)
            rst[sorted_index] = torch.tensor(budgets).type(torch.int)
            return rst

        budgets = bin_allocate(top_k, strata_num)

        selected_images = []
        for i in range(stratas):
            mask = torch.logical_and(scores >= bin_range(i)[0], scores < bin_range(i)[1])
            pool = image_ids[mask.numpy()]
            rand_index = np.random.permutation(pool.shape[0])
            selected_images.extend(pool[rand_index][:budgets[i]].tolist())

        print('Selected number:', len(selected_images))
        # Update the dataset's data list to only include selected images
        self.data_list = [data for data in self.data_list if data['img_id'] in selected_images]



    def select_images_by_ratio(self, top_k, skip_ratio):
        """Selects images based on the highest average perimeter-to-area ratios, considering all classes."""
        self.calculate_ratios()
        num_classes = len(self.METAINFO['classes'])
        class_image_totals = {cls: {} for cls in self.METAINFO['classes']}

        # Precompute ratios and organize by class
        for data_info in self.data_list:
            img_id = data_info['img_id']
            instances = data_info['instances']
            for instance in instances:
                if instance['ignore_flag'] == 1:
                    continue

                # mask = mask_utils.decode(instance['mask'])
                ratio = instance['ratio']
                cls_name = self.METAINFO['classes'][instance['bbox_label'] - 1]

                if img_id not in class_image_totals[cls_name]:
                    class_image_totals[cls_name][img_id] = []

                class_image_totals[cls_name][img_id].append(ratio)

        # Calculate mean ratios and select images
        selected_images = set()
        for cls, img_totals in class_image_totals.items():
            # mean_ratios = {img_id: np.mean(ratios) for img_id, ratios in img_totals.items() if ratios}
            mean_ratios = {img_id: np.sum(ratios) for img_id, ratios in img_totals.items() if ratios}
            sorted_imgs = sorted(mean_ratios.items(), key=lambda x: x[1], reverse=True)
            skip_count = int(len(sorted_imgs) * skip_ratio)
            sorted_imgs = sorted_imgs[skip_count:]  # Apply skip ratio
            initial_n_per_class = max(1, top_k // num_classes)
            selected_images.update([img_id for img_id, _ in sorted_imgs[:initial_n_per_class]])

        # If needed, select additional images to meet top_k
        current_total = len(selected_images)
        while current_total < top_k:
            needed = top_k - current_total
            per_class_additional = max(1, needed // num_classes)

            additional_images = []
            for cls, img_totals in class_image_totals.items():
                mean_ratios = {img_id: ratios for img_id, ratios in img_totals.items() if img_id not in selected_images}
                sorted_imgs = sorted(mean_ratios.items(), key=lambda x: x[1], reverse=True)
                additional_images.extend(sorted_imgs[:per_class_additional])

            selected_images.update([img_id for img_id, _ in additional_images])
            new_total = len(selected_images)
            if new_total == current_total:  # Break if no new images are added
                break
            current_total = new_total

        # Update the dataset's data list
        self.data_list = [data for data in self.data_list if data['img_id'] in selected_images]
        print('Org size: ', len(self.data_list))
        print('Selected number: ', len(selected_images))

    def calculate_size_distribution(self, size_thresholds={'small': 32**2, 'middle': 64**2}):
        size_counts = {'small': 0, 'middle': 0, 'large': 0}

        # Loop over all data entries
        for data_info in self.data_list:
            instances = data_info['instances']
            
            # Loop over all instances in the data entry
            for instance in instances:
                if instance['ignore_flag'] == 1:
                    continue

                # Decode the mask and calculate the area
                if isinstance(instance['mask'], list):
                    mask = np.zeros((data_info['height'], data_info['width']), dtype=np.uint8)
                    polygons = instance['mask']
                    for polygon in polygons:
                        poly = np.array(polygon).reshape((len(polygon) // 2, 2))
                        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                else:
                    print(type(mask))
                    mask = mask_utils.decode(instance['mask'])


                area = np.sum(mask)

                # Determine the size category more efficiently
                if area < size_thresholds['small']:
                    size_category = 'small'
                elif area < size_thresholds['middle']:
                    size_category = 'middle'
                else:
                    size_category = 'large'

                size_counts[size_category] += 1

        total_masks = sum(size_counts.values())
        if total_masks > 0:
            size_proportions = {size: count / total_masks for size, count in size_counts.items()}
        else:
            size_proportions = {size: 0 for size in size_counts}

        return size_counts, size_proportions
    
    def select_images_by_store_score(self, top_k):
        """Selects images based on the total ratio scores from all instances within each image."""
        if self.ratio == 'store_ratio':
            print('Selected by total P-2-A Ratio ...')
            txt_score = 'p2a_ratio'
        elif self.ratio == 'store_aum_score':
            print('Selected by total AUM Ratio ...')
            txt_score = 'roi_aum_score'
        elif self.ratio == 'store_el2n_score':
            print('Selected by total EL2N Ratio ...')
            txt_score = 'roi_el2n_score'
        elif self.ratio == 'store_entropy_score':
            print('Selected by total Entropy Ratio ...')
            txt_score = 'roi_score'
        
    
        image_scores = {}

        for data_info in self.data_list:
            img_id = data_info['img_id']
            instances = data_info['instances']
            
            total_score = 0
            num_instances_without_txt_score = 0

            for instance in instances:
                if instance.get('ignore_flag', 0) == 1:
                    continue
                total_score += instance.get(txt_score, 0)
                # if 'txt_score' not in instance:
                #     num_instances_without_txt_score += 1

            if total_score > 0:  # Only consider images with a positive total score
                image_scores[img_id] = total_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        selected_images = [img_id for img_id, _ in sorted_images[:top_k]]
        
        # Update the dataset's data list to only include selected images
        # self.data_list = [data for data in self.data_list if data['img_id'] in selected_images]
        # print('Selected number: ', len(selected_images))

    def select_images_by_total_ratio(self, top_k, json_file_path='work_dirs/coco_image_cb_scs.json'):
        import os
        from PIL import Image
        import matplotlib.pyplot as plt
        import json
    
        if os.path.exists(json_file_path):
            print(f"Loading image scores from {json_file_path} ...")
            with open(json_file_path, 'r') as f:
                image_scores = json.load(f)
        else:
            print(f"No existing JSON file found. Calculating image scores ...")
            """Selects images based on the total ratio scores from all instances within each image."""
            if self.ratio == 'total_ratio':
                print('Selected by total P-2-A Ratio ...')
                self.calculate_ratios()
            elif self.ratio == 'norm_total_ratio':
                print('Selected by norm total P-2-A Ratio ...')
                self.calculate_norm_ratios()
            elif self.ratio == 'total_ratio_norm':
                print('Selected by total P-2-A Ratio with norm...')
                self.calculate_ratios_norm()
            elif self.ratio == 'norm_total_ratio_norm':
                print('Selected by norm total P-2-A Ratio with norm...')
                import time
                start_time = time.time()
                self.calculate_norm_ratios_norm()
        
            image_scores = {}

            for data_info in self.data_list:
                img_id = data_info['img_id']
                instances = data_info['instances']
                
                total_score = 0
                for instance in instances:
                    if instance['ignore_flag'] == 1:
                        continue
                    total_score += instance['ratio']

                if total_score > 0:  # Only consider images with a positive total score
                    image_scores[img_id] = total_score

            # end_time = time.time()  # End timing
            # print("# --------------------------- #")
            # print(f"Function execution took: {end_time - start_time} seconds")
            # print("# --------------------------- #")

            with open(json_file_path, 'w') as f:
                json.dump(image_scores, f)
            print(f"Saved image scores to {json_file_path}")

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        selected_images = [img_id for img_id, _ in sorted_images[:top_k]]
        
        # Update the dataset's data list to only include selected images
        self.data_list = [data for data in self.data_list if str(data['img_id']) in selected_images]
        print('Selected number: ', len(selected_images))


    def full_init(self) -> None:
    
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # get proposals from file
        if self.proposal_file is not None:
            self.load_proposals()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()

        if self.ratio != None:
            if  'calculation' in self.ratio:
                self.select_images_by_ratio(top_k=self.top_k, skip_ratio=0.1)  # 
                size_counts, size_proportions = self.calculate_size_distribution()
                print('Size counts: ', size_counts)
                print('Size Proportions: ', size_proportions)
            elif 'store' in self.ratio:
                self.select_images_by_store_score(self.top_k)
            elif 'ratio' in self.ratio:
                self.select_images_by_total_ratio(self.top_k)
            elif 'ccs' in self.ratio:
                import time
                start_time = time.time() 
                self.select_images_by_store_score_stratified(self.top_k)

                end_time = time.time()  # End timing
                print("# --------------------------- #")
                print(f"Stratified execution took: {end_time - start_time} seconds")
                print("# --------------------------- #")

        print('Dataset size: ', len(self.data_list))


        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True


