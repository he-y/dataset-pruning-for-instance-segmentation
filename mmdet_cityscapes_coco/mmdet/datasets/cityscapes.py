# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/cityscapes.py # noqa
# and https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa

from typing import List

from mmdet.registry import DATASETS
from .coco import CocoDataset
import random

import numpy as np
import pycocotools.mask as mask_utils
import cv2


@DATASETS.register_module()
class CityscapesDataset(CocoDataset):
    """Dataset for Cityscapes."""

    METAINFO = {
        'classes': ('person', 'rider', 'car', 'truck', 'bus', 'train',
                    'motorcycle', 'bicycle'),
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]
    }



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
            all_is_crowd = all([
                instance['ignore_flag'] == 1
                for instance in data_info['instances']
            ])
            if filter_empty_gt and (img_id not in ids_in_cat or all_is_crowd):
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
    
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

                mask = mask_utils.decode(instance['mask'])
                ratio = instance['ratio']
                cls_name = self.METAINFO['classes'][instance['bbox_label'] - 1]

                if img_id not in class_image_totals[cls_name]:
                    class_image_totals[cls_name][img_id] = []

                class_image_totals[cls_name][img_id].append(ratio)

        # Calculate mean ratios and select images
        selected_images = set()
        for cls, img_totals in class_image_totals.items():
            mean_ratios = {img_id: np.mean(ratios) for img_id, ratios in img_totals.items() if ratios}
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

        print('*******************************')
        print('ratio: ', self.ratio)
        # print('top k: ', self.top_k_city)
        print('top k: ', self.top_k)
        print('random percentage: ', self.percentage)
        print('*******************************')

        if self.ratio != None:
            if 'calculation' in self.ratio:
                self.select_images_by_ratio(self.top_k_city, skip_ratio=0.75)
                # self.select_images_by_ratio(top_k=1500, skip_ratio=0.5)
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
    

        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True
