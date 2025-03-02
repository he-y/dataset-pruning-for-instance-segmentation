import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from collections import defaultdict
import random

import torch
import numpy as np
import pycocotools.mask as mask_util
from torchvision import transforms
import cv2

from .generalized_dataset import GeneralizedDataset


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
)

def target_to_coco_ann(target):
    image_id = target['image_id'].item()
    boxes = target['boxes']
    masks = target['masks']
    labels = target['labels'].tolist()

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
    area = boxes[:, 2] * boxes[:, 3]
    area = area.tolist()
    boxes = boxes.tolist()
    
    rles = [
        mask_util.encode(np.array(mask[:, :, None], dtype=np.uint8, order='F'))[0]
        for mask in masks
    ]
    for rle in rles:
        rle['counts'] = rle['counts'].decode('utf-8')

    anns = []
    for i, rle in enumerate(rles):
        anns.append(
            {
                'image_id': image_id,
                'id': i,
                'category_id': labels[i],
                'segmentation': rle,
                'bbox': boxes[i],
                'area': area[i],
                'iscrowd': 0,
            }
        )
    return anns     


def target_to_coco_ann_2(target):
    image_id = target['image_id'].item()
    boxes = target['boxes']
    masks = target['masks']
    labels = target['labels'].tolist()

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
    boxes = boxes.tolist()
    
    rles = [
        mask_util.encode(np.array(mask[:, :, None], dtype=np.uint8, order='F'))[0]
        for mask in masks
    ]
    for rle in rles:
        rle['counts'] = rle['counts'].decode('utf-8')

    areas = []
    perimeters = []
    for mask in masks:
        contours, _ = cv2.findContours(mask.numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_max = max(contours, key=cv2.contourArea)
        if contours:
            area = cv2.contourArea(contour_max)
            perimeter = cv2.arcLength(contour_max, True)

            # vis
            # if isinstance(mask, torch.Tensor):
                # mask = mask.detach().cpu().numpy()
            
            # 确保 mask 是二值化的
            # _, binary_mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
            # 创建一个用于绘制轮廓的副本
            # contour_drawn_mask = np.stack((binary_mask,)*3, axis=-1)  # 转换为3通道以便绘制彩色轮廓
            
            # 在 mask 上用红色线条绘制所有轮廓
            # cv2.drawContours(contour_drawn_mask, contour_max, -1, (0, 0, 255), 2)  # 使用红色轮廓

            # 保存原始 mask 和绘制了轮廓的 mask
            # cv2.imwrite('mask_with_contours.jpg', contour_drawn_mask)
        else:
            area = 0
            perimeter = 0
        areas.append(area)
        perimeters.append(perimeter)

    anns = []
    for i, (rle, area, perim) in enumerate(zip(rles, areas, perimeters)):
        ratio = perim / area if area != 0 else 0  # Avoid division by zero
        anns.append({
            'image_id': image_id,
            'id': i,
            'category_id': labels[i],
            'segmentation': rle,
            'bbox': boxes[i],
            'area': area,
            'iscrowd': 0,
            'ratio': ratio
        })
    return anns


def get_image_total_ratio(voc_dataset):
    # Dictionary to store the total_ratio for each image
    image_ratios = {}
    
    for img_id in voc_dataset.ids:
        target = voc_dataset.get_target(img_id)
        anns = target_to_coco_ann_2(target)
        total_ratio = sum(ann['ratio'] for ann in anns)
        image_ratios[img_id] = total_ratio
    
    return image_ratios

def sort_images_by_ratio(voc_dataset, top_k):
    image_ratios = get_image_total_ratio(voc_dataset)
    # Sort images by their total_ratio in descending order
    sorted_images = sorted(image_ratios.items(), key=lambda x: x[1], reverse=True)
    
    # Select top k images
    top_k_images = sorted_images[:top_k]
    
    return [img_id for img_id, _ in top_k_images]


def get_image_total_score(voc_dataset):
    # Dictionary to store the total_ratio for each image
    image_ratios = {}
    
    for img_id in voc_dataset.ids:
        target = voc_dataset.get_target(img_id)
        anns = target_to_coco_ann_2(target)
        total_ratio = sum(ann['roi_score'] for ann in anns)
        image_ratios[img_id] = total_ratio
    
    return image_ratios

def sort_images_by_score(voc_dataset, top_k):
    image_ratios = get_image_total_score(voc_dataset)
    # Sort images by their total_ratio in descending order
    sorted_images = sorted(image_ratios.items(), key=lambda x: x[1], reverse=True)
    
    # Select top k images
    top_k_images = sorted_images[:top_k]
    
    return [img_id for img_id, _ in top_k_images]


class VOCDataset(GeneralizedDataset):
    # download VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    def __init__(self, data_dir, split, train=False):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.train = train
        
        # instances segmentation task
        id_file = os.path.join(data_dir, "ImageSets/Segmentation/{}.txt".format(split))
        self.ids = [id_.strip() for id_ in open(id_file)]
        self.id_compare_fn = lambda x: int(x.replace("_", ""))
        
        self.ann_file = os.path.join(data_dir, "Annotations/instances_{}.json".format(split))
        self._coco = None
        
        # classes's values must start from 1, because 0 means background in the model
        self.classes = {i: n for i, n in enumerate(VOC_CLASSES, 1)}
        
        checked_id_file = os.path.join(os.path.dirname(id_file), "checked_{}.txt".format(split))
        if train:
            if not os.path.exists(checked_id_file):
                self.make_aspect_ratios()
            self.check_dataset(checked_id_file)
            
    def make_aspect_ratios(self):
        self._aspect_ratios = []
        for img_id in self.ids:
            anno = ET.parse(os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_id)))
            size = anno.findall("size")[0]
            width = size.find("width").text
            height = size.find("height").text
            ar = int(width) / int(height)
            self._aspect_ratios.append(ar)

    def get_image(self, img_id):
        image = Image.open(os.path.join(self.data_dir, "JPEGImages/{}.jpg".format(img_id)))
        return image.convert("RGB")
        
    def get_target(self, img_id):
        masks = Image.open(os.path.join(self.data_dir, 'SegmentationObject/{}.png'.format(img_id)))
        masks = transforms.ToTensor()(masks)
        uni = masks.unique()
        uni = uni[(uni > 0) & (uni < 1)]
        masks = (masks == uni.reshape(-1, 1, 1)).to(torch.uint8)
        
        anno = ET.parse(os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_id)))
        boxes = []
        labels = []
        for obj in anno.findall("object"):
            bndbox = obj.find("bndbox")
            bbox = [int(bndbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]
            name = obj.find("name").text
            label = VOC_CLASSES.index(name) + 1

            boxes.append(bbox)
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels)

        img_id = torch.tensor([self.ids.index(img_id)])
        target = dict(image_id=img_id, boxes=boxes, labels=labels, masks=masks)
        return target
    
    def sample_images(self, num_samples):
        if num_samples > len(self.ids):
            raise ValueError("Requested more samples than available in dataset")

        sampled_indices = random.sample(range(len(self.ids)), num_samples)
        self.ids = [self.ids[idx] for idx in sampled_indices]
    
    # def __getitem__(self, i):
    #     img_id = self.ids[i]
    #     image = self.get_image(img_id)
    #     image = transforms.ToTensor()(image)
    #     target = self.get_target(img_id) if self.train else {}
    #     return [image, target]

    def sample_top_k_images(self, k, method):
        if method == 'ratio':
            top_k_ids = sort_images_by_ratio(self, k)
        elif method == 'roi_score':
            top_k_ids = sort_images_by_score(self, k)
        # Update the dataset to only contain top k images
        self.ids = top_k_ids

    # def sort_images_by_class_ratio(self, top_k):
    #     num_classes = len(VOC_CLASSES)
    #     # k_per_class = max(1, top_k // num_classes)
    #     k_per_class = max(1, top_k // 16)       # 500, 13; 200, 16; 100, 16; 75, 16; 

    #     class_image_ratios = {cls: [] for cls in VOC_CLASSES}

    #     # Calculate ratios per class
    #     for img_id in self.ids:
    #         target = self.get_target(img_id)
    #         anns = target_to_coco_ann_2(target)  # Assuming target_to_coco_ann_2 calculates the ratio per object

    #         for ann in anns:
    #             cls_name = VOC_CLASSES[ann['category_id'] - 1]
    #             class_image_ratios[cls_name].append((img_id, ann['ratio']))

    #     # Sort and select top k_per_class images per class
    #     class_top_images = {}
    #     for cls, img_list in class_image_ratios.items():
    #         sorted_images = sorted(img_list, key=lambda x: x[1], reverse=True)
    #         class_top_images[cls] = sorted_images[:k_per_class]

    #     # Collect and deduplicate images
    #     all_top_images = set()
    #     for images in class_top_images.values():
    #         all_top_images.update([img for img, _ in images])

    #     # Ensure we have exactly top_k images if possible
    #     while len(all_top_images) < top_k:
    #         extra_images = []
    #         for images in class_top_images.values():
    #             extra_images.extend([img for img in images if img[0] not in all_top_images])
    #         extra_images = sorted(extra_images, key=lambda x: x[1], reverse=True)
    #         all_top_images.update([img for img, _ in extra_images[:top_k - len(all_top_images)]])

    #         if not extra_images or len(extra_images) < top_k - len(all_top_images):
    #             break

    #     # Update dataset to include only selected images
    #     print(len(all_top_images))
    #     self.ids = list(all_top_images)[:top_k]

    def sort_images_by_class_ratio(self, top_k):
        k = top_k
        
        num_classes = len(VOC_CLASSES)
        class_image_totals = {cls: {} for cls in VOC_CLASSES}

        initial_n_per_class = max(1, top_k // num_classes)

        for img_id in self.ids:
            target = self.get_target(img_id)
            anns = target_to_coco_ann_2(target)
            for ann in anns:
                cls_name = VOC_CLASSES[ann['category_id'] - 1]
                if img_id not in class_image_totals[cls_name]:
                    class_image_totals[cls_name][img_id] = 0
                class_image_totals[cls_name][img_id] += ann['ratio']

        selected_images = set()

        for cls, img_totals in class_image_totals.items():
            sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1], reverse=True)
            # sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1])
            selected_images.update([img_id for img_id, _ in sorted_imgs[:initial_n_per_class]])

        current_total = len(selected_images)
        while current_total < k:
            needed = k - current_total
            per_class_additional = max(1, needed // num_classes)

            for cls, img_totals in class_image_totals.items():
                sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1], reverse=True)
                # sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1])
                already_selected = len([img for img in sorted_imgs if img[0] in selected_images])
                additional_images = sorted_imgs[already_selected:already_selected + per_class_additional]
                selected_images.update([img_id for img_id, _ in additional_images])

            current_total = len(selected_images)
            if needed == k - current_total:  
                break

        print('Selected number: ', len(selected_images))
        self.ids = list(selected_images)[:k]


    def sort_images_by_class_score(self, top_k):
        k = top_k
        
        num_classes = len(VOC_CLASSES)
        class_image_totals = {cls: {} for cls in VOC_CLASSES}

        initial_n_per_class = max(1, top_k // num_classes)

        for img_id in self.ids:
            target = self.get_target(img_id)
            anns = target_to_coco_ann_2(target)
            for ann in anns:
                cls_name = VOC_CLASSES[ann['category_id'] - 1]
                if img_id not in class_image_totals[cls_name]:
                    class_image_totals[cls_name][img_id] = 0
                class_image_totals[cls_name][img_id] += ann['roi_score']

        selected_images = set()

        for cls, img_totals in class_image_totals.items():
            sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1], reverse=True)
            # sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1])
            selected_images.update([img_id for img_id, _ in sorted_imgs[:initial_n_per_class]])

        current_total = len(selected_images)
        while current_total < k:
            needed = k - current_total
            per_class_additional = max(1, needed // num_classes)

            for cls, img_totals in class_image_totals.items():
                sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1])
                # sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1], reverse=True)
                already_selected = len([img for img in sorted_imgs if img[0] in selected_images])
                additional_images = sorted_imgs[already_selected:already_selected + per_class_additional]
                selected_images.update([img_id for img_id, _ in additional_images])

            current_total = len(selected_images)
            if needed == k - current_total:  
                break

        print('Selected number: ', len(selected_images))
        self.ids = list(selected_images)[:k]

    def sort_images_by_class_ratio_ccs(self, top_k, skip_ratio):
        k = top_k
        num_classes = len(VOC_CLASSES)
        class_image_totals = {cls: {} for cls in VOC_CLASSES}

        initial_n_per_class = max(1, top_k // num_classes)

        for img_id in self.ids:
            target = self.get_target(img_id)
            anns = target_to_coco_ann_2(target)
            for ann in anns:
                cls_name = VOC_CLASSES[ann['category_id'] - 1]
                if img_id not in class_image_totals[cls_name]:
                    class_image_totals[cls_name][img_id] = 0
                class_image_totals[cls_name][img_id] += ann['ratio']

        selected_images = set()

        for cls, img_totals in class_image_totals.items():
            sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1], reverse=True)
            skip_count = int(len(sorted_imgs) * skip_ratio)
            sorted_imgs = sorted_imgs[skip_count:]
            selected_images.update([img_id for img_id, _ in sorted_imgs[:initial_n_per_class]])

        current_total = len(selected_images)
        while current_total < k:
            needed = k - current_total
            per_class_additional = max(1, needed // num_classes)

            for cls, img_totals in class_image_totals.items():
                sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1], reverse=True)
                already_selected = [img for img, _ in sorted_imgs if img in selected_images]
                additional_needed = per_class_additional - len(already_selected)
                additional_images = sorted_imgs[len(already_selected):len(already_selected) + additional_needed]
                selected_images.update([img_id for img_id, _ in additional_images])

            current_total = len(selected_images)
            if needed == k - current_total:  
                break

        print('Selected number: ', len(selected_images))
        self.ids = list(selected_images)[:top_k]


    def sort_images_by_class_ratio_ccs_v2(self, top_k, skip_ratio):
        k = top_k
        num_classes = len(VOC_CLASSES)
        class_image_totals = {cls: {} for cls in VOC_CLASSES}

        # initial_n_per_class = max(1, top_k // num_classes)

        for img_id in self.ids:
            target = self.get_target(img_id)
            anns = target_to_coco_ann_2(target)
            for ann in anns:
                cls_name = VOC_CLASSES[ann['category_id'] - 1]
                if img_id not in class_image_totals[cls_name]:
                    class_image_totals[cls_name][img_id] = 0
                class_image_totals[cls_name][img_id] += ann['ratio']

        selected_images = set()

        n = 1
        num_intervals = 4
        while len(selected_images) < k:
            for cls, img_totals in class_image_totals.items():
                sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1], reverse=True)
                skip_count = int(len(sorted_imgs) * skip_ratio)
                remaining_imgs = sorted_imgs[skip_count:]  # 排除最难的部分
                interval_size = len(remaining_imgs) // num_intervals  # 每个区间的大小
                if cls == 'tvmonitor':
                    print('test...')

                for j in range(num_intervals):
                    count = 0
                    for i in range(j * interval_size, (j + 1) * interval_size):
                        if i < len(remaining_imgs) and count < n:
                            print(i)
                            img_id = remaining_imgs[i][0]
                            if img_id not in selected_images:
                                selected_images.add(img_id)
                                count += 1
                                if len(selected_images) >= k:
                                    break
                    if len(selected_images) >= k:
                        break
                if len(selected_images) >= k:
                    break


        print('Selected number: ', len(selected_images))
        self.ids = list(selected_images)[:top_k]


    def sort_images_by_scale_and_ratio(self, top_k):
        k = top_k
        
        size_thresholds = {'small': 32**2, 'middle': 64**2}  
        size_ratios = {'small': 0.18, 'middle': 0.18, 'large': 0.64}  
        size_budgets = {size: int(size_ratios[size] * top_k + 3) for size in size_ratios}  
        
        num_classes = len(VOC_CLASSES)
        class_image_totals = {cls: {} for cls in VOC_CLASSES}
        # size_counts = {size: 0 for size in size_ratios.keys()}  
        # initial_n_per_class = max(1, top_k // num_classes)
        
        
        for img_id in self.ids:
            target = self.get_target_with_size(img_id, size_thresholds)
            anns = target_to_coco_ann_2(target)
            predominant_size = max(set(target['sizes']), key=target['sizes'].count) 

            for ann in anns:
                cls_name = VOC_CLASSES[ann['category_id'] - 1]
                if img_id not in class_image_totals[cls_name]:
                    class_image_totals[cls_name][img_id] = {'score': 0, 'size': predominant_size}
                class_image_totals[cls_name][img_id]['score'] += ann['ratio']

        selected_images = set()
        size_counts = {'small': 0, 'middle': 0, 'large': 0}
        sorted_imgs_per_class = {
            cls: sorted(img_totals.items(), key=lambda x: x[1]['score'], reverse=True)
            for cls, img_totals in class_image_totals.items()
        }
        n = 2
        while len(selected_images) < top_k:
            for cls, sorted_imgs in sorted_imgs_per_class.items():
                count = 0  # 用于跟踪当前类别已选择的图片数
                for img_id, img_data in list(sorted_imgs):  # 使用list确保迭代时可删除元素
                    if count >= n:
                        break  # 如果已经选择了 n 张图片，则跳出循环
                    
                    img_size = img_data['size']
                    # 在添加到selected_images之前检查图片是否已被选中
                    if img_id not in selected_images and size_counts[img_size] < size_budgets[img_size]:
                        # 更新尺寸的计数器
                        size_counts[img_size] += 1
                        # 将图片ID添加到选定的图片集合中
                        selected_images.add(img_id)

                        # 从列表中移除已选的图像，以避免重复处理
                        sorted_imgs.remove((img_id, img_data))
                        count += 1  # 更新当前类别的选择计数

                    # 检查是否已满足总数
                    if len(selected_images) >= top_k:
                        break


        self.ids = list(selected_images)[:top_k]
        print('Selected number: ', len(selected_images))

    def calculate_size_distribution(self, size_thresholds={'small': 32**2, 'middle': 64**2}):
        size_counts = {'small': 0, 'middle': 0, 'large': 0}

        for img_id in self.ids:
            target = self.get_target(img_id) 
            masks = target['masks']  

            for i in range(masks.shape[0]):
                mask = masks[i].float()
                area = mask.sum().item()

                if area < size_thresholds['small']:
                    size_counts['small'] += 1
                elif area < size_thresholds['middle']:
                    size_counts['middle'] += 1
                else:
                    size_counts['large'] += 1

        total_masks = sum(size_counts.values())
        size_proportions = {size: count / total_masks for size, count in size_counts.items()}

        return size_counts, size_proportions


    def count_object_per_category(self):
        category_counts = {category: 0 for category in VOC_CLASSES}
        
        for img_id in self.ids:
            target = self.get_target(img_id)
            labels = target['labels'].tolist()  # Convert tensor to list
            
            for label in labels:
                category = VOC_CLASSES[label - 1]  # Get category name from label
                category_counts[category] += 1
        
        return category_counts
    
    def count_images_per_category(self):
        category_image_counts = {category: set() for category in VOC_CLASSES}
        
        for img_id in self.ids:
            target = self.get_target(img_id)
            labels = target['labels'].tolist()  # Convert tensor to list
            
            for label in labels:
                category = VOC_CLASSES[label - 1]  # Get category name from label
                category_image_counts[category].add(img_id)  # Add image ID to set
        
        # Convert sets to counts
        category_counts = {category: len(image_ids) for category, image_ids in category_image_counts.items()}
        
        return category_counts

    

    def save_image_filenames(self, output_dir, include_category_counts=False):
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filenames_path = os.path.join(output_dir, 'image_filenames.txt')
        category_counts_path = os.path.join(output_dir, 'category_counts.txt')
        
        with open(filenames_path, 'w') as file:
            for img_id in self.ids:
                filename = f"{img_id}.jpg"  # Assuming all images are JPGs
                file.write(filename + '\n')
        
        if include_category_counts:
            category_counts = self.count_images_per_category()
            with open(category_counts_path, 'w') as file:
                for category, count in category_counts.items():
                    file.write(f"{category}: {count}\n")
        
        print(f"Image filenames have been saved to {filenames_path}")
        if include_category_counts:
            print(f"Category counts have been saved to {category_counts_path}")

    def get_target_with_size(self, img_id, size_thresholds):
        target = self.get_target(img_id)  # 原有的 get_target 函数
        masks = target['masks']
        sizes = []
        for mask in masks:
            area = mask.sum().item()
            if area < size_thresholds['small']:
                sizes.append('small')
            elif area < size_thresholds['middle']:
                sizes.append('middle')
            else:
                sizes.append('large')
        target['sizes'] = sizes
        return target



    
    @property
    def coco(self):
        if self._coco is None:
            from pycocotools.coco import COCO
            self.convert_to_coco_format(overwrite=False)
            self._coco = COCO(self.ann_file)
        return self._coco
    
    def convert_to_coco_format(self, overwrite=False):
        if overwrite or not os.path.exists(self.ann_file):
            print("Generating COCO-style annotations...")
            voc_dataset = VOCDataset(self.data_dir, self.split, True)
            instances = defaultdict(list)
            instances["categories"] = [{"id": i + 1, "name": n} for i, n in enumerate(VOC_CLASSES)]

            ann_id_start = 0
            for image, target in voc_dataset:
                image_id = target["image_id"].item()

                filename = voc_dataset.ids[image_id] + ".jpg"
                h, w = image.shape[-2:]
                img = {"id": image_id, "file_name": filename, "height": h, "width": w}
                instances["images"].append(img)

                # anns = target_to_coco_ann(target)
                anns = target_to_coco_ann_2(target)
                for ann in anns:
                    ann["id"] += ann_id_start
                    instances["annotations"].append(ann)
                ann_id_start += len(anns)

            json.dump(instances, open(self.ann_file, "w"))
            print("Created successfully: {}".format(self.ann_file))
        
