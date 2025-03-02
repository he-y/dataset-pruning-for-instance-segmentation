import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from .generalized_dataset import GeneralizedDataset
       
        
class COCODataset(GeneralizedDataset):
    def __init__(self, data_dir, split, train=False):
        super().__init__()
        from pycocotools.coco import COCO
        
        self.data_dir = data_dir
        self.split = split
        self.train = train
        
        if 'VOC' in self.data_dir:
            self.ann_file = os.path.join(data_dir, "Annotations/instances_{}.json".format(split))  # voc
        elif 'coco' in os.path.basename(self.data_dir):
            self.ann_file = os.path.join(data_dir, "annotations/instances_{}.json".format(split))    # coco
        elif 'cityscapes' in os.path.basename(self.data_dir):
            self.ann_file = os.path.join(data_dir, "annotations/instances{}.json".format(split))    # cityscapes

        self.coco = COCO(self.ann_file)
        self.ids = [str(k) for k in self.coco.imgs]
        
        # classes's values must start from 1, because 0 means background in the model
        self.classes = {k: v["name"] for k, v in self.coco.cats.items()}
        
        checked_id_file = os.path.join(data_dir, "checked_{}.txt".format(split))
        if train:
            if not os.path.exists(checked_id_file):
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()]
            self.check_dataset(checked_id_file)
        
    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]

        if 'VOC' in self.data_dir:
            image = Image.open(os.path.join(self.data_dir, "JPEGImages/", img_info["file_name"]))
        elif 'coco' in os.path.basename(self.data_dir):
            image = Image.open(os.path.join(self.data_dir, "{}".format(self.split), img_info["file_name"]))
        elif 'cityscapes' in os.path.basename(self.data_dir):
            if 'train' in self.split:
                image = Image.open(os.path.join(self.data_dir, "leftImg8bit/train", img_info["file_name"]))
            elif 'val' in self.split:
                image = Image.open(os.path.join(self.data_dir, "leftImg8bit/val", img_info["file_name"]))
            elif 'test' in self.split:
                image = Image.open(os.path.join(self.data_dir, "leftImg8bit/test", img_info["file_name"]))
        return image.convert("RGB")
    
    @staticmethod
    def convert_to_xyxy(boxes): # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax)
        
    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)

        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks)
        return target
    
    def count_images_per_category(self, filename='./results/city_category_counts.json'):
        """
        Counts the number of images and instances per category.
        If the JSON file exists at the specified path, it loads the counts from it.
        If not, it computes the counts, saves them to the JSON file, and returns them.
        
        :param filename: The path to the JSON file for caching counts.
        :return: category_counts, category_instance_counts
        """
        import json
        if os.path.exists(filename):
            print(f"Loading category counts from {filename}")
            with open(filename, 'r') as f:
                data = json.load(f)
                category_counts = data['category_counts']
                category_instance_counts = data['category_instance_counts']
        else:
            category_image_counts = {category: set() for category in self.classes.values()}
            category_instance_counts = {category: 0 for category in self.classes.values()}
            
            for img_id in self.ids:
                target = self.get_target(img_id)
                labels = target['labels'].tolist()  # Convert tensor to list
                
                for label in labels:
                    category = self.classes[label]  # Get category name from label using class dictionary
                    category_image_counts[category].add(img_id)  # Add image ID to set
                    category_instance_counts[category] += 1
            
            # Convert sets to counts
            category_counts = {category: len(image_ids) for category, image_ids in category_image_counts.items()}
            
            # Save the counts to a JSON file
            data = {
                'category_counts': category_counts,
                'category_instance_counts': category_instance_counts
            }
            with open(filename, 'w') as f:
                json.dump(data, f)
            print(f"Category counts saved to {filename}")
        
        return category_counts, category_instance_counts

    
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
        size_proportions = {size: count / total_masks for size, count in size_counts.items() if total_masks > 0}

        return size_counts, size_proportions

    
    def get_target2(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        masks = []
        roi_scores = []  # 新增：用于存储每个注释的 roi_score

        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                mask = self.coco.annToMask(ann)
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)
                roi_scores.append(ann.get('roi_score', 0))  # 从注释中安全获取 roi_score

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)
            roi_scores = torch.tensor(roi_scores, dtype=torch.float32)  # 将 roi_scores 转换为 Tensor

        target = {
            "image_id": torch.tensor([img_id]),
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "roi_scores": roi_scores  # 将 roi_scores 包含在目标字典中
        }
        return target

    def vis_images_scores(self, image_scores):

        # Extracting the image indices and their corresponding scores
        image_indices = [str(index) for index in image_scores.keys()]
        scores = [score for score in image_scores.values()]

        # Creating the bar plot with adjustments
        plt.figure(figsize=(14, 8))
        bars = plt.bar(image_indices, scores, color='blue', width=10)  # Adjust bar width here

        # Customizing the x-axis
        plt.xticks(ticks=np.arange(0, len(image_indices), 100), labels=np.arange(1, len(image_indices)+1, 100))  # showing every 100th label

        # Customizing the y-axis
        if max(scores) > 10:
            plt.yticks(ticks=np.arange(0, max(scores) + 10, 10))  # setting y-axis intervals
            plt.ylim(0, max(scores) + 2)  # setting y-axis limits
        else:
            # plt.yticks(ticks=np.arange(0, max(scores) + 0.5))  # setting y-axis intervals
            plt.ylim(0, max(scores) + 0.01)  # setting y-axis limits

        # Creating the bar plot
        plt.xlabel('Image Index')
        plt.ylabel('Boundary Complexity Score')
        plt.title('Score Distribution of Selected Images')
        plt.savefig('./vis/Selected_Images_Scores.png')

    def vis_specific_category_scores(self, category_id):
        # Initialize all image scores to 0 using the correct type for keys
        image_scores = {int(img_id): 0 for img_id in self.ids}  # Assuming self.ids contains strings, convert them to int

        # Filter and collect scores for the specified category
        for img_id in self.ids:
            img_id = int(img_id)  # Ensure img_id is an integer
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                cat_id = ann['category_id']
                if cat_id == category_id:  # Filter by category
                    score = ann.get('norm_p2a_ratio', 0)
                    image_scores[img_id] += score  # img_id is already an int, so this should match keys in image_scores

        # Now visualize the scores of the filtered category
        self.vis_images_scores(image_scores)

    def visualize_normalized_category_scores_and_instance_count(self, category_instance_counts, category_normalized_scores):
        categories = list(category_normalized_scores.keys())
        sorted_categories = sorted(categories)  # Sort categories by cat_id
        normalized_scores = [category_normalized_scores[cat_id] for cat_id in sorted_categories]
        instance_counts = [category_instance_counts[cat_id] for cat_id in sorted_categories]

        x = np.arange(len(categories))  # the label locations

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plotting normalized scores on the left Y-axis
        ax1.set_xlabel('Category ID')
        ax1.set_ylabel('Normalized Score Sum', color='tab:blue')
        rects1 = ax1.bar(x, normalized_scores, width=0.35, label='Normalized Score Sum', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax1.set_ylim([0, max(normalized_scores) * 5])  # 例如，Y轴最大值设置为分数最大值的1.1倍
        ax1.set_yticks(np.arange(0, max(normalized_scores) * 5, step=1))  # 设置刻度间隔为0.1

        # Adding the right Y-axis for instance count
        ax2 = ax1.twinx()
        ax2.set_ylabel('Instance Count', color='#FFA07A')
        rects2 = ax2.bar(x + 0.35, instance_counts, width=0.35, label='Instance Count', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Adding title and custom x-axis tick labels
        ax1.set_title('Total Normalized Score and Instance Count per Category')
        ax1.set_xticks(x + 0.35 / 2)
        ax1.set_xticklabels(sorted_categories)

        # Adding value labels above the bars
        def add_value_labels(rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        add_value_labels(rects1, ax1)
        add_value_labels(rects2, ax2)

        fig.tight_layout()  # To adjust layout for the two axes
        plt.savefig('./vis/class_score.png')

    def visualize_random_sampled_instance_areas(self, num_samples=1000):
        import random
        """
        Randomly samples a number of instance areas and visualizes their distribution.
        :param num_samples: The number of instances to sample for visualization.
        """
        # Calculate all areas first
        all_areas = []
    
        # Iterate through all image IDs to calculate areas
        for img_id in self.ids:
            target = self.get_target(img_id)
            masks = target['masks']  # Assume masks is a tensor of shape [N, H, W]
            
            # Calculate area for each instance in the image
            for i in range(masks.shape[0]):
                mask = masks[i].float()  # Convert mask to float
                area = mask.sum().item()  # Sum of mask values (area)
                all_areas.append(area)
        
        # Randomly sample areas
        sampled_areas = random.sample(all_areas, len(all_areas))
        
        # Visualize the area distribution using a histogram
        plt.figure(figsize=(10, 6))
        plt.hist(sampled_areas, bins=20, color='lightblue', edgecolor='black', density=True)
        plt.xlabel('Area Size')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Instance Areas for COCO dataset')
        plt.tight_layout()
        plt.savefig('./results/figs/random_all_instance_areas.png')

    def visualize_instance_areas_image_distribution(self):
        import json
        """
        Randomly samples a number of instance areas and visualizes their distribution.
        :param num_samples: The number of instances to sample for visualization.
        """
        filename = './results/city_area_inform.json'

        if os.path.exists(filename):
            # Load data from the JSON file
            print(f"Loading instance area ratios from {filename}")
            with open(filename, 'r') as f:
                instance_area_ratios = json.load(f)
        else:
            # Calculate all areas first
            instance_area_ratios = []
        
            # Iterate through all image IDs to calculate area ratios
            for img_id in self.ids:
                target = self.get_target(img_id)
                masks = target['masks']  # Assume masks is a tensor of shape [N, H, W]
                image_area = self.coco.imgs[int(img_id)]['height'] * self.coco.imgs[int(img_id)]['width']  # Total image area (H * W)
                
                # Calculate instance_area/image_area ratio for each instance
                for i in range(masks.shape[0]):
                    mask = masks[i].float()  # Convert mask to float
                    instance_area = mask.sum().item()  # Sum of mask values (instance area)
                    area_ratio = instance_area / image_area  # Ratio of instance area to image area
                    instance_area_ratios.append(area_ratio)

            # Save the instance_area_ratios to a JSON file
            with open(filename, 'w') as f:
                json.dump(instance_area_ratios, f)
        
        # Randomly sample area ratios
        # sampled_ratios = random.sample(instance_area_ratios, len(instance_area_ratios))
        
        # Create histogram data (without plotting yet) to get the heights and bin edges
        counts, bins, patches = plt.hist(instance_area_ratios, bins=20)

        # Convert counts to percentages (each count divided by total number of samples, then multiplied by 100)
        total_count = len(instance_area_ratios)
        # percentages = (counts / total_count) * 100
        percentages = np.round((counts / total_count) * 100, 2)
        
        # Plot the percentages
        bin_centers = 0.5 * (bins[1:] + bins[:-1])  # Calculate bin centers for correct placement of the bars
        plt.bar(bin_centers, percentages, width=(bins[1] - bins[0]))

        # Define custom colors for the bins
        custom_colors = ['#74ccac'] * 1 + ['#f89b75'] * 4 + ['#a6afca'] * 6 + ['#e694c8'] * 9
        
        # Apply the custom colors to each bar
        for patch, color in zip(patches, custom_colors):
            patch.set_facecolor(color)
        
        # Convert density to percentage by multiplying heights by 100
        # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}%'.format(y)))

        
        plt.xlabel('Instance Area / Image Area')  # 横轴：实例面积与图像面积的比值
        plt.ylabel('Percentage')  # 纵轴：百分比频率
        plt.title(f'Distribution of object size (Percentage) in COCO')
        plt.tight_layout()
        plt.savefig('./results/figs/distribution_instance_areas_images_ratio.png')

    def sort_images_by_class_score(self, top_k):
        k = top_k
        
        num_classes = len(self.classes)
        class_image_totals = {cls: {} for cls in self.classes.values()}

        initial_n_per_class = max(1, top_k // num_classes)

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            # target = self.get_target2(img_id)
            # anns = anns['annotations']
            for ann in anns:
                cls_name = self.classes[ann['category_id']]
                if img_id not in class_image_totals[cls_name]:
                    class_image_totals[cls_name][img_id] = 0
                class_image_totals[cls_name][img_id] += ann.get('roi_score', 0)  # Assuming roi_score is pre-defined

        selected_images = set()

        for cls, img_totals in class_image_totals.items():
            # sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1], reverse=True)
            sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1])
            selected_images.update([img_id for img_id, _ in sorted_imgs[:initial_n_per_class]])

        current_total = len(selected_images)
        while current_total < k:
            needed = k - current_total
            per_class_additional = max(1, needed // num_classes)

            for cls, img_totals in class_image_totals.items():
                # sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1], reverse=True)
                sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1])
                already_selected = len([img for img in sorted_imgs if img[0] in selected_images])
                additional_images = sorted_imgs[already_selected:already_selected + per_class_additional]
                selected_images.update([img_id for img_id, _ in additional_images])

            current_total = len(selected_images)
            if needed == k - current_total:  
                break

        print('Selected number: ', len(selected_images))
        self.ids = list(selected_images)[:k]


    def sort_images_by_total_ratio(self, top_k):
        k = top_k
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            # if self.coco.imgs[img_id]['file_name'] == '2007_000528.jpg':
            #     print('test ...')

            total_score = 0
            for ann in anns:
                total_score += ann.get('p2a_ratio', 0)
                # print(ann.get('p2a_ratio', 0))

            image_scores[img_id] = total_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        # self.vis_images_scores(image_scores)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images

    def sort_images_by_total_norm_ratio(self, top_k):
        k = top_k
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_score = 0
            for ann in anns:
                total_score += ann.get('norm_p2a_ratio', 0)

            image_scores[img_id] = total_score

        # self.vis_images_scores(image_scores)
        # self.vis_specific_category_scores(category_id=15)
        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=False)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images

    def sort_images_by_total_boundary_ratio(self, top_k):
        k = top_k
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_score = 0
            for ann in anns:
                total_score += ann.get('boundary_p2a_ratio', 0)

            image_scores[img_id] = total_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images

    def sort_images_by_total_norm_ratio_norm(self, top_k):
        k = top_k
        image_scores = {}
        category_scores = {}
        category_instance_counts = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('norm_p2a_ratio', 0)
                if cat_id in category_scores:
                    category_scores[cat_id] += score
                    category_instance_counts[cat_id] += 1
                else:
                    category_scores[cat_id] = score
                    category_instance_counts[cat_id] = 1

        
        ###
        category_normalized_scores = {cat_id: 0 for cat_id in category_scores.keys()}
        ###
        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_normalized_score = 0
            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('norm_p2a_ratio', 0)
                if category_scores[cat_id] > 0:
                    normalized_score = score / category_scores[cat_id]
                    total_normalized_score += normalized_score

                    ###
                    category_normalized_scores[cat_id] += normalized_score
                    ###

            image_scores[img_id] = total_normalized_score
        # Sort categories and corresponding data by cat_id (category ID)

        # self.visualize_normalized_category_scores_and_instance_count(category_instance_counts, category_normalized_scores)


        # self.vis_images_scores(image_scores)
        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        # sorted_images = sorted(image_scores.items(), key=lambda x: x[1])
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images

    def sort_images_by_total_boundary_ratio_norm(self, top_k):
        k = top_k
        image_scores = {}
        category_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('boundary_p2a_ratio', 0)
                if cat_id in category_scores:
                    category_scores[cat_id] += score
                else:
                    category_scores[cat_id] = score

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_normalized_score = 0
            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('boundary_p2a_ratio', 0)
                if category_scores[cat_id] > 0:
                    normalized_score = score / category_scores[cat_id]
                    total_normalized_score += normalized_score

            image_scores[img_id] = total_normalized_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images

    def sort_images_by_total_norm_ratio_count(self, top_k):
        k = top_k
        image_scores = {}
        category_counts = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                cat_id = ann['category_id']
                if cat_id in category_counts:
                    category_counts[cat_id] += 1
                else:
                    category_counts[cat_id] = 1

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_normalized_score = 0
            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('norm_p2a_ratio', 0)
                if category_counts[cat_id] > 0:
                    normalized_score = score / category_counts[cat_id]
                    total_normalized_score += normalized_score

            image_scores[img_id] = total_normalized_score

        # 对图像进行排序并选择 top_k
        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images


    def sort_images_by_total_roi_score(self, top_k):
        k = top_k
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_score = 0
            for ann in anns:
                total_score += ann.get('roi_score', 0)

            image_scores[img_id] = total_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images

    def sort_images_by_total_roi_boundary_score(self, top_k):
        k = top_k
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_score = 0
            for ann in anns:
                total_score += ann.get('roi_boundary_score', 0)

            image_scores[img_id] = total_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images

    def sort_images_by_total_roi_boundary_norm_score(self, top_k):
        k = top_k
        image_scores = {}
        category_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('roi_boundary_score', 0)
                if cat_id in category_scores:
                    category_scores[cat_id] += score
                else:
                    category_scores[cat_id] = score

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_normalized_score = 0
            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('roi_boundary_score', 0)
                if category_scores[cat_id] > 0:
                    normalized_score = score / category_scores[cat_id]
                    total_normalized_score += normalized_score

            image_scores[img_id] = total_normalized_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images


    def sort_images_by_total_roi_el2n_score(self, top_k):
        k = top_k
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_score = 0
            for ann in anns:
                total_score += ann.get('roi_el2n_score', 0)

            image_scores[img_id] = total_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images


    def sort_images_by_total_roi_el2n_boundary_score(self, top_k):
        k = top_k
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_score = 0
            for ann in anns:
                total_score += ann.get('roi_el2n_boundary_score', 0)

            image_scores[img_id] = total_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images

    def sort_images_by_total_roi_el2n_norm_score(self, top_k):
        k = top_k
        image_scores = {}
        category_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('roi_el2n_score', 0)
                if cat_id in category_scores:
                    category_scores[cat_id] += score
                else:
                    category_scores[cat_id] = score

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_normalized_score = 0
            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('roi_el2n_score', 0)
                if category_scores[cat_id] > 0:
                    normalized_score = score / category_scores[cat_id]
                    total_normalized_score += normalized_score

            image_scores[img_id] = total_normalized_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images

    def sort_images_by_total_roi_el2n_boundary_norm_score(self, top_k):
        k = top_k
        image_scores = {}
        category_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('roi_el2n_boundary_score', 0)
                if cat_id in category_scores:
                    category_scores[cat_id] += score
                else:
                    category_scores[cat_id] = score

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_normalized_score = 0
            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('roi_el2n_boundary_score', 0)
                if category_scores[cat_id] > 0:
                    normalized_score = score / category_scores[cat_id]
                    total_normalized_score += normalized_score

            image_scores[img_id] = total_normalized_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images

    def sort_images_by_total_ratio_norm(self, top_k):
        k = top_k
        image_scores = {}
        category_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('ratio', 0)
                if cat_id in category_scores:
                    category_scores[cat_id] += score
                else:
                    category_scores[cat_id] = score

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_normalized_score = 0
            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('ratio', 0)
                if category_scores[cat_id] > 0:
                    normalized_score = score / category_scores[cat_id]
                    total_normalized_score += normalized_score

            image_scores[img_id] = total_normalized_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images


    def sort_images_by_total_forgetting_score(self, top_k):    
        k = top_k
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_score = 0
            for ann in anns:
                total_score += ann.get('forgetting_score', 0)

            image_scores[img_id] = total_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images


    def sort_images_by_total_aum_score(self, top_k):    
        k = top_k
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_score = 0
            for ann in anns:
                total_score += ann.get('roi_aum_score', 0)

            image_scores[img_id] = total_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=False)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images

    def sort_images_by_total_aum_score_reverse(self, top_k):    
        k = top_k
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_score = 0
            for ann in anns:
                total_score += ann.get('roi_aum_score', 0)

            image_scores[img_id] = total_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images


    def sort_images_by_total_aum_reverse_norm(self, top_k):
        k = top_k
        image_scores = {}
        category_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('roi_aum_score', 0)
                if cat_id in category_scores:
                    category_scores[cat_id] += score
                else:
                    category_scores[cat_id] = score

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_normalized_score = 0
            for ann in anns:
                cat_id = ann['category_id']
                score = ann.get('roi_aum_score', 0)
                if category_scores[cat_id] > 0:
                    normalized_score = score / category_scores[cat_id]
                    total_normalized_score += normalized_score

            image_scores[img_id] = total_normalized_score

        sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_images = [img_id for img_id, _ in sorted_images[:k]]
        
        print('Selected number: ', len(selected_images))
        self.ids = selected_images


    def sort_images_by_class_score_ccs(self, top_k, skip_ratio):
        k = top_k
        
        num_classes = len(self.classes)
        class_image_totals = {cls: {} for cls in self.classes.values()}

        initial_n_per_class = max(1, top_k // num_classes)

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                cls_name = self.classes[ann['category_id']]
                if img_id not in class_image_totals[cls_name]:
                    class_image_totals[cls_name][img_id] = 0
                class_image_totals[cls_name][img_id] += ann.get('roi_score', 0)  # Assuming roi_score is pre-defined

        selected_images = set()
        
        for cls, img_totals in class_image_totals.items():
            sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1], reverse=True)
            skip_count = int(len(sorted_imgs) * skip_ratio)
            print('Skip count: ', str(skip_count))
            sorted_imgs = sorted_imgs[skip_count:]
            selected_images.update([img_id for img_id, _ in sorted_imgs[:initial_n_per_class]])

        current_total = len(selected_images)
        while current_total < k:
            needed = k - current_total
            per_class_additional = max(1, needed // num_classes)

            for cls, img_totals in class_image_totals.items():
                sorted_imgs = sorted(img_totals.items(), key=lambda x: x[1], reverse=True)
                already_selected = len([img for img in sorted_imgs if img[0] in selected_images])
                additional_images = sorted_imgs[already_selected:already_selected + per_class_additional]
                selected_images.update([img_id for img_id, _ in additional_images])

            current_total = len(selected_images)
            if needed == k - current_total:  
                break

        print('Selected number: ', len(selected_images))
        self.ids = list(selected_images)[:k]

    # stratified entropy
    def stratified_sampling_by_roi_score(self, top_k, pruning_hard_rate):
        stratas = 50
        print('Using stratified sampling for image selection...')
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_score = sum(ann.get('roi_score', 0) for ann in anns)
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

        strata_num = []
        ##### calculate number for each strata #####
        for i in range(stratas):
            start, end = bin_range(i)
            num = torch.logical_and(scores >= start, scores < end).sum()
            strata_num.append(num)

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

        ##### sampling in each strata #####
        selected_images = []
        image_ids = np.array(list(image_scores.keys()))
        for i in range(stratas):
            start, end = bin_range(i)
            mask = torch.logical_and(scores >= start, scores < end)
            pool = image_ids[mask.numpy()]
            rand_index = np.random.permutation(pool.shape[0])
            selected_images.extend(pool[rand_index][:budgets[i]].tolist())

        print('Selected number:', len(selected_images))
        self.ids = selected_images

    # stratified el2n
    def stratified_sampling_by_el2n(self, top_k, pruning_hard_rate):
        stratas = 50
        print('Using stratified sampling for image selection...')
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_score = sum(ann.get('roi_el2n_score', 0) for ann in anns)
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

        strata_num = []
        ##### calculate number for each strata #####
        for i in range(stratas):
            start, end = bin_range(i)
            num = torch.logical_and(scores >= start, scores < end).sum()
            strata_num.append(num)

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

        ##### sampling in each strata #####
        selected_images = []
        for i in range(stratas):
            start, end = bin_range(i)
            mask = torch.logical_and(scores >= start, scores < end)
            pool = image_ids[mask.numpy()]
            rand_index = np.random.permutation(pool.shape[0])
            selected_images.extend(pool[rand_index][:budgets[i]].tolist())

        print('Selected number:', len(selected_images))
        self.ids = selected_images

    # stratified reverse aum
    def stratified_sampling_by_reverse_aum_score(self, top_k, pruning_hard_rate):
        stratas = 50
        print('Using stratified sampling for image selection...')
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_score = sum(ann.get('roi_el2n_score', 0) for ann in anns)
            image_scores[img_id] = total_score

        scores = torch.tensor(list(image_scores.values()), dtype=torch.float32)
        image_ids = np.array(list(image_scores.keys()))

        if pruning_hard_rate > 0:
            sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
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

        strata_num = []
        ##### calculate number for each strata #####
        for i in range(stratas):
            start, end = bin_range(i)
            num = torch.logical_and(scores >= start, scores < end).sum()
            strata_num.append(num)

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

        ##### sampling in each strata #####
        selected_images = []
        for i in range(stratas):
            start, end = bin_range(i)
            mask = torch.logical_and(scores >= start, scores < end)
            pool = image_ids[mask.numpy()]
            rand_index = np.random.permutation(pool.shape[0])
            selected_images.extend(pool[rand_index][:budgets[i]].tolist())

        print('Selected number:', len(selected_images))
        self.ids = selected_images


    # stratified aum
    def stratified_sampling_by_aum_score(self, top_k, pruning_hard_rate):
        stratas = 50
        print('Using stratified sampling for image selection...')
        image_scores = {}

        for img_id in self.ids:
            img_id = int(img_id)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            total_score = sum(ann.get('roi_el2n_score', 0) for ann in anns)
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

        strata_num = []
        ##### calculate number for each strata #####
        for i in range(stratas):
            start, end = bin_range(i)
            num = torch.logical_and(scores >= start, scores < end).sum()
            strata_num.append(num)

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

        ##### sampling in each strata #####
        selected_images = []
        for i in range(stratas):
            start, end = bin_range(i)
            mask = torch.logical_and(scores >= start, scores < end)
            pool = image_ids[mask.numpy()]
            rand_index = np.random.permutation(pool.shape[0])
            selected_images.extend(pool[rand_index][:budgets[i]].tolist())

        print('Selected number:', len(selected_images))
        self.ids = selected_images

    def save_image_filenames(self, output_dir, include_category_counts=False):
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Saving image filenames
        filenames_path = os.path.join(output_dir, 'image_filenames.txt')
        with open(filenames_path, 'w') as file:
            for img_id in self.ids:
                img_info = self.coco.imgs[int(img_id)]
                filename = img_info["file_name"]
                file.write(filename + '\n')
        
        # Optionally saving category counts
        if include_category_counts:
            category_counts = self.count_images_per_category()
            category_counts_path = os.path.join(output_dir, 'category_counts.txt')
            with open(category_counts_path, 'w') as file:
                for category, count in category_counts.items():
                    file.write(f"{category}: {count}\n")
        
        print(f"Image filenames have been saved to {filenames_path}")
        if include_category_counts:
            print(f"Category counts have been saved to {category_counts_path}")


    
    