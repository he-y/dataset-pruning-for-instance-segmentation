import sys
import time

import torch
import json
import numpy as np

import pycocotools.mask as mask_util
import cv2
import math

from .utils import Meter, TextArea
try:
    from .datasets import CocoEvaluator, prepare_for_coco, evaluate_per_category
except:
    pass


def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch
                   
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        S = time.time()
        
        losses = model(image, target)

        total_loss = sum(losses.values())
        m_m.update(time.time() - S)
            
        S = time.time()
        total_loss.backward()
        b_m.update(time.time() - S)
        
        optimizer.step()
        optimizer.zero_grad()

        if num_iters % args.print_freq == 0:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
           
    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    return A / iters
            

def evaluate(model, data_loader, device, args, generate=True):
    iter_eval = None
    if generate:
        iter_eval = generate_results(model, data_loader, device, args)

    dataset = data_loader #
    iou_types = ["bbox", "segm"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(args.results, map_location="cpu")

    S = time.time()
    coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    # temp = sys.stdout
    # sys.stdout = TextArea()

    # coco_evaluator.summarize()

    # output = sys.stdout
    # sys.stdout = temp

    output = 'test'
    coco_evaluator.summarize()

    # coco_eval = coco_evaluator.coco_eval["bbox"]
    # coco_eval = coco_evaluator.coco_eval["segm"]
    # # calculate COCO info for all classes
    # coco_stats, print_coco = summarize(coco_eval)

    # # calculate voc info for every classes(IoU=0.5)

    # category_index = dataset.classes

    # stats, _ = summarize(coco_eval, catId=2)

    # voc_map_info_list = []
    # for i in range(len(category_index)):
    #     print(category_index[i + 1])
    #     stats, _ = summarize(coco_eval, catId=i)
    #     voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))

    # print_voc = "\n".join(voc_map_info_list)
    # print(print_voc)

        
    return output, iter_eval


def draw_voc_inform(category_counts, category_ratio_sums):
    # Define VOC classes and create a mapping from class IDs to names
    import matplotlib.pyplot as plt
    import numpy as np

    VOC_CLASSES = (
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"
    )
    CLASS_ID_TO_NAME = {i + 1: name for i, name in enumerate(VOC_CLASSES)}

    # Prepare data for plotting
    categories = sorted(category_counts.keys())
    counts = [category_counts[cat_id] for cat_id in categories]
    ratio_sums = [category_ratio_sums[cat_id] for cat_id in categories]
    category_names = [CLASS_ID_TO_NAME.get(cat_id, 'Unknown') for cat_id in categories]

    # Plotting the data
    x = np.arange(len(categories))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot instance counts
    ax1.bar(x - width/2, counts, width, label='Instance Count', color='skyblue')
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Number of Instances', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    # Plot sum of ratios on the same x-axis but different y-axis
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, ratio_sums, width, label='Sum of Ratios', color='lightgreen')
    ax2.set_ylabel('Sum of Perimeter-to-Area Ratios', color='lightgreen')
    ax2.tick_params(axis='y', labelcolor='lightgreen')

    # Set x-ticks and labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(category_names, rotation=45)

    # Add title and legend
    plt.title('Instance Counts and Sum of Ratios per Category')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    plt.tight_layout()
    plt.savefig('./results/figs/category_counts_and_norm_ratios_norm.png')
    plt.show()

# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args):
    # iters = len(data_loader) if args.iters < 0 else args.iters
    iters = len(data_loader)
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}

        S = time.time()
        #torch.cuda.synchronize()
        output = model(image)
        m_m.update(time.time() - S)
        
        prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output[0].items()}}
        coco_results.extend(prepare_for_coco(prediction))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    torch.save(coco_results, args.results)
        
    return A / iters

def P2A_ratio_calculation(target):
    image_id = target['image_id'].item()
    boxes = target['boxes']
    masks = target['masks']
    labels = target['labels'].tolist()

    # if image_id == 17:
    #     print('test ...')

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
    boxes = boxes.tolist()
    
    rles = [
        mask_util.encode(np.array(mask.cpu()[:, :, None], dtype=np.uint8, order='F'))[0]
        for mask in masks
    ]
    for rle in rles:
        rle['counts'] = rle['counts'].decode('utf-8')

    areas = []
    perimeters = []
    for mask in masks:
        contours, _ = cv2.findContours(mask.cpu().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour_max = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour_max)
            perimeter = cv2.arcLength(contour_max, True)
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


def norm_P2A_ratio_calculation(target):
    image_id = target['image_id'].item()
    boxes = target['boxes']
    masks = target['masks']
    labels = target['labels'].tolist()

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
    boxes = boxes.tolist()
    
    rles = [
        mask_util.encode(np.array(mask.cpu()[:, :, None], dtype=np.uint8, order='F'))[0]
        for mask in masks
    ]
    for rle in rles:
        rle['counts'] = rle['counts'].decode('utf-8')

    areas = []
    perimeters = []
    for mask in masks:
        contours, _ = cv2.findContours(mask.cpu().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_max = max(contours, key=cv2.contourArea)
        if contours:
            area = cv2.contourArea(contour_max)
            perimeter = cv2.arcLength(contour_max, True)
        else:
            area = 0
            perimeter = 0
        areas.append(area)
        perimeters.append(perimeter)

    anns = []
    for i, (rle, area, perim) in enumerate(zip(rles, areas, perimeters)):
        if area != 0:
            circle_perimeter = 2 * math.sqrt(math.pi * area)
            normalized_ratio = perim / circle_perimeter
        else:
            normalized_ratio = 0  # Avoid division by zero for shapes with no area

        anns.append({
            'image_id': image_id,
            'id': i,
            'category_id': labels[i],
            'segmentation': rle,
            'bbox': boxes[i],
            'area': area,
            'iscrowd': 0,
            'ratio': normalized_ratio
        })
    return anns

def norm_P2A_ratio_norm_calculation(target):

    image_id = target['image_id'].item()
    boxes = target['boxes']
    masks = target['masks']
    labels = target['labels'].tolist()

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
    boxes = boxes.tolist()

    rles = [
        mask_util.encode(np.array(mask.cpu()[:, :, None], dtype=np.uint8, order='F'))[0]
        for mask in masks
    ]
    for rle in rles:
        rle['counts'] = rle['counts'].decode('utf-8')

    areas = []
    perimeters = []
    for mask in masks:
        contours, _ = cv2.findContours(mask.cpu().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour_max = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour_max)
            perimeter = cv2.arcLength(contour_max, True)
        else:
            area = 0
            perimeter = 0
        areas.append(area)
        perimeters.append(perimeter)

    anns = []
    for i, (rle, area, perim) in enumerate(zip(rles, areas, perimeters)):
        if area != 0:
            circle_perimeter = 2 * math.sqrt(math.pi * area)
            normalized_ratio = perim / circle_perimeter
        else:
            normalized_ratio = 0  # Avoid division by zero for shapes with no area

        anns.append({
            'image_id': image_id,
            'id': i,
            'category_id': labels[i],
            'segmentation': rle,
            'bbox': boxes[i],
            'area': area,
            'iscrowd': 0,
            'ratio': normalized_ratio
        })

    return anns



def boundary_P2A_ratio_calculation(target, erosion_size=10):
    image_id = target['image_id'].item()
    boxes = target['boxes']
    masks = target['masks']
    labels = target['labels'].tolist()

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
    boxes = boxes.tolist()
    
    rles = [
        mask_util.encode(np.array(mask.cpu()[:, :, None], dtype=np.uint8, order='F'))[0]
        for mask in masks
    ]
    for rle in rles:
        rle['counts'] = rle['counts'].decode('utf-8')

    areas = []
    perimeters = []
    for mask in masks:
        # Convert the mask to a numpy array
        mask_np = mask.cpu().numpy()

        # Create the smaller mask by erosion
        kernel = np.ones((erosion_size, erosion_size), np.uint8)
        mask_small = cv2.erode(mask_np, kernel, iterations=1)

        # Calculate the area of mask - mask_small
        mask_diff = cv2.subtract(mask_np, mask_small)
        area_diff = np.sum(mask_diff)

        # Calculate perimeters for both mask and mask_small
        contour_info = [
            cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for image in [mask_np, mask_small]
        ]
        
        total_perimeter = 0
        for contours, _ in contour_info:
            if contours:
                contour_max = max(contours, key=cv2.contourArea)
                total_perimeter += cv2.arcLength(contour_max, True)

        areas.append(area_diff)
        perimeters.append(total_perimeter)

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


def get_p2a_ratio_scores(data_loader, model, device, get_score):
    # from collections import defaultdict

    # Initialize dictionaries to hold counts and ratio sums
    # category_counts = defaultdict(int)
    # category_ratio_sums = defaultdict(list) 

    # category_scores = {}
    # cls_norm = True
    
    with torch.no_grad():  # Disable gradient computation
        print('Dataset size: ', len(data_loader))
        for i, (image, target) in enumerate(data_loader):
            image = image.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            
            ### 
            if get_score == 'ratio':
                anns = P2A_ratio_calculation(target)   
            elif get_score == 'norm_ratio':
                anns = norm_P2A_ratio_calculation(target)  
            elif get_score == 'boundary_ratio':
                anns = boundary_P2A_ratio_calculation(target)  
            ratios = [ann['ratio'] for ann in anns]

            if ratios!=None:
                    update_json_annotation(target, ratios, data_loader.ann_file, get_score)
            
            # for an in ann:
            #     category_id = an['category_id']
            #     ratio = an['ratio']
            #     category_counts[category_id] += 1
            #     category_ratio_sums[category_id] += ratio

            # if i % 100 == 0:
            #     print(i)

        # draw_voc_inform(category_counts, category_ratio_sums)

    return None


def get_p2a_ratio_scores_coco(data_loader, model, device, get_score, batch_size=50000):
    # 先读取整个 JSON 文件到内存
    with open(data_loader.ann_file, 'r') as f:
        data = json.load(f)
    
    # 预处理，建立 image_id 到 annotations 的快速索引
    annotations_index = {}
    for ann in data['annotations']:
        if ann['image_id'] not in annotations_index:
            annotations_index[ann['image_id']] = []
        annotations_index[ann['image_id']].append(ann)

    updates_count = 0
    processed_images = 0

    with torch.no_grad():  # Disable gradient computation
        print('Dataset size: ', len(data_loader))
        for i, (image, target) in enumerate(data_loader):
            image = image.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            
            # 模拟得分计算
            ann = P2A_ratio_calculation(target)  # 实现这个函数以适应你的具体逻辑
            ratios = [ann['ratio'] for ann in ann]

            # 更新 JSON 数据
            # update_json_annotation_fast(target, ratios, annotations_index, get_score)
            
            # processed_images += 1
            
            # # Batch save to JSON file
            # if processed_images % batch_size == 0 or processed_images == len(data_loader):
            #     with open(data_loader.ann_file, 'w') as f:
            #         json.dump(data, f)
            #     print(f"Batch {updates_count + 1} saved.")
            #     updates_count += 1

            if i % 100 == 0:
                print(i)
            
            # draw_voc_inform()

    return None

def get_roi_scores(data_loader, model, device, get_score, batch_size=50000):
    # model.eval() 
    # 先读取整个 JSON 文件到内存
    with open(data_loader.ann_file, 'r') as f:
        data = json.load(f)
    
    # 预处理，建立 image_id 到 annotations 的快速索引
    annotations_index = {}
    for ann in data['annotations']:
        if ann['image_id'] not in annotations_index:
            annotations_index[ann['image_id']] = []
        annotations_index[ann['image_id']].append(ann)

    updates_count = 0
    processed_images = 0

    model.train()  # Set model to train for calculating loss.
    results = []
    
    with torch.no_grad():  # Disable gradient computation
        print('Dataset size: ', len(data_loader))
        start_time = time.time()
        for i, (image, target) in enumerate(data_loader):
            image = image.to(device)
            target = {k: v.to(device) for k, v in target.items()}
            
            losses = model(image, target, get_score=get_score)
            if 'boundary_roi' in get_score:
                roi_losses = losses.get('roi_mask_boundary_loss', {})
            elif 'boundary_el2n' in get_score:
                roi_losses = losses.get('roi_mask_boundary_el2n_loss', {})
            elif 'el2n' in get_score:
                roi_losses = losses.get('roi_mask_el2n_loss', {})
            elif 'aum' in get_score:
                roi_losses = losses.get('roi_mask_aum', {})
            else:
                roi_losses = losses.get('roi_mask_loss', {})

            
            # Capture the results to update the JSON
            if roi_losses!=None:
                update_json_annotation_fast(target, roi_losses, annotations_index, get_score)
                # results.append(roi_losses)
            
            # processed_images += 1
            
            # # Batch save to JSON file
            # if processed_images % batch_size == 0 or processed_images == len(data_loader):
            #     with open(data_loader.ann_file, 'w') as f:
            #         json.dump(data, f)
            #     print(f"Batch {updates_count + 1} saved.")
            #     updates_count += 1

            if i % 100 == 0:
                print(i)

    return results

def update_forgetting_counts(image_id, per_gt_losses, historical_losses, forgetting_counts):
    for mask_index, loss_value in enumerate(per_gt_losses):
        # 创建一个基于image_id和mask_index的唯一标识符
        unique_id = f"{image_id}_{mask_index}"
        
        # 更新历史最低损失和遗忘计数
        historical_lowest = historical_losses.get(unique_id, float('inf'))
        if loss_value > historical_lowest:
            forgetting_counts[unique_id] = forgetting_counts.get(unique_id, 0) + 1
        historical_losses[unique_id] = min(historical_lowest, loss_value)

    return forgetting_counts

def get_roi_scores_fogetting(data_loader, model, device, optimizer, epoch, args):
    model.train()  # Set model to training mode to calculate and update forgetting scores.

    results = []
    print('Dataset size: ', len(data_loader))
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()

    historical_losses = {}
    forgetting_counts = {}

    for i, (image, target) in enumerate(data_loader):
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch
                   
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        S = time.time()
        
        losses = model(image, target, args.get_score)

        ### forgetting 
        roi_losses = losses.get('roi_mask_loss', {})
        forgetting_counts = update_forgetting_counts(target['image_id'].item(), roi_losses, historical_losses, forgetting_counts)

        if 'roi_mask_loss' in losses:
            del losses['roi_mask_loss']
        ###

        total_loss = sum(losses.values())
        total_loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

    return forgetting_counts

def update_json_annotation(target, roi_losses, ann_file, get_score):
    # Load existing annotations
    with open(ann_file, 'r') as f:
        data = json.load(f)
        annotations = data['annotations']

    image_id = target['image_id'].item()

    # 创建一个映射当前image_id的所有annotations的列表
    current_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
    # 对当前image_id的annotations进行索引重置
    for local_index, ann in enumerate(current_annotations):
        if local_index < len(roi_losses):
            if 'forgetting' in get_score:
                ann['forgetting_score'] = roi_losses[local_index].item()

            if 'boundary_roi' in get_score:
                ann['roi_boundary_score'] = roi_losses[local_index].item()
            elif 'boundary_el2n' in get_score:
                ann['roi_boundary_el2n_score'] = roi_losses[local_index].item()
            elif 'el2n' in get_score:
                ann['roi_el2n_score'] = roi_losses[local_index].item()
            elif 'aum' in get_score:
                ann['roi_aum_score'] = roi_losses[local_index].item()
            elif 'boundary_ratio' in get_score:
                ann['boundary_p2a_ratio'] = roi_losses[local_index]
            elif 'norm_ratio' in get_score:
                ann['norm_p2a_ratio'] = roi_losses[local_index]
            elif 'ratio' in get_score:
                ann['p2a_ratio'] = roi_losses[local_index]
            else:
                ann['roi_score'] = roi_losses[local_index].item()
        else:
            if 'forgetting' in get_score:
                ann['forgetting_score'] = 0

            if 'ratio' in get_score:
                ann['p2a_ratio'] = 0

            if 'boundary_roi' in get_score:
                ann['roi_boundary_score'] = 0  
            elif 'boundary_el2n' in get_score:
                ann['roi_boundary_el2n_score'] = 0
            elif 'el2n' in get_score:
                ann['roi_el2n_score'] = 0
            elif 'aum' in get_score:
                ann['roi_aum_score'] = 0
            elif 'boundary_ratio' in get_score:
                ann['boundary_p2a_ratio'] = 0
            elif 'norm_ratio' in get_score:
                ann['norm_p2a_ratio'] = 0
            else:
                ann['roi_score'] = 0 

    # Save updated annotations
    with open(ann_file, 'w') as f:
        json.dump(data, f)

def update_json_annotation_fast(target, roi_losses, annotations_index, get_score):
    image_id = target['image_id'].item()
    current_annotations = annotations_index[image_id]
    
    for local_index, ann in enumerate(current_annotations):
        if local_index < len(roi_losses):
            score_type_updates = {
                'forgetting': 'forgetting_score',
                'boundary_roi': 'roi_boundary_score',
                'boundary_el2n': 'roi_boundary_el2n_score',
                'el2n': 'roi_el2n_score',
                'aum': 'roi_aum_score',
                'ratio': 'p2a_ratio',
                'roi': 'roi_score'
            }
            for key, value in score_type_updates.items():
                if key in get_score:
                    ann[value] = roi_losses[local_index] if key == 'ratio' else roi_losses[local_index].item()
                    # print(ann[value])
                # else:
                #     ann[value] = 0

def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
                
			
            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
			
			
            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info

# def update_json_annotation(target, roi_losses, ann_file):
#     # Load existing annotations
#     with open(ann_file, 'r') as f:
#         annotations = json.load(f)

#     # Update annotations with roi_scores
#     for ann in annotations:
#         if ann['image_id'] == target['image_id'].item():
#             for label, roi_loss in roi_losses.items():
#                 # Update or append the roi_score for the specific label
#                 ann['roi_score'] = roi_loss.mean().item()  # Assuming mean is needed if multiple losses

#     # Save updated annotations
#     with open(ann_file, 'w') as f:
#         json.dump(annotations, f)

# def update_json_annotation(target, roi_losses, ann_file):
#     # Load existing annotations
#     with open(ann_file, 'r') as f:
#         annotations = json.load(f)

#     # Assuming target contains a unique identifier 'image_id' that matches with 'id' in annotations
#     image_id = target['image_id'].item()
#     for ann in annotations:
#         if ann['image_id'] == image_id:
#             # Compute average roi_loss for the object
#             object_id = ann['id']
#             if object_id in roi_losses:
#                 ann['roi_score'] = roi_losses[object_id].item()  # Or any other aggregation method

    # # Save updated annotations
    # with open(ann_file, 'w') as f:
    #     json.dump(annotations, f)