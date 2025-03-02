# MaskRCNN VOC Dataset Pruning

## Train on full data
``` shell
python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 0
```

## Generate the importance score
Please replace the `METHOD` with the following method: `roi`, `boundary_roi`, `el2n`, `boundary_el2n`, `forgetting`, `aum` .
``` shell
python get_run_score.py --get_score METHOD
```

## Train by score
### Random
``` bash
# Pruning rate 50%
python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method random

# Pruning rate 60%
python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method random

# Pruning rate 70%
python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method random

# Pruning rate 80%
python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method random
```

### P2A ratio
Please replace the `NUM` with the following method: `732` , `878`, `1024`, `1170` (Pruning rate 50%, 40%, 30%, 20%).

Please replace the `METHOD` with the following method: `ratio`, `total_ratio`, `total_ratio_norm` .
``` bash
python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num NUM --rank_method METHOD
```

### Entropy
Please replace the `NUM` with the following method: `732` , `878`, `1024`, `1170` (Pruning rate 50%, 40%, 30%, 20%).

Please replace the `METHOD` with the following method: `roi_score`, `roi_total_score`, `roi_total_boundary_score`, `roi_total_boundary_score_norm` .
``` bash
python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num NUM --rank_method METHOD
```

### EL2N
Please replace the `NUM` with the following method: `732` , `878`, `1024`, `1170` (Pruning rate 50%, 40%, 30%, 20%).

Please replace the `METHOD` with the following method: `roi_total_el2n_score`, `roi_total_boundary_el2n_score`, `roi_total_el2n_score_norm`, `roi_total_boundary_el2n_score_norm` .
``` bash
python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num NUM --rank_method METHOD
```

### AUM
Please replace the `NUM` with the following method: `732` , `878`, `1024`, `1170` (Pruning rate 50%, 40%, 30%, 20%).

Please replace the `METHOD` with the following method: `aum_total_score`, `aum_total_score_norm`.
``` bash
python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num NUM --rank_method METHOD
```

### CCS
Please replace the `NUM` with the following method: `732` , `878`, `1024`, `1170` (Pruning rate 50%, 40%, 30%, 20%).

Please replace the `METHOD` with the following method: `ratio_scale`, `ratio_ccs`, `score_ccs`, `ratio_ccs_v2`.
``` bash
python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num NUM --rank_method METHOD
```