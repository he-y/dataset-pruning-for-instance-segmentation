``` bash
CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 0

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 1000

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 500

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 200

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 100

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 75

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 50

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir /data/zhaoqihao/test_dd/dataset_pruning_segmentation/mmdet_cityscapes_coco/data/cityscapes --voc_select_num 0
```

``` bash
CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 0

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 1000

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 500

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 200

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 100 --rank_method ratio_scale

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 75

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 50
```


``` bash
CUDA_VISIBLE_DEVICES=0 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 293 --rank_method score_ccs

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset voc --data-dir VOCdevkit/VOC2012 --voc_select_num 200 --rank_method ratio_scale

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1300 --rank_method score_ccs
```


``` bash
CUDA_VISIBLE_DEVICES=0,1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir coco2017

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir coco2017 
```



``` bash
CUDA_VISIBLE_DEVICES=0 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method roi_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method roi_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method roi_score

CUDA_VISIBLE_DEVICES=0 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method roi_score
```

``` bash
CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method roi_total_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method roi_total_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method roi_total_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method roi_total_score
```

``` bash
CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method roi_total_boundary_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method roi_total_boundary_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method roi_total_boundary_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method roi_total_boundary_score
```

``` bash
CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method roi_total_boundary_score_norm

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method roi_total_boundary_score_norm

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method roi_total_boundary_score_norm

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method roi_total_boundary_score_norm
```

``` bash
CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method roi_total_el2n_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method roi_total_el2n_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method roi_total_el2n_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method roi_total_el2n_score
```

``` bash
CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method roi_total_boundary_el2n_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method roi_total_boundary_el2n_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method roi_total_boundary_el2n_score

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method roi_total_boundary_el2n_score
```

``` bash
CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method roi_total_el2n_score_norm

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method roi_total_el2n_score_norm

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method roi_total_el2n_score_norm

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method roi_total_el2n_score_norm
```

``` bash
CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method total_ratio_norm

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method total_ratio_norm

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method total_ratio_norm

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method total_ratio_norm
```

``` bash
CUDA_VISIBLE_DEVICES=0 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method aum_total_score --dir_path ./results/result_732_scratch_total_org_aum_voc_0819

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method aum_total_score --dir_path ./results/result_878_scratch_total_org_aum_voc_0819

CUDA_VISIBLE_DEVICES=2 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method aum_total_score --dir_path ./results/result_1024_scratch_total_org_aum_voc_0819

CUDA_VISIBLE_DEVICES=3 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method aum_total_score --dir_path ./results/result_1170_scratch_total_org_aum_voc_0819
```


``` bash
# CCS

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method stratified_el2n --dir_path ./results/result_732_scratch_stratified_total_el2n_voc_0819

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method stratified_el2n --dir_path ./results/result_878_scratch_stratified_total_el2n_voc_0819

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method stratified_el2n --dir_path ./results/result_1024_scratch_stratified_total_el2n_voc_0819

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method stratified_el2n --dir_path ./results/result_1170_scratch_stratified_total_el2n_voc_0819
```

``` bash
# CCS

CUDA_VISIBLE_DEVICES=0 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method stratified_reverse_aum --dir_path ./results/result_732_scratch_stratified_total_reverse_aum_voc_0819

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method stratified_reverse_aum --dir_path ./results/result_878_scratch_stratified_total_reverse_aum_voc_0819

CUDA_VISIBLE_DEVICES=2 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method stratified_reverse_aum --dir_path ./results/result_1024_scratch_stratified_total_reverse_aum_voc_0819

CUDA_VISIBLE_DEVICES=3 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method stratified_reverse_aum --dir_path ./results/result_1170_scratch_stratified_total_reverse_aum_voc_0819
```

``` bash
# CCS

CUDA_VISIBLE_DEVICES=0 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method total_norm_ratio --dir_path ./results/result_732_scratch_total_norm_ratio_voc_0819

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method total_norm_ratio --dir_path ./results/result_878_scratch_total_norm_ratio_voc_0819

CUDA_VISIBLE_DEVICES=2 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method total_norm_ratio --dir_path ./results/result_1024_scratch_total_norm_ratio_voc_0819

CUDA_VISIBLE_DEVICES=3 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method total_norm_ratio --dir_path ./results/result_1170_scratch_total_norm_ratio_voc_0819


CUDA_VISIBLE_DEVICES=0 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method total_norm_ratio_norm_count --dir_path ./results/result_732_scratch_total_norm_ratio_norm_count_voc_0828

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 878 --rank_method total_norm_ratio_norm_count --dir_path ./results/result_878_scratch_total_norm_ratio_norm_count_voc_0828

CUDA_VISIBLE_DEVICES=2 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1024 --rank_method total_norm_ratio_norm_count --dir_path ./results/result_1024_scratch_total_norm_ratio_norm_count_voc_0828

CUDA_VISIBLE_DEVICES=3 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 1170 --rank_method total_norm_ratio_norm_count --dir_path ./results/result_1170_scratch_total_norm_ratio_norm_count_voc_0828

CUDA_VISIBLE_DEVICES=0 python train.py --use-cuda --iters 2000 --dataset coco --data-dir VOCdevkit/VOC2012 --voc_select_num 732 --rank_method stratified_aum --dir_path ./results/results_test

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir /mnt/data1/data_yalun/seg_dp/dataset_pruning_segmentation/mmdet_cityscapes_coco/data/coco --voc_select_num 732 --rank_method stratified_aum --dir_path ./results/results_test

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir /mnt/data1/data_yalun/seg_dp/dataset_pruning_segmentation/mmdet_cityscapes_coco/data/coco --voc_select_num 732 --rank_method stratified_aum --dir_path ./results/results_test

CUDA_VISIBLE_DEVICES=1 python train.py --use-cuda --iters 2000 --dataset coco --data-dir /mnt/data1/data_yalun/seg_dp/dataset_pruning_segmentation/mmdet_cityscapes_coco/data/cityscapes --voc_select_num 732 --rank_method stratified_aum --dir_path ./results/results_test

```
