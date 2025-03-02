``` bash
# cityscape
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes.py 2
CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes_random.py 2

CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes_ratio_ccs.py 2

CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes_random_1483.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes_random_1783.py 2

CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes_random_2073.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes_random_2372.py 2

# cityscape el2n
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes_total_entropy_1483.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes_total_entropy_1783.py 2

CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes_total_entropy_2073.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes_total_entropy_2372.py 2

# total ratio

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes_ratio_ccs.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/cityscapes/mask-rcnn_r50_fpn_1x_cityscapes_ratio_ccs.py 2 --resume work_dirs/mask-rcnn_r50_fpn_1x_cityscapes_total_ratio_1783/epoch_2.pth  
```

``` bash
# coco
CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py 2 --resume work_dirs/mask-rcnn_r50_fpn_1x_coco/epoch_7.pth  

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_random_sample.py 2

CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_ccs_ratio_50.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_random_sample_20.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_ccs_ratio_20.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_ccs_ratio_10.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_ccs_ratio_10.py 2

CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_ccs_ratio_30.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_ccs_ratio_40.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_ccs_ratio_50.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_ccs_ratio_80.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_ccs_ratio_70.py 2

# coco total score
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/solov2/solov2_r50_fpn_1x_coco_store_ccs_58660.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/solov2/solov2_r50_fpn_1x_coco_store_ccs_70339.py 2

CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco_random_58660.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/solov2/solov2_r50_fpn_1x_coco_random_93805.py 2

# coco total score
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_store_ccs_58660.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_store_ccs_70339.py 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_store_ccs_82103.py 2

CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco_store_ccs_93888.py 2

# queryinst
CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/queryinst/queryinst_r50_fpn_1x_coco_store_entropy_58660 2

CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/queryinst/queryinst_r50_fpn_1x_coco_store_entropy_58660 2

CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh configs/solov2/solov2_r50_fpn_1x_cityscapes_norm_ratio_norm_1483 2

CUDA_VISIBLE_DEVICES=2,3 bash ./tools/dist_train.sh configs/solov2/solov2_r50_fpn_1x_cityscapes_norm_ratio_norm_1483 2
```


``` bash
# test
python tools/test.py \
    work_dirs/mask-rcnn_r50_fpn_1x_cityscapes_total_ratio_1483/mask-rcnn_r50_fpn_1x_cityscapes_total_ratio_1483.py \
    work_dirs/mask-rcnn_r50_fpn_1x_cityscapes_total_ratio_1483/epoch_8.pth

```