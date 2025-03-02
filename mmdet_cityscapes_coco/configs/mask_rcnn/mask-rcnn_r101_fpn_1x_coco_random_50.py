_base_ = './mask-rcnn_r50_fpn_1x_coco_random_sample_58660.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
