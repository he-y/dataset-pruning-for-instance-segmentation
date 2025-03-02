from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc

from .utils import AnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork
from .pooler import RoIAlign
from .roi_heads import RoIHeads
from .transform import Transformer

# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn

    
def maskrcnn_resnet50(pretrained, num_classes):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.
    
    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """

    print('-----------------------')
    print('Using Official Mask RCNN')
    print('-----------------------')

        
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

    return model