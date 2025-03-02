import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from .pooler import RoIAlign
from .utils import Matcher, BalancedPositiveNegativeSampler, roi_align
from .box_ops import BoxCoder, box_iou, process_box, nms


def fastrcnn_loss(class_logit, box_regression, label, regression_target):
    classifier_loss = F.cross_entropy(class_logit, label)

    N, num_pos = class_logit.shape[0], regression_target.shape[0]
    box_regression = box_regression.reshape(N, -1, 4)
    box_regression, label = box_regression[:num_pos], label[:num_pos]
    box_idx = torch.arange(num_pos, device=label.device)

    box_reg_loss = F.smooth_l1_loss(box_regression[box_idx, label], regression_target, reduction='sum') / N

    return classifier_loss, box_reg_loss

def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)
            
    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target)
    return mask_loss

# boundary entropy
def maskrcnn_boundary_loss_per_gt(mask_logit, proposal, matched_idx, label, gt_mask, boundary_width=3):
    device = proposal.device
    matched_idx = matched_idx[:, None].to(device)
    roi = torch.cat((matched_idx, proposal), dim=1).float() 
            
    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(device).float()
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    boundary_mask = compute_boundary_mask(gt_mask.squeeze(1), boundary_width)
    boundary_mask = roi_align(boundary_mask[:, None, :, :], roi, 1., M, M, -1)[:, 0]

    boundary_mask = boundary_mask.to(device)
    mask_target = mask_target * boundary_mask

    mask_logit_selected = mask_logit[torch.arange(label.shape[0], device=device), label] * boundary_mask

    mask_loss = F.binary_cross_entropy_with_logits(mask_logit_selected, mask_target, reduction='none')
    # mask_loss = (mask_loss * boundary_mask).sum() / boundary_mask.sum()

    per_gt_losses = torch.zeros(gt_mask.shape[0], device=device)

    # 计算每个gt_mask对应所有proposal的损失平均值
    for i in range(gt_mask.shape[0]):
        mask_indices = (matched_idx.squeeze() == i)
        if mask_indices.any():
            current_boundary_mask = boundary_mask[mask_indices]
            per_gt_losses[i] = (mask_loss[mask_indices] * current_boundary_mask).sum() / current_boundary_mask.sum() if current_boundary_mask.sum() > 0 else torch.tensor(0.0, device=device)

    return per_gt_losses


def compute_boundary_mask(gt_mask, boundary_width):
    from skimage.morphology import binary_dilation, binary_erosion

    boundary_mask = torch.zeros_like(gt_mask)
    for i in range(gt_mask.shape[0]):
        mask = gt_mask[i].cpu().numpy()
        dilated = binary_dilation(mask, footprint=np.ones((boundary_width, boundary_width)))
        eroded = binary_erosion(mask, footprint=np.ones((boundary_width, boundary_width)))
        boundary = dilated & ~eroded
        boundary_mask[i] = torch.tensor(boundary, dtype=torch.float32)
    
    return boundary_mask

# entropy
def maskrcnn_loss_average_per_gt(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)
            
    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)
    losses = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target, reduction='none')

    per_gt_losses = torch.zeros(len(gt_mask), device=roi.device)

    for i in range(len(gt_mask)):
        mask_indices = (matched_idx.squeeze() == i)
        if mask_indices.any():
            per_gt_losses[i] = losses[mask_indices].mean()

    return per_gt_losses

# el2n
def maskrcnn_el2n_loss_average_per_gt(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)
            
    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)
    # l2 loss
    losses = F.mse_loss(mask_logit[idx, label], mask_target, reduction='none')

    per_gt_losses = torch.zeros(len(gt_mask), device=roi.device)

    for i in range(len(gt_mask)):
        mask_indices = (matched_idx.squeeze() == i)
        if mask_indices.any():
            per_gt_losses[i] = losses[mask_indices].mean()

    return per_gt_losses
    
# boundary el2n
def maskrcnn_boundary_el2n_loss_per_gt(mask_logit, proposal, matched_idx, label, gt_mask, boundary_width=3):
    device = proposal.device
    matched_idx = matched_idx[:, None].to(device)
    roi = torch.cat((matched_idx, proposal), dim=1).float() 
            
    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(device).float()
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    boundary_mask = compute_boundary_mask(gt_mask.squeeze(1), boundary_width)
    boundary_mask = roi_align(boundary_mask[:, None, :, :], roi, 1., M, M, -1)[:, 0]

    boundary_mask = boundary_mask.to(device)
    mask_target = mask_target * boundary_mask

    mask_logit_selected = mask_logit[torch.arange(label.shape[0], device=device), label] * boundary_mask

    mask_loss = F.mse_loss(mask_logit_selected, mask_target, reduction='none')

    per_gt_losses = torch.zeros(gt_mask.shape[0], device=device)

    for i in range(gt_mask.shape[0]):
        mask_indices = (matched_idx.squeeze() == i)
        if mask_indices.any():
            current_boundary_mask = boundary_mask[mask_indices]
            per_gt_losses[i] = (mask_loss[mask_indices] * current_boundary_mask).sum() / current_boundary_mask.sum() if current_boundary_mask.sum() > 0 else torch.tensor(0.0, device=device)

    return per_gt_losses


# forgetting (test)
def maskrcnn_forgetting_average_per_gt(mask_logit, proposal, matched_idx, label, gt_mask, last_correctness, forgetting_count):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)
            
    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)
    losses = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target, reduction='none')
    
    current_correctness = (mask_logit[idx, label].sigmoid() > 0.5) == mask_target.bool()

    per_gt_losses = torch.zeros(len(gt_mask), device=roi.device)
    per_gt_forgetting = torch.zeros(len(gt_mask), device=roi.device)

    for i in range(len(gt_mask)):
        mask_indices = (matched_idx.squeeze() == i)
        if mask_indices.any():
            per_gt_losses[i] = losses[mask_indices].mean()
            if last_correctness[i] == 1 and not current_correctness[mask_indices].any():
                forgetting_count[i] += 1
            last_correctness[i] = current_correctness[mask_indices].max().item()

    return forgetting_count

# aum
def maskrcnn_aum_per_gt(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)

    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)

    margins = torch.zeros(len(gt_mask), device=roi.device)

    else_count = 0

    for i in range(len(gt_mask)):
        mask_indices = (matched_idx.squeeze() == i)
        if mask_indices.any():
            selected_logit = mask_logit[mask_indices]
            max_logit = selected_logit.max(dim=1)[0]  


            if selected_logit.size(1) > 1:
                second_max_logit = selected_logit.topk(2, dim=1)[0][:, 1]
            else:
                second_max_logit = max_logit
                else_count += 1
            margins[i] = (max_logit - second_max_logit).mean()  
    if else_count > 0:
        print('Error Number: ', else_count)

    return margins



class RoIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 score_thresh, nms_thresh, num_detections):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor
        
        self.mask_roi_pool = None
        self.mask_predictor = None
        
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)
        
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_size = 1

        self.last_correctness = None  # 保存上一次的正确性
        self.forgetting_count = None  # 遗忘计数

    def reset_forgetting_tracking(self, num_samples):
        self.last_correctness = torch.zeros(num_samples, dtype=torch.float, device="cuda")
        self.forgetting_count = torch.zeros(num_samples, dtype=torch.float, device="cuda")
        
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True
        
    def select_training_samples(self, proposal, target):
        gt_box = target['boxes']
        gt_label = target['labels']
        proposal = torch.cat((proposal, gt_box))
        
        iou = box_iou(gt_box, proposal)
        pos_neg_label, matched_idx = self.proposal_matcher(iou)
        pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)
        idx = torch.cat((pos_idx, neg_idx))
        
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])
        proposal = proposal[idx]
        matched_idx = matched_idx[idx]
        label = gt_label[matched_idx]
        num_pos = pos_idx.shape[0]
        label[num_pos:] = 0
        
        return proposal, matched_idx, label, regression_target
    
    def fastrcnn_inference(self, class_logit, box_regression, proposal, image_shape):
        N, num_classes = class_logit.shape
        
        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=-1)
        box_regression = box_regression.reshape(N, -1, 4)
        
        boxes = []
        labels = []
        scores = []
        logits = []
        for l in range(1, num_classes):
            score, box_delta = pred_score[:, l], box_regression[:, l]

            keep = score >= self.score_thresh
            box, score, box_delta = proposal[keep], score[keep], box_delta[keep]
            box = self.box_coder.decode(box_delta, box)
            
            box, score = process_box(box, score, image_shape, self.min_size)
            
            keep = nms(box, score, self.nms_thresh)[:self.num_detections]
            box, score = box[keep], score[keep]
            label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)
            # score
            logit = class_logit[keep]
            
            boxes.append(box)
            labels.append(label)
            scores.append(score)
            # score
            logits.append(logit)

        results = dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores), logits=torch.cat(logits))
        return results
    
    def forward(self, feature, proposal, image_shape, target, get_score=None):
        if self.training:
            proposal, matched_idx, label, regression_target = self.select_training_samples(proposal, target)
        
        box_feature = self.box_roi_pool(feature, proposal, image_shape)
        class_logit, box_regression = self.box_predictor(box_feature)
        
        result, losses = {}, {}
        if self.training:
            classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)
            losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)


        else:
            result = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape)
            
        if self.has_mask():
            if self.training:
                num_pos = regression_target.shape[0]
                
                mask_proposal = proposal[:num_pos]
                pos_matched_idx = matched_idx[:num_pos]
                mask_label = label[:num_pos]
                
                '''
                # -------------- critial ----------------
                box_regression = box_regression[:num_pos].reshape(num_pos, -1, 4)
                idx = torch.arange(num_pos, device=mask_label.device)
                mask_proposal = self.box_coder.decode(box_regression[idx, mask_label], mask_proposal)
                # ---------------------------------------
                '''
                
                if mask_proposal.shape[0] == 0:
                    losses.update(dict(roi_mask_loss=torch.tensor(0)))
                    return result, losses
            else:
                mask_proposal = result['boxes']
                
                if mask_proposal.shape[0] == 0:
                    result.update(dict(masks=torch.empty((0, 28, 28))))
                    return result, losses
                
            mask_feature = self.mask_roi_pool(feature, mask_proposal, image_shape)
            mask_logit = self.mask_predictor(mask_feature)

            
            if self.training:
                gt_mask = target['masks']
                mask_loss = maskrcnn_loss(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                # mask_loss = maskrcnn_loss_average_per_gt(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                losses.update(dict(roi_mask_loss=mask_loss))

            else:
                label = result['labels']
                idx = torch.arange(label.shape[0], device=label.device)
                mask_logit = mask_logit[idx, label]

                mask_prob = mask_logit.sigmoid()
                result.update(dict(masks=mask_prob))

            if get_score != None and 'forgetting' not in get_score:
                if 'boundary_roi' in get_score:
                    # print('Score name: ', get_score)
                    mask_loss = maskrcnn_boundary_loss_per_gt(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                    losses.update(dict(roi_mask_boundary_loss=mask_loss))
                elif 'boundary_el2n' in get_score:
                    # print('Score name: ', get_score)
                    mask_loss = maskrcnn_boundary_el2n_loss_per_gt(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                    losses.update(dict(roi_mask_boundary_el2n_loss=mask_loss))
                elif 'el2n' in get_score:
                    # print('Score name: ', get_score)
                    mask_loss = maskrcnn_el2n_loss_average_per_gt(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                    losses.update(dict(roi_mask_el2n_loss=mask_loss))
                elif 'aum' in get_score:
                    # print('Score name: ', get_score)
                    mask_loss = maskrcnn_aum_per_gt(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                    losses.update(dict(roi_mask_aum=mask_loss))
                else:
                    mask_loss = maskrcnn_loss_average_per_gt(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                    losses.update(dict(roi_mask_loss=mask_loss))

            ### forgetting ###
            if get_score is not None and 'forgetting' in get_score:
                    mask_loss = maskrcnn_loss_average_per_gt(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                    losses.update(dict(roi_mask_loss=mask_loss))

                
        # return result, losses, class_logit
        return result, losses