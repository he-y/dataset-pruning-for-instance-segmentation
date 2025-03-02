from torch import nn
import numpy as np
import torch.nn.functional as F
import torch

def get_lord_error_fn_2(cls_logits, targets, ord):
    num_classes = cls_logits.shape[-1]
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    softmax_logits = F.softmax(cls_logits, dim=-1)

    errors = []
    for target_one_hot in targets_one_hot:
        target_errors = softmax_logits - target_one_hot
        if ord == 2:
            target_scores = torch.norm(target_errors, p=2, dim=1)
        elif ord == 1:
            target_scores = torch.norm(target_errors, p=1, dim=1)
        else:
            raise NotImplementedError(f"Ord {ord} not implemented.")
        min_score = torch.min(target_scores).item()
        errors.append(min_score)

    return np.array(errors)

def get_lord_error_fn(cls_logits, targets, ord):

    targets_one_hot = F.one_hot(targets, num_classes=21).float()

    softmax_logits = F.softmax(cls_logits, dim=-1)
    matched_targets = []
    for logit in softmax_logits:
        max_index = torch.argmax(logit)
        matched_target = torch.zeros_like(targets_one_hot[0])
        matched_target[max_index] = 1
        matched_targets.append(matched_target)
    
    matched_targets = torch.stack(matched_targets)
    errors = softmax_logits - matched_targets

    scores = np.linalg.norm(errors.cpu().numpy(), ord=ord, axis=-1)
    return scores

def get_l2_error_fn(cls_logits, targets):
    return get_lord_error_fn_2(cls_logits, targets, 2)

def get_l1_error_fn(cls_logits, targets):
    return get_lord_error_fn(cls_logits, targets, 1)

def get_margin_error(cls_logits, targets, score_type):
    P = np.array(F.softmax(cls_logits, dim=-1))
    correct_logits = targets.astype(bool)
    margins = P[~correct_logits] - P[correct_logits]
    if score_type == 'max':
        scores = np.max(margins, axis=-1)
    elif score_type == 'sum':
        scores = np.sum(margins, axis=-1)
    return scores

def get_max_margin_error(cls_logits, targets):
    return get_margin_error(cls_logits, targets, 'max')

def get_sum_margin_error(cls_logits, targets):
    return get_margin_error(cls_logits, targets, 'sum')

def compute_bbox_scores(cls_logits, targets, score_type='l2_error'):
    if score_type == 'l2_error':
        scores = get_l2_error_fn(cls_logits, targets)
    elif score_type == 'l1_error':
        scores = get_l1_error_fn(cls_logits, targets)
    elif score_type == 'max_margin':
        scores = get_max_margin_error(cls_logits, targets)
    elif score_type == 'sum_margin':
        scores = get_sum_margin_error(cls_logits, targets)
    else:
        raise NotImplementedError(f"Score type {score_type} not implemented.")
    return scores
