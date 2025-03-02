import numpy as np
from .pruning_score import compute_bbox_scores


def process_and_save_scores(image_batches, score_type, output_path):
    all_scores = []
    
    for batch_index, (cls_logits_batch, targets_batch) in enumerate(image_batches):
        # 计算每个批次的得分
        scores_batch = compute_bbox_scores(cls_logits_batch, targets_batch, score_type)
        all_scores.append(scores_batch)
        print(f"Processed batch {batch_index + 1}/{len(image_batches)}")
    
    # 将所有批次的得分拼接起来
    all_scores = np.concatenate(all_scores, axis=0)
    
    # 保存得分到文件
    np.save(output_path, all_scores)
    print(f"Scores saved to {output_path}")

# 示例调用
image_batches = [
    (cls_logits_batch1, targets_batch1),
    (cls_logits_batch2, targets_batch2),
    # 继续添加其他批次...
]

score_type = 'l2_error' 
output_path = 'bbox_scores.npy'

process_and_save_scores(image_batches, score_type, output_path)