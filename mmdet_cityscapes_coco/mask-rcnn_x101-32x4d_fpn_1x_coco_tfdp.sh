#!/bin/bash

# Define the list of pruning rates
PRUNING_RATES=(50 40 30 20)

# Loop through each pruning rate and run the command
for PRUNING_RATE in "${PRUNING_RATES[@]}"
do
    echo "Running training with PRUNING RATE: $PRUNING_RATE"
    bash ./tools/dist_train.sh configs/mask_rcnn/mask-rcnn_x101-32x4d_fpn_1x_coco_tfdp_"${PRUNING_RATE}".py 2
done

echo "All training processes completed."
