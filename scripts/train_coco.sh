#!/bin/bash

export year=2014
export train_set=trainval
export test_set=val
export dataset=coco

export nDonkeys=6
export integral=true
export images_per_batch=4
export batchSize=64
export scale=800
export weightDecay=0
export test_best_proposals_number=400
export test_nsamples=1000

export proposals=sharpmask
export nEpochs=3200
export step=2800
export save_folder="logs/coco_${model}_${proposals}_$RANDOM$RANDOM"

mkdir -p $save_folder

th train.lua | tee $save_folder/log.txt

