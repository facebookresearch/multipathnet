#!/bin/bash

export dataset=coco
export test_set=val
export year=2014
export scale=800

export transformer=ImagenetTransformer
export test_model=./data/models/resnet18_integral_coco.t7
export proposals=sharpmask

export test_nsamples=5000
export test_best_proposals_number=400
export max_size=1000

th run_test.lua
