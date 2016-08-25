#!/bin/bash

export proposals=selective_search
export test_best_proposals_number=2000
export best_proposals_number=2000

export save_folder=logs/fastrcnn_voc2007_${RANDOM}${RANDOM}
mkdir -p $save_folder

th train.lua | tee $save_folder/log.txt

# model=vgg
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.358
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.688
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.331
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.102
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.211
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.415
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.359
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.470
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.474
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.352
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.525

# model=alexnet
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.260
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.558
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.212
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.050
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.309
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.301
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.396
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.402
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.169
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.278
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.449
