#!/bin/bash

export proposals=selective_search
export test_best_proposals_number=2000

th run_test.lua

# model=data/models/caffenet_fast_rcnn_iter_40000.t7
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.668
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.320
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.062
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.202
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.406
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.350
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.462
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.468
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.260
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.344
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.518


# model=../data/models/vgg16_fast_rcnn_iter_40000.t7
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.264
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.559
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.217
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.034
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.132
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.318
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.304
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.400
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.285
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.456
