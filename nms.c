/*------------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------*/

#include <TH/TH.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

float overlap(const float *a, const float *b)
{
  float a_x1 = a[0];
  float a_y1 = a[1];
  float a_x2 = a[2];
  float a_y2 = a[3];

  float b_x1 = b[0];
  float b_y1 = b[1];
  float b_x2 = b[2];
  float b_y2 = b[3];

  float x1 = MAX(a_x1, b_x1);
  float y1 = MAX(a_y1, b_y1);
  float x2 = MIN(a_x2, b_x2);
  float y2 = MIN(a_y2, b_y2);

  float w = x2 - x1 + 1;
  float h = y2 - y1 + 1;

  float intersection = w * h;

  float aarea = (a_x2 - a_x1 + 1) * (a_y2 - a_y1 + 1);
  float barea = (b_x2 - b_x1 + 1) * (b_y2 - b_y1 + 1);

  float iou = intersection / (aarea + barea - intersection);
  return (w <= 0 || h <= 0) ? 0 : iou;
}

void boxoverlap(THFloatTensor *result, THFloatTensor *a, THFloatTensor *b)
{
  int N = a->size[0];
  THFloatTensor_resize1d(result, N);
  float *a_data = THFloatTensor_data(a);
  float *b_data = THFloatTensor_data(b);
  float *r_data = THFloatTensor_data(result);

  for(int i=0;i<N;++i)
  {
    r_data[i] = overlap(a_data, b_data);
    a_data += 4;
  }
}


void NMS(THFloatTensor *keep, THFloatTensor *scored_boxes, float threshold)
{
  int N = scored_boxes->size[0];
  float **boxes = calloc(N, sizeof(float*));
  float *scored_boxes_data = THFloatTensor_data(scored_boxes);
  for(int i=0; i<N; ++i)
    boxes[i] = scored_boxes_data + i*5;

  float **boxes_original = boxes;

  int num = N;
  int numNMS = 0;

  while(num) {
    // Greedily select the highest scoring bounding box
    int best = -1;
    float bestS = -10000000;
    for(int i = 0; i < num; i++) {
      if(boxes[i][4] > bestS) {
        bestS = boxes[i][4];
        best = i;
      }
    }
    float *b = boxes[best];
    float *tmp = boxes[0];
    boxes[0] = boxes[best];
    boxes[best] = tmp;
    boxes++;
    numNMS++;

    // Remove all bounding boxes where the percent area of overlap is greater than overlap
    int numGood = 0;
    for(int i = 0; i < num-1; i++) {
      float inter_over_union = overlap(b, boxes[i]);
      if(inter_over_union <= threshold) {
        tmp = boxes[numGood];
        boxes[numGood++] = boxes[i];
        boxes[i] = tmp;
      }
    }
    num = numGood;
  }

  THFloatTensor_resize2d(keep, numNMS, 5);
  float *keep_data = THFloatTensor_data(keep);
  for(int i=0; i<numNMS; ++i)
    memcpy(keep_data + i*5, boxes_original[i], sizeof(float)*5);

  free(boxes_original);
}

void bbox_vote(THFloatTensor *res, THFloatTensor *nms_boxes, THFloatTensor *scored_boxes, float threshold)
{
  THAssert(THFloatTensor_isContiguous(nms_boxes));
  THAssert(THFloatTensor_isContiguous(scored_boxes));
  THFloatTensor_resizeAs(res, nms_boxes);
  THFloatTensor_zero(res);

  int N_nms   = nms_boxes->size[0];
  int N_boxes = scored_boxes->size[0];
  THAssert(nms_boxes->size[1] == 5);
  THAssert(scored_boxes->size[1] == 5);
  // THFloatTensor* overlaps = THFloatTensor_newWithSize1d(N_boxes);

  float *nms_data = THFloatTensor_data(nms_boxes);
  float *scored_data = THFloatTensor_data(scored_boxes);
  float *res_data = THFloatTensor_data(res);

  for(int i=0; i<N_nms; i++) {
    for(int j=0; j<N_boxes; j++) {
      float ov = overlap(scored_data + 5*j, nms_data + 5*i);
      if(ov > threshold) {
        for(int field = 0; field<4; field++) {
          res_data[5*i+field] += scored_data[5*j+field] * scored_data[5*j+4];
        }
        res_data[5*i+4] += scored_data[5*j+4];
      }
    }
    for(int field = 0; field<4; field++) {
      res_data[5*i+field] /= res_data[5*i+4];
    }
    res_data[5*i+4] = nms_data[5*i+4];
  }
}
