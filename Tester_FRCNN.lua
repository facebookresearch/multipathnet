--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local utils = paths.dofile('utils.lua')
local tds = require 'tds'
local testCoco = require 'testCoco.init'
require 'sys'

local Tester = torch.class('fbcoco.Tester_FRCNN')

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

function Tester:__init(module, transformer, dataset, scale, max_size, opt)
   self.dataset = dataset
   self.module = module
   self.transformer = transformer
   if module and transformer then
      self.detec = fbcoco.ImageDetect(self.module, self.transformer, scale, max_size)
   end
   self.num_iter = opt.test_num_iterative_loc or 1

   self.nms_thresh = opt.test_nms_threshold or 0.3
   self.bbox_vote_thresh = opt.test_bbox_voting_nms_threshold or 0.5

   self.threads = Threads(10,
   function()
      require 'torch'
   end)

   if module then
      module:apply(function(m)
         if torch.type(m) ==  'nn.DataParallelTable' then
            self.data_parallel_n = #m.gpuAssignments
         end
      end)
      print('data_parallel_n', self.data_parallel_n)

      -- to determine num of output classes
      local input = {torch.CudaTensor(self.data_parallel_n or 2, 3, 224, 224),
      torch.Tensor{1, 1, 1, 100, 100}:view(1, 5):expand(2, 5):cuda()}
      module:forward(input)

      self.num_classes = module.output[1]:size(2) - 1
      self.thresh = torch.ones(self.num_classes):mul(-1.5)
   end
end

function Tester:testOne(i)
   local dataset = self.dataset
   local thresh = self.thresh

   local img_boxes = tds.hash()
   local timer = torch.Timer()
   local timer2 = torch.Timer()
   local timer3 = torch.Timer()

   timer:reset()
   local boxes = dataset:getROIBoxes(i):float()
   -- print('#boxes', boxes:size())
   local im = dataset:getImage(i)
   timer3:reset()

   local all_output = {}
   local all_bbox_pred = {}

   local output, bbox_pred = self.detec:detect(im, boxes, self.data_parallel_n, true)
   local num_classes = output:size(2) - 1

   -- clamp predictions within image
   local bbox_pred_tmp = bbox_pred:view(-1, 2)
   bbox_pred_tmp:select(2,1):clamp(1, im:size(3))
   bbox_pred_tmp:select(2,2):clamp(1, im:size(2))

   table.insert(all_output, output)
   table.insert(all_bbox_pred, bbox_pred)
   for i = 2, self.num_iter do
      -- have to copy to cuda because of torch/cutorch LongTensor differences
      self.boxselect = self.boxselect or nn.SelectBoxes():cuda()
      local new_boxes = self.boxselect:forward{output:cuda(), bbox_pred:cuda()}:float()
      output, bbox_pred = self.detec:detect(im, new_boxes, self.data_parallel_n, false)
      table.insert(all_output, output)
      table.insert(all_bbox_pred, bbox_pred)
   end

   if opt.test_use_rbox_scores then
      assert(#all_output > 1)
      -- we use the scores from iter n+1 for the boxes at iter n
      -- this means we lose one iteration worth of boxes
      table.remove(all_output, 1)
      table.remove(all_bbox_pred)
   end

   output = utils.joinTable(all_output, 1)
   bbox_pred = utils.joinTable(all_bbox_pred, 1)

   local tt2 = timer3:time().real

   timer2:reset()
   local nms_timer = torch.Timer()
   for j = 1, num_classes do
      local scores = output:select(2, j+1)
      local idx = torch.range(1, scores:numel()):long()
      local idx2 = scores:gt(thresh[j])
      idx = idx[idx2]
      local scored_boxes = torch.FloatTensor(idx:numel(), 5)
      if scored_boxes:numel() > 0 then
         local bx = scored_boxes:narrow(2, 1, 4)
         bx:copy(bbox_pred:narrow(2, j*4+1, 4):index(1, idx))
         scored_boxes:select(2, 5):copy(scores[idx2])
      end
      img_boxes[j] = utils.nms(scored_boxes, self.nms_thresh)
      if opt.test_bbox_voting then
         local rescaled_scored_boxes = scored_boxes:clone()
         local scores = rescaled_scored_boxes:select(2,5)
         scores:pow(opt.test_bbox_voting_score_pow or 1)

         img_boxes[j] = utils.bbox_vote(img_boxes[j], rescaled_scored_boxes, self.test_bbox_voting_nms_threshold)
      end
   end
   self.threads:synchronize()
   local nms_time = nms_timer:time().real

   if i % 1 ==  0 then
      print(('test: (%s) %5d/%-5d dev: %d, forward time: %.3f, '
      .. 'select time: %.3fs, nms time: %.3fs, '
      .. 'total time: %.3fs'):format(dataset.dataset_name,
      i, dataset:size(),
      cutorch.getDevice(),
      tt2, timer2:time().real,
      nms_time, timer:time().real));
   end
   return img_boxes, {output, bbox_pred}
end

function Tester:test()
   self.module:evaluate()
   self.dataset:loadROIDB()

   local aboxes_t = tds.hash()

   local raw_output = tds.hash()
   local raw_bbox_pred = tds.hash()

   for i = 1, self.dataset:size() do
      local img_boxes, raw_boxes = self:testOne(i)
      aboxes_t[i] = img_boxes
      if opt.test_save_raw and opt.test_save_raw ~= '' then
         raw_output[i] = raw_boxes[1]:float()
         raw_bbox_pred[i] = raw_boxes[2]:float()
      end
   end

   if opt.test_save_raw and opt.test_save_raw ~= '' then
      torch.save(opt.test_save_raw, {raw_output, raw_bbox_pred})
   end

   aboxes_t = self:keepTopKPerImage(aboxes_t, 100) -- coco only accepts 100/image
   local aboxes = self:transposeBoxes(aboxes_t)
   aboxes_t = nil

   return self:computeAP(aboxes)
end

function Tester:keepTopKPerImage(aboxes_t, k)
   for j = 1,self.dataset:size() do
      aboxes_t[j] = utils.keep_top_k(aboxes_t[j], k)
   end
   return aboxes_t
end

function Tester:transposeBoxes(aboxes_t)
   -- print("Running topk. max= ", self.max_per_set)
   local aboxes = tds.hash()
   for j = 1, self.num_classes do
      aboxes[j] = tds.hash()
      for i = 1, self.dataset:size() do
         aboxes[j][i] = aboxes_t[i][j]
      end
   end
   return aboxes
end

function Tester:computeAP(aboxes)
   return testCoco.evaluate(self.dataset.dataset_name, aboxes)
end

