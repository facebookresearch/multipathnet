--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local DataLoader = require 'loaders.dataloader'
local ConcatLoader = require 'loaders.concatloader'
local NarrowLoader = require 'loaders.narrowloader'

local utils = paths.dofile'utils.lua'
local stringx = require('pl.stringx')

local DataSetCOCO = {}

function DataSetCOCO:create(name, roidbfile, nsamples, offset)
   local dataset
   if name == 'coco_trainval2014' then
      local train = DataLoader('coco_train2014')
      local val = DataLoader('coco_val2014')

      dataset = ConcatLoader{train, loader.NarrowLoader(val, 5001, val:nImages() - 5000)}
   elseif name == 'coco_val5k2014' then
      local val = DataLoader('coco_val2014')
      dataset = NarrowLoader(val, 1, 5000)
   elseif name == 'coco_val35k2014' then
      local val = DataLoader('coco_val2014')
      dataset = NarrowLoader(val, 5001, val:nImages() - 5000)
   elseif name == 'pascal_trainval2007,2012' then
      dataset = ConcatLoader{
         DataLoader('pascal_train2007'),
         DataLoader('pascal_val2007'),
         DataLoader('pascal_train2012'),
         DataLoader('pascal_val2012'),
      }
   elseif name == 'pascal_trainval2007' then
      dataset = ConcatLoader{
         DataLoader('pascal_train2007'),
         DataLoader('pascal_val2007'),
      }
   else
      dataset = DataLoader(name)
   end

   if offset and offset ~= -1 then
      local size = dataset:nImages()
      nsamples = math.min(size - offset + 1, nsamples)
      dataset = NarrowLoader(dataset, offset, nsamples)
   end

   self.dataset_name = name
   dataset.do_normalize = false
   self.dataset = dataset
   self.classes = {}
   if dataset.categories then -- coco_test2014 does not have categories
      for i,v in ipairs(dataset.categories) do self.classes[i] = v.name end
   end
   self.roidbfile = roidbfile
   self.min_area = 0
   self.min_proposal_area = 0
   self.nsamples = nsamples

   self.sample_n_per_box = 0
   self.sample_sigma = 1
   self.allow_missing_proposals=true
   return self
end

function DataSetCOCO:allowMissingProposals(allow_missing_proposals)
   self.allow_missing_proposals = allow_missing_proposals
   return self
end

function DataSetCOCO:size()
   if self.nsamples and self.nsamples >=0 then
      return self.nsamples
   end
   return self.dataset:nImages()
end

function DataSetCOCO:getImage(i)
   return self.dataset:loadImage(i)
end

function DataSetCOCO:getNumClasses()
   return self.dataset:nCategories()
end

function DataSetCOCO:setMinArea(area)
   assert(torch.type(area) == 'number')
   self.min_area = area
end

function DataSetCOCO:setMinProposalArea(area)
   assert(torch.type(area) == 'number')
   self.min_proposal_area = area
end

function DataSetCOCO:getAnnotation(i)
   local object = {}
   for j,a in ipairs(self.dataset:getAnnotationsForImage(i)) do
      if a.area > self.min_area then
         assert(a.difficult)
         local bbox = a.bbox:clone():float()
         bbox:narrow(1,3,2):add(bbox:narrow(1,1,2)):add(1)
         table.insert(object, {bbox = bbox, class_id = a.category, difficult = a.difficult, iscrowd = a.iscrowd})
      end
   end
   return object
end

local function TableConcat(t1,t2)
   if not t1 or t1:nElement() == 0 then
      return t2:float()
   end
   if not t2 or t2:nElement() == 0 then
      return t1:float()
   end
   return torch.cat(t1:float(), t2:float(), 1)
end

function DataSetCOCO:loadAndMergeProposals(roidbfile)
   local dt
   if type(roidbfile) == 'table' then
      dt = {boxes={}, scores={}, images={}}
      local img2idx = {}
      for i = 1, #roidbfile do
         assert(roidbfile[i] and paths.filep(roidbfile[i]),'proposals file ('..roidbfile[i]..') not found')
         local dt2 = torch.load(roidbfile[i])
         for k,v in pairs(dt2.images) do
            if not img2idx[v] then
               table.insert(dt.images, v)
               img2idx[v] = #dt.images
            end
            local idx = img2idx[v]
            dt.boxes[idx] = TableConcat(dt.boxes[idx], dt2.boxes[k])
            if dt2.scores then
               dt.scores[idx] = TableConcat(dt.scores[idx], dt2.scores[k])
            else
               -- lets just score unscored proposals as 0
               dt.scores[idx] = TableConcat(dt.scores[idx],
               torch.FloatTensor(dt2.boxes[k]:size(1)):zero())
            end
         end
      end
   elseif type(roidbfile) == 'string' then
      assert(roidbfile and paths.filep(roidbfile),'proposals file ('..roidbfile..') not found')
      dt = torch.load(roidbfile)
   else
      error("???")
   end
   return dt
end

local permute_tensor = torch.LongTensor{2,1,4,3}

local function filterScore(boxes, scores, best_number)
   if not scores then
      return boxes
   end
   if boxes:size(1) > best_number then -- select boxes with best scores
      local _,idx = scores:sort(true)
      idx = idx:narrow(1,1,best_number)
      -- print('scores', scores:size())
      -- print('idx', idx:size())
      boxes = boxes:index(1,idx)
      scores = scores:index(1,idx)
   end
   return boxes, scores
end

local function filterArea(boxes, scores, area)
   if area == 0 then
      return boxes, scores
   else
      assert(boxes:nDimension() == 2)
      local wh = boxes:narrow(2,3,2):clone():add(-1, boxes:narrow(2,1,2))
      local s = wh:select(2,1):cmul(wh:select(2,2))
      local idx = s:gt(area):nonzero()
      idx = idx:view(idx:nElement())
      local new_boxes = boxes:index(1, idx)
      local new_scores = scores and scores:index(1, idx)
      -- print("filterArea: reduced proposals from " .. boxes:size(1) .. " to " .. new_boxes:size(1))
      return new_boxes, new_scores
   end
end

function DataSetCOCO:loadROIDB(best_number)
   if self.roidb then
      return
   end
   local roidbfile = self.roidbfile

   print("Loading proposals at ", roidbfile)
   local dt = self:loadAndMergeProposals(roidbfile)
   print("Done loading proposals")

   assert(#dt.boxes == #dt.images)
   print('# proposal images', #dt.boxes)
   print('# dataset images', self.dataset:nImages())
   -- assert(#dt.boxes >= self.dataset:nImages(), 'proposals have fewer boxes than dataset ' .. #dt.boxes .. ' ' .. self.dataset:nImages())
   if dt.scores then
      assert(#dt.boxes == #dt.scores)
      assert(best_number and torch.type(best_number) == 'number','best_number has to be a valid number, e.g. 500 or 5000')
   end

   self.roidb   = {}
   self.scoredb = {}

   print('# images', #dt.images)
   print('nImages', self.dataset:nImages())
   local im2box = {}
   for i = 1,#dt.images do
      im2box[dt.images[i] ] = i
   end

   for i=1,self.dataset:nImages() do
      local file_name = self.dataset:getImage(i).file_name
      if not self.allow_missing_proposals then
         assert(im2box[file_name], file_name .. " is not in proposals")
      elseif not im2box[file_name] then
         print("WARNING: " .. i .. " " .. file_name .. " is not in proposals")
      end
      if im2box[file_name] then --assert(im2box[file_name], file_name .. " is not in proposals")
         local boxes = dt.boxes[im2box[file_name] ]:float()

         local scores = dt.scores and dt.scores[im2box[file_name] ]:float()
         scores = scores and scores:reshape(scores:nElement())
         boxes, scores = filterArea(boxes, scores, self.min_proposal_area)
         boxes, scores = filterScore(boxes, scores, best_number)

         boxes = boxes:size(2) ~= 4 and torch.FloatTensor(0,4) or boxes:index(2,permute_tensor)
         self.roidb[i] = boxes
         self.scoredb[i] = scores
      end
   end
end

function DataSetCOCO:getROIBoxes(i)
   if not self.roidb then self:loadROIDB() end
   assert(self.roidb[i], "No proposals for image " .. self.dataset:getImage(i).file_name)
   return self.roidb[i]
end

function DataSetCOCO:getROIScores(i)
   if not self.roidb then self:loadROIDB() end
   return self.scoredb[i]
end


function DataSetCOCO:getGTBoxes(i)
   local anno = self:getAnnotation(i)
   local valid_objects = {}
   local gt_boxes = torch.FloatTensor()
   local gt_classes = {}

   for idx,obj in ipairs(anno) do
      if not obj.difficult or obj.difficult == 0 and not obj.iscrowd then
         table.insert(valid_objects,idx)
      end
   end

   gt_boxes:resize(#valid_objects,4)
   for idx0,idx in ipairs(valid_objects) do
      gt_boxes[idx0]:copy(anno[idx].bbox)
      table.insert(gt_classes, anno[idx].class_id)
   end
   return gt_boxes,gt_classes,valid_objects,anno
end


local function sampleAroundGTBoxes(boxes, n_per_box, sigma)
   local samples = torch.repeatTensor(boxes, n_per_box, 1)
   return samples:add(torch.FloatTensor(#samples):normal(0,sigma))
end


function DataSetCOCO:attachProposals(i)
   if not self.roidb then self:loadROIDB() end

   local boxes = self:getROIBoxes(i)
   -- handle
   local gt_boxes,gt_classes,valid_objects,anno = self:getGTBoxes(i)
   if self.sample_n_per_box > 0 and gt_boxes:numel() > 0 then
      local sampled = sampleAroundGTBoxes(gt_boxes, self.sample_n_per_box, self.sample_sigma)
      boxes = boxes:cat(sampled, 1)
   end

   local all_boxes
   if anno then
      if #valid_objects > 0 and boxes:dim() > 0 then
         all_boxes = torch.cat(gt_boxes,boxes,1)
      elseif boxes:dim() == 0 then
         all_boxes = gt_boxes
      else
         all_boxes = boxes
      end
   else
      gt_boxes = torch.FloatTensor(0,4)
      all_boxes = boxes
   end

   local num_boxes = boxes:dim() > 0 and boxes:size(1) or 0
   local num_gt_boxes = #gt_classes

   local rec = {}
   if num_gt_boxes > 0 and num_boxes > 0 then
      rec.gt = torch.cat(torch.ByteTensor(num_gt_boxes):fill(1),
      torch.ByteTensor(num_boxes):fill(0)    )
   elseif num_boxes > 0 then
      rec.gt = torch.ByteTensor(num_boxes):fill(0)
   elseif num_gt_boxes > 0 then
      rec.gt = torch.ByteTensor(num_gt_boxes):fill(1)
   else
      rec.gt = torch.ByteTensor(0)
   end

   rec.overlap_class = torch.FloatTensor(num_boxes+num_gt_boxes,self:getNumClasses()):fill(0)
   rec.overlap = torch.FloatTensor(num_boxes+num_gt_boxes,num_gt_boxes):fill(0)
   for idx=1,num_gt_boxes do
      local o = utils.boxoverlap(all_boxes,gt_boxes[idx])
      local tmp = rec.overlap_class[{{},gt_classes[idx]}] -- pointer copy
      tmp[tmp:lt(o)] = o[tmp:lt(o)]
      rec.overlap[{{},idx}] = utils.boxoverlap(all_boxes,gt_boxes[idx])
   end
   -- get max class overlap
   --rec.overlap,rec.label = rec.overlap:max(2)
   --rec.overlap = torch.squeeze(rec.overlap,2)
   --rec.label   = torch.squeeze(rec.label,2)
   --rec.label[rec.overlap:eq(0)] = 0

   if num_gt_boxes > 0 then
      rec.overlap,rec.correspondance = rec.overlap:max(2)
      rec.overlap = torch.squeeze(rec.overlap,2)
      rec.correspondance   = torch.squeeze(rec.correspondance,2)
      rec.correspondance[rec.overlap:eq(0)] = 0
   else
      rec.overlap = torch.FloatTensor(num_boxes+num_gt_boxes):fill(0)
      rec.correspondance = torch.LongTensor(num_boxes+num_gt_boxes):fill(0)
   end
   rec.label = torch.IntTensor(num_boxes+num_gt_boxes):fill(0)

   do -- handle crowds
      -- find crowd boxes
      local crowds = {}
      for i,v in ipairs(anno) do
         if v.iscrowd then table.insert(crowds, v.bbox)end
      end
      if #crowds > 0 then
         -- compute intersections of all objects with each crowd
         local inters = torch.FloatTensor(#crowds, all_boxes:size(1))
         for i,v in ipairs(crowds) do
            inters[i] = utils.intersection(all_boxes, v)
         end
         local maxinters = inters:max(1)
         local mask = maxinters:gt(0.7):select(1,1)
         -- don't want to exclude ground truth boxes
         mask:narrow(1,1,num_gt_boxes):fill(0)
         rec.overlap:maskedFill(mask, -1)
      end
   end

   for idx=1,(num_boxes+num_gt_boxes) do
      local corr = rec.correspondance[idx]
      if corr > 0 then
         local obj = anno[valid_objects[corr] ]
         rec.label[idx] = obj.class_id
      end
   end

   rec.boxes = all_boxes
   if num_gt_boxes > 0 and num_boxes > 0 then
      rec.class = torch.cat(torch.CharTensor(gt_classes),
      torch.CharTensor(num_boxes):fill(0))
   elseif num_boxes > 0 then
      rec.class = torch.CharTensor(num_boxes):fill(0)
   elseif num_gt_boxes > 0 then
      rec.class = torch.CharTensor(gt_classes)
   else
      rec.class = torch.CharTensor(0)
   end

   function rec:size()
      return (num_boxes+num_gt_boxes)
   end

   return rec
end

return DataSetCOCO
