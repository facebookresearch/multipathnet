--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local BatchProviderROI, parent = torch.class('fbcoco.BatchProviderROI', 'fbcoco.BatchProviderBase')
local utils = paths.dofile'utils.lua'
local tablex = require'pl.tablex'

function BatchProviderROI:__init(dataset, imgs_per_batch, scale, max_size, transformer, fg_threshold, bg_threshold)
   assert(transformer,'must provide transformer!')

   self.dataset = dataset

   self.batch_size = 128
   self.fg_fraction = 0.25

   self.fg_threshold = fg_threshold
   self.bg_threshold = bg_threshold

   self.imgs_per_batch = imgs_per_batch or 2
   self.scale = scale or 600
   self.max_size = max_size or 1000
   self.image_transformer = transformer

   self.scale_jitter    = scale_jitter or 0    -- uniformly jitter the scale by this frac
   self.aspect_jitter   = aspect_jitter or 0   -- uniformly jitter the scale by this frac
   self.crop_likelihood = crop_likelihood or 0 -- likelihood of doing a random crop (in each dimension, independently)
   self.crop_attempts = 10                     -- number of attempts to try to find a valid crop
   self.crop_min_frac = 0.7                             -- a crop must preserve at least this fraction of the iamge
end

-- Prepare foreground / background rois for one image
-- there is a check if self.bboxes has a table prepared for this image already
-- because we prepare the rois during training to save time on loading
function BatchProviderROI:setupOne(i)
   local rec = self.dataset:attachProposals(i)

   local bf = rec.overlap:ge(self.fg_threshold):nonzero()
   local bg = rec.overlap:ge(self.bg_threshold[1]):cmul(
   rec.overlap:lt(self.bg_threshold[2])):nonzero()
   return {
      [0] = self.takeSubset(rec, bg, i, true),
      [1] = self.takeSubset(rec, bf, i, false)
   }
end

-- Calculate rois and supporting data for the first 1000 images
-- to compute mean/var for bbox regresion
function BatchProviderROI:setupData()
   local regression_values = {}
   local subset_size = 1000
   for i = 1,1000 do
      local v = self:setupOne(i)[1]
      if v then
         table.insert(regression_values, utils.convertTo(v.rois, v.gtboxes))
      end
   end
   regression_values = torch.FloatTensor():cat(regression_values,1)

   self.bbox_regr = {
      mean = regression_values:mean(1),
      std = regression_values:std(1)
   }
   return self.bbox_regr
end

-- sample until find a valid combination of bg/fg boxes
function BatchProviderROI:permuteIdx()
   local boxes, img_idx = {}, {}
   for i=1,self.imgs_per_batch do
      local curr_idx
      local bboxes = {}
      while not bboxes[0] or not bboxes[1] do
         curr_idx = torch.random(self.dataset:size())
         tablex.update(bboxes, self:setupOne(curr_idx))
      end
      table.insert(boxes, bboxes)
      table.insert(img_idx, curr_idx)
   end
   local do_flip = torch.FloatTensor(self.imgs_per_batch):random(0,1)
   return torch.IntTensor(img_idx), boxes, do_flip
end

function BatchProviderROI:selectBBoxes(boxes, im_scales, im_sizes, do_flip)
   local rois = {}
   local labels = {}
   local gtboxes = {}
   for im,v in ipairs(boxes) do
      local flip = do_flip[im] == 1

      local bg = self.selectBBoxesOne(v[0],self.bg_num_each,im_scales[im],im_sizes[im],flip)
      local fg = self.selectBBoxesOne(v[1],self.fg_num_each,im_scales[im],im_sizes[im],flip)

      local imrois = torch.FloatTensor():cat(bg.rois, fg.rois, 1)
      imrois = torch.FloatTensor(imrois:size(1),1):fill(im):cat(imrois, 2)
      local imgtboxes = torch.FloatTensor():cat(bg.gtboxes, fg.gtboxes, 1)
      local imlabels = torch.IntTensor():cat(bg.labels, fg.labels, 1)

      table.insert(rois, imrois)
      table.insert(gtboxes, imgtboxes)
      table.insert(labels, imlabels)
   end
   gtboxes = torch.FloatTensor():cat(gtboxes,1)
   rois = torch.FloatTensor():cat(rois,1)
   labels = torch.IntTensor():cat(labels,1)
   return rois, labels, gtboxes
end


function BatchProviderROI:sample()
   collectgarbage()
   self.fg_num_each = self.fg_fraction * self.batch_size
   self.bg_num_each = self.batch_size - self.fg_num_each

   local img_idx, boxes, do_flip = self:permuteIdx()
   local images, im_scales, im_sizes = self:getImages(img_idx, do_flip)
   local rois, labels, gtboxes = self:selectBBoxes(boxes, im_scales, im_sizes, do_flip)

   local bboxregr_vals = torch.FloatTensor(rois:size(1), 4*(self.dataset:getNumClasses() + 1)):zero()

   for i,label in ipairs(labels:totable()) do
      if label > 1 then
         local out = bboxregr_vals[i]:narrow(1,(label-1)*4 + 1,4)
         utils.convertTo(out, rois[i]:narrow(1,2,4), gtboxes[i])
         out:add(-1,self.bbox_regr.mean):cdiv(self.bbox_regr.std)
      end
   end

   local batches = {images, rois}
   local targets = {labels, {labels, bboxregr_vals}, g_donkey_idx}

   return batches, targets
end
