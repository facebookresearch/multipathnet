--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local BatchProviderBase = torch.class('fbcoco.BatchProviderBase')

function BatchProviderBase:getImages(img_ids, do_flip)
   local num_images = img_ids:size(1)

   local imgs = {}
   local im_sizes = {}
   local im_scales = {}

   for i=1,num_images do
      local im = self.dataset:getImage(img_ids[i])
      im = self.image_transformer(im)
      local flip = do_flip[i] == 1
      if flip then im = image.hflip(im) end
      local im_size = im[1]:size()
      local im_size_min = math.min(im_size[1],im_size[2])
      local im_size_max = math.max(im_size[1],im_size[2])
      local im_scale = self.scale/im_size_min
      local aspect_jitter = 1 + (torch.uniform(-1,1)-0.5)*self.aspect_jitter
      local scale_jitter  = 1 + (torch.uniform(-1,1)-0.5)*self.scale_jitter
      local im_scale = im_scale * scale_jitter
      im_scale = {im_scale * math.sqrt(aspect_jitter), im_scale / math.sqrt(aspect_jitter)}
      local im_s = {im_size[1]*im_scale[1],im_size[2]*im_scale[1]}
      for dim = 1,2 do
         if im_s[dim] > self.max_size then
            local rat = im_s[dim] / self.max_size
            im_s = {im_s[1] / rat, im_s[2] / rat}
            im_scale = {im_scale[1] / rat, im_scale[2] / rat}
         end
      end
      table.insert(imgs,image.scale(im,im_s[2],im_s[1]))
      table.insert(im_sizes,im_s)
      table.insert(im_scales,im_scale)
   end
   -- create single tensor with all images, padding with zero for different sizes
   im_sizes = torch.IntTensor(im_sizes)
   local max_shape = im_sizes:max(1)[1]
   local images = torch.FloatTensor(num_images,3,max_shape[1],max_shape[2]):zero()
   for i,v in ipairs(imgs) do
      images[{i, {}, {1,v:size(2)}, {1,v:size(3)}}]:copy(v)
   end
   return images, im_scales, im_sizes
end


function BatchProviderBase.takeSubset(rec, t, i, is_bg)
   local idx = torch.type(t) == 'table' and torch.LongTensor(t) or t:long()
   local n = idx:numel()
   if n == 0 then return end
   if idx:dim() == 2 then idx = idx:select(2,1) end
   local window = {
      indexes = torch.IntTensor(n),
      rois = torch.FloatTensor(n,4),
      labels = torch.IntTensor(n):fill(1),
      gtboxes = torch.FloatTensor(n,4):zero(),
      size = function() return n end,
   }
   window.indexes:fill(i)
   window.rois:copy(rec.boxes:index(1,idx))
   if not is_bg then
      window.labels:add(rec.label:index(1,idx))
      local corresp = rec.correspondance:index(1,idx)
      window.gtboxes:copy(rec.boxes:index(1, corresp))
   end
   return window
end


function BatchProviderBase.selectBBoxesOne(bboxes, num_max, im_scale, im_size, flip)
   local rois = {}
   local labels = {}
   local gtboxes = {}

   local n = bboxes:size()
   local im_scale = torch.FloatTensor(im_scale):repeatTensor(2)

   local function preprocess_bbox(dd, flip)
      dd = dd:clone():add(-1):cmul(im_scale):add(1)
      if flip then
         local tt = dd[1]
         dd[1] = im_size[2]-dd[3] +1
         dd[3] = im_size[2]-tt    +1
      end
      return dd:view(1,4)
   end

   for i=1,math.min(num_max, n) do
      local position = torch.random(n)
      table.insert(rois,    preprocess_bbox(bboxes.rois[position],flip))
      table.insert(gtboxes, preprocess_bbox(bboxes.gtboxes[position], flip))
      table.insert(labels, bboxes.labels[position])
   end

   return {
      gtboxes = torch.FloatTensor():cat(gtboxes,1),
      rois = torch.FloatTensor():cat(rois,1),
      labels = torch.IntTensor(labels),
   }
end

