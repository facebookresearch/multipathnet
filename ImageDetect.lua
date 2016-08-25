--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local utils = paths.dofile'utils.lua'
local ImageDetect = torch.class('fbcoco.ImageDetect')

function ImageDetect:__init(model, transformer, scale, max_size)
   assert(model, 'must provide model!')
   assert(transformer, 'must provide transformer!')
   self.model = model
   self.image_transformer = transformer
   self.scale = scale or {600}
   self.max_size = max_size or 1000
   self.sm = nn.SoftMax():cuda()
end

local function getImages(self,images,im)
   local num_scales = #self.scale

   local imgs = {}
   local im_sizes = {}
   local im_scales = {}

   im = self.image_transformer:forward(im)

   local im_size = im[1]:size()
   local im_size_min = math.min(im_size[1],im_size[2])
   local im_size_max = math.max(im_size[1],im_size[2])
   for i=1,num_scales do
      local im_scale = self.scale[i]/im_size_min
      if torch.round(im_scale*im_size_max) > self.max_size then
         im_scale = self.max_size/im_size_max
      end
      local im_s = {im_size[1]*im_scale,im_size[2]*im_scale}
      table.insert(imgs,image.scale(im,im_s[2],im_s[1]))
      table.insert(im_sizes,im_s)
      table.insert(im_scales,im_scale)
   end
   -- create single tensor with all images, padding with zero for different sizes
   im_sizes = torch.IntTensor(im_sizes)
   local max_shape = im_sizes:max(1)[1]
   images:resize(num_scales,3,max_shape[1],max_shape[2]):zero()
   for i=1,num_scales do
      images[i][{{},{1,imgs[i]:size(2)},{1,imgs[i]:size(3)}}]:copy(imgs[i])
   end
   return im_scales
end

local function project_im_rois(im_rois,scales)
   local levels
   local rois = torch.FloatTensor()
   if #scales > 1 then
      local scales = torch.FloatTensor(scales)
      local widths = im_rois[{{},3}] - im_rois[{{},1}] + 1
      local heights = im_rois[{{},4}] - im_rois[{{}, 2}] + 1

      local areas = widths * heights
      local scaled_areas = areas:view(-1,1) * torch.pow(scales:view(1,-1),2)
      local diff_areas = torch.abs(scaled_areas - 224 * 224)
      levels = select(2, diff_areas:min(2))
   else
      levels = torch.FloatTensor()
      rois:resize(im_rois:size(1),5)
      rois[{{},1}]:fill(1)
      rois[{{},{2,5}}]:copy(im_rois):add(-1):mul(scales[1]):add(1)
   end
   return rois
end

local function recursiveSplit(x, bs, dim)
   if type(x) == 'table' then
      local res = {}
      for k,v in pairs(x) do
         local tmp = v:split(bs,dim)
         for i=1,#tmp do
            if not res[i] then res[i] = {} end
            res[i][k] = tmp[i]
         end
      end
      return res
   else
      return x:split(bs, dim)
   end
end

function ImageDetect:memoryEfficientForward(model, input, bs, recompute_features)
   local images = input[1]
   local rois = input[2]
   local recompute_features = recompute_features == nil and true or recompute_features
   assert(model.output[1]:numel() > 0)

   local rest = nn.Sequential()
   for i=2,#model.modules do rest:add(model:get(i)) end
   local final = model:get(#model.modules)

   -- assuming the net has bbox regression part
   self.output = self.output or {torch.CudaTensor(), torch.CudaTensor()}
   local num_classes = self.model.output[1]:size(2)
   self.output[1]:resize(rois:size(1), num_classes)
   self.output[2]:resize(rois:size(1), num_classes * 4)

   if recompute_features then
      model:get(1):forward{images,rois}
   else
      model:get(1).output[2] = rois
   end

   local features = model:get(1).output
   assert(features[2]:size(1) == rois:size(1))

   local roi_split = features[2]:split(bs,1)
   local output1_split = self.output[1]:split(bs,1)
   local output2_split = self.output[2]:split(bs,1)

   for i,v in ipairs(roi_split) do
      local out = rest:forward({features[1], v})
      output1_split[i]:copy(out[1])
      output2_split[i]:copy(out[2])
   end

   local function test()
      local output_full = model:forward({images,rois})

      local output_split = self.output
      assert((output_full[1] - output_split[1]):abs():max() == 0)
      assert((output_full[2] - output_split[2]):abs():max() == 0)
   end
   --test()
   return self.output
end

function ImageDetect:computeRawOutputs(im, boxes, min_images, recompute_features)
   self.model:evaluate()

   local inputs = {torch.FloatTensor(),torch.FloatTensor()}
   local im_scales = getImages(self,inputs[1],im)
   inputs[2] = project_im_rois(boxes,im_scales)
   if min_images then
      assert(inputs[1]:size(1) == 1)
      inputs[1] = inputs[1]:expand(min_images, inputs[1]:size(2), inputs[1]:size(3), inputs[1]:size(4))
   end

   self.inputs_cuda = self.inputs_cuda or {torch.CudaTensor(),torch.CudaTensor()}
   self.inputs_cuda[1]:resize(inputs[1]:size()):copy(inputs[1])
   self.inputs_cuda[2]:resize(inputs[2]:size()):copy(inputs[2])

   return self.model:forward(self.inputs_cuda)
end

-- supposes boxes is in [x1,y1,x2,y2] format
function ImageDetect:detect(im, boxes, min_images, recompute_features)
   self.model:evaluate()

   local inputs = {torch.FloatTensor(),torch.FloatTensor()}
   local im_scales = getImages(self,inputs[1],im)
   inputs[2] = project_im_rois(boxes,im_scales)
   if min_images then
      assert(inputs[1]:size(1) == 1)
      inputs[1] = inputs[1]:expand(min_images, inputs[1]:size(2), inputs[1]:size(3), inputs[1]:size(4))
   end

   self.inputs_cuda = self.inputs_cuda or {torch.CudaTensor(),torch.CudaTensor()}
   self.inputs_cuda[1]:resize(inputs[1]:size()):copy(inputs[1])
   self.inputs_cuda[2]:resize(inputs[2]:size()):copy(inputs[2])

   local output0
   if opt and opt.disable_memory_efficient_forward then
      print('memory efficient forward disabled')
      output0 = self.model:forward(self.inputs_cuda)
   else
      output0 = self:memoryEfficientForward(self.model, self.inputs_cuda, 500, recompute_features)
   end

   local class_values, bbox_values
   if torch.type(output0) == 'table' then
      class_values= output0[1]
      bbox_values = output0[2]:float()
      for i,v in ipairs(bbox_values:split(4,2)) do
         utils.convertFrom(v,boxes,v)
      end
   else
      class_values = output0
   end
   if not self.model.noSoftMax then
      class_values = self.sm:forward(class_values)
   end
   return class_values:float(), bbox_values
end
