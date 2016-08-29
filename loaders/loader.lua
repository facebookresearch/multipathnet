--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

-- Loader for MSCOCO annotations

local ffi = require 'ffi'
local image = require 'image'
local coco = require 'coco'
local class = require 'class'

local dataset = class('loaders.dataLoader')

function createAnnotationList(m)
   local annotation_table = {}
   local annotations = {}
   local map = {}

   -- Find annotations for each image
   for i=1,m.images.id:numel() do
      map[i] = {}
   end
   for i=1,m.annotations.image_id:size(1) do
      local imageIdx = m.annotations.image_idx[i]
      table.insert(map[imageIdx], i)
   end

   for i,anns in ipairs(map) do
      table.insert(annotation_table, {#annotations + 1, #anns})
      for _,v in ipairs(anns) do
         table.insert(annotations, v)
      end
   end

   return {
      table = torch.IntTensor(annotation_table),
      annotations = torch.IntTensor(annotations),
   }
end

function dataset:load(path, image_dir)
   local cocoApi = coco.CocoApi(path)
   for k,v in pairs(cocoApi) do
      self[k] = v
   end
   self.images = self.data.images
   self.categories = {}
   for i,v in ipairs(cocoApi:getCatIds():totable()) do
      self.categories[i] = cocoApi:loadCats(v)[1].name
   end
   self.image_dir = image_dir
   self.annotationList = createAnnotationList(self.data)
   return self
end

-- Gets image properties as a table:
--  width, height, file_name, annotations
function dataset:getImage(idx)
   return {
      width = self.images.width[idx],
      height = self.images.height[idx],
      id = self.images.id[idx],
      file_name = ffi.string(self.images.file_name[idx]),
      annotations = self:getImageAnnotations(idx),
      image_dir = self.images.image_dir and self.images.image_dir[idx],
      idx = idx,
   }
end

-- Gets the image data as a ByteTensor
function dataset:loadImage(idx)
   local metadata = self:getImage(idx)
   local dir = self.image_dir or self.image_dir[metadata.image_dir]
   local path = paths.concat(dir, metadata.file_name)
   return image.load(path, 3, 'double')
end

-- Gets annotations:
--  bbox, polygons/rle, category, area, image
function dataset:getAnnotation(idx)
   local a = self.data.annotations
   assert(idx <= a.id:numel(), 'no annotation for '..idx)
   local iscrowd = a.iscrowd[idx] == 1

   local annotation = {
      bbox = a.bbox[idx],
      image = a.image_idx[idx],
      area = a.area[idx],
      category = a.category_idx[idx],
      iscrowd = iscrowd,
      idx = idx,
      difficult = a.ignore and a.ignore[idx] or 0,
   }

   return annotation
end

-- Category names
function dataset:categoryNames()
   local names = {}
   for _,cat in ipairs(self.categories) do
      table.insert(names, cat.name)
   end
   return names
end

-- Total number of categories (i.e. 80)
function dataset:nCategories()
   return #self.categories
end

-- Total number of categories (i.e. 80)
function dataset:nAnnotations()
   return self.annotations.image:size(1)
end

-- Total number of categories (i.e. 80)
function dataset:nImages()
   return self.images.id:size(1)
end

-- Random annotation for a given category
function dataset:randomAnnotation(category)
   local list = self.classListSample[category]
   local annotationIdx = list[math.ceil(torch.uniform() * list:nElement())]

   return self:getAnnotation(annotationIdx)
end

-- Indices of all labeled annotations for a given image
function dataset:getImageAnnotations(idx)
   local offset, len = table.unpack(self.annotationList.table[idx]:totable())
   if len == 0 then return {} end
   return self.annotationList.annotations:narrow(1, offset, len):totable()
end

-- All annotations for a given image
function dataset:getAnnotationsForImage(idx)
   local tbl = {}
   for _,annIdx in ipairs(self:getImageAnnotations(idx)) do
      table.insert(tbl, self:getAnnotation(annIdx))
   end
   return tbl
end

return dataset
