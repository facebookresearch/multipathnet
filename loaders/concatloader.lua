--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

-- Combines multiple coco.DataLoaders

local class = require 'class'

local ConcatLoader = class('ConcatLoader')

function ConcatLoader:__init(loaders)
   self.__loaders = loaders
   -- Offsets for images and annotations
   self.__imageOffset = {}
   self.categories = loaders[1].categories
   local i = 0, 0
   for _,loader in ipairs(loaders) do
      self.__imageOffset[loader] = i
      i = i + loader:nImages()
   end
end

function ConcatLoader:__getLoader(idx, sizeFn)
   local offset = idx
   for _,l in ipairs(self.__loaders) do
      local sz = l[sizeFn](l)

      if offset <= sz then
         return l, offset
      end
      offset = offset - sz
   end
   error('Invalid index: ' .. idx)
end

-- Remove indices into the data loader
function ConcatLoader.__removeOffsets(res)
   if torch.type(res) == 'table' then
      if #res > 0 then
         for i,r in ipairs(res) do
            res[i] = ConcatLoader.__removeOffsets(r)
         end
         return res
      end
      res.image = nil
      res.idx = nil
      res.annotations = nil
   end
   return res
end

function ConcatLoader:nCategories()
   return #self.categories
end
local function delegate(fn, sizeFn)
   return function(self, idx)
      local loader, i = self:__getLoader(idx, sizeFn)
      local res = loader[fn](loader, i)

      return ConcatLoader.__removeOffsets(res)
   end
end

local function delegateSum(fn)
   return function(self)
      local res = 0
      for _,loader in ipairs(self.__loaders) do
         res = res + loader[fn](loader)
      end
      return res
   end
end

ConcatLoader.getImage = delegate('getImage', 'nImages')
ConcatLoader.loadImage = delegate('loadImage', 'nImages')
ConcatLoader.getAnnotation = delegate('getAnnotation', 'nAnnotations')
ConcatLoader.getAnnotationsForImage = delegate('getAnnotationsForImage', 'nImages')
ConcatLoader.nImages = delegateSum('nImages')
ConcatLoader.nAnnotations = delegateSum('nAnnotations')

return ConcatLoader
