--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

-- View of subset of coco.DataLoader

local class = require 'class'

local NarrowLoader = class('NarrowLoader')
local ConcatLoader = require 'loaders.concatloader'

function NarrowLoader:__init(loader, start, len)
   assert(start > 0 and start <= loader:nImages(), 'invalid start: ' .. start)
   assert(len > 0 and start + len - 1 <= loader:nImages(), 'invalid len: ' .. len)
   self.__loader = loader
   self.__start = start
   self.__len = len
   self.categories = loader.categories
end

local function delegate(name)
   return function(self, idx)
      assert(idx >= 1 and idx <= self.__len, 'invalid index: ' .. idx)
      local res = self.__loader[name](self.__loader, idx + self.__start - 1)

      return ConcatLoader.__removeOffsets(res)
   end
end

NarrowLoader.getImage = delegate('getImage')
NarrowLoader.loadImage = delegate('loadImage')
NarrowLoader.getAnnotationsForImage = delegate('getAnnotationsForImage')

function NarrowLoader:getAnnotation(idx)
   local res = self.__loader:getAnnotation(idx)
   return ConcatLoader.__removeOffsets(res)
end

function NarrowLoader:nImages()
   return self.__len
end

function NarrowLoader:nCategories()
   return #self.categories
end

function NarrowLoader:nAnnotations()
   return self.__loader:nAnnotations()
end

return NarrowLoader
