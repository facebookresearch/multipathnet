--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

-- script to run the python coco tester on the saved results file from run_test.lua
--

local testCoco = {}

local Coco = require 'testCoco.coco'
local loader = require 'loaders.dataloader'
require 'xlua'

local function getAboxes(res, class)
    if type(res) == 'string' then -- res_folder
        return torch.load(('%s/%.2d.t7'):format(res, class))
    elseif type(res) == 'table' or type(res) == 'cdata' then -- table or tds.hash
        return res[class]
    else
        error("Unknown res object: type " .. type(res))
    end
end

local annotations_path = 'data/annotations/'

function testCoco.evaluate(dataset_name, res)
   local annFile
   if dataset_name == 'coco_val2014' then
      annFile = 'instances_val2014.json'
   elseif dataset_name == 'pascal_test2007' then
      annFile = 'pascal_test2007.json'
   end
   annFile = paths.concat(annotations_path, annFile)

   local dataset = loader(dataset_name)

   print("Loading COCO image ids...")
   local image_ids = {}
   for i = 1, dataset:nImages() do
     if i % 10000 == 0 then print("  "..i..'/'..dataset:nImages()) end
     image_ids[i] = dataset:getImage(i).id
   end
   print('#image_ids',#image_ids)

   local nClasses = dataset:nCategories()

   print("Loading files to calculate sizes...")
   local nboxes = 0
   for class = 1, nClasses do
     local aboxes = getAboxes(res, class)

     for _,u in pairs(aboxes) do
       if u:nDimension() > 0 then
         nboxes = nboxes + u:size(1)
       end
     end
     -- xlua.progress(class, nClasses)
   end
   print("Total boxes: " .. nboxes)

   local boxt = torch.FloatTensor(nboxes, 7)

   print("Loading files to create giant tensor...")
   local offset = 1
   for class = 1, nClasses do
     local aboxes = getAboxes(res, class)
     for img,t in pairs(aboxes) do
       if t:nDimension() > 0 then
         local sub = boxt:narrow(1,offset,t:size(1))
         sub:select(2, 1):fill(image_ids[img]) -- image ID
         sub:select(2, 2):copy(t:select(2, 1) - 1) -- x1 0-indexed
         sub:select(2, 3):copy(t:select(2, 2) - 1) -- y1 0-indexed
         sub:select(2, 4):copy(t:select(2, 3) - t:select(2, 1)) -- w
         sub:select(2, 5):copy(t:select(2, 4) - t:select(2, 2)) -- h
         sub:select(2, 6):copy(t:select(2, 5)) -- score
         sub:select(2, 7):fill(dataset.data.categories.id[class])    -- class
         offset = offset + t:size(1)
       end
     end
     -- xlua.progress(class, nClasses)
   end

   local coco = Coco(annFile)
   return coco:evaluate(boxt)
end

return testCoco
