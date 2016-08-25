--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

-- uses the 'donkey' pattern
-- constructs threads for running the model on multiple GPUs

local module = {}

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')
local tds = require 'tds'

local function _setup(dataset_name, proposals_path)
   require 'cutorch'
   require 'fbcoco'
   require 'inn'
   require 'cudnn'
   require 'nngraph'
   local utils = paths.dofile 'utils.lua'
   local model_utils = paths.dofile 'models/model_utils.lua'

   nn.DataParallelTable.deserializeNGPUs = cutorch.getDeviceCount()
   nn.ModelParallelTable.deserializeNGPUs = cutorch.getDeviceCount()

   local transformer = model_utils[opt.transformer]()
   local model = model_utils.load(opt.test_model):cuda()
   if opt.test_nGPU > 1 then
      utils.removeDataParallel(model)
   end
   utils.removeDataParallel(model) -- TODO: see why it complains

   model:evaluate()
   if opt.test_add_nosoftmax then
      print("Setting noSoftMax=true")
      model.noSoftMax = true
   end
   -- patch to use inplace dropout everywhere
   for k,v in ipairs(model:findModules'nn.Dropout') do v.inplace = true end
   ds = paths.dofile'DataSetJSON.lua':create(dataset_name, proposals_path, opt.test_nsamples, opt.test_data_offset)
   ds:setMinProposalArea(opt.test_min_proposal_size)
   ds:loadROIDB(opt.test_best_proposals_number)
   tester = fbcoco.Tester_FRCNN(model, transformer, ds, {opt.scale}, opt.max_size, opt)
end

function module:setup(nThreads, dataset_name, proposals_path)
   self.nThreads = nThreads
   if self.nThreads > 1 then
      _setup(dataset_name, proposals_path)
      local _opt = opt
      self.threads = Threads(self.nThreads,
      function()
         require 'cutorch'
      end,
      function(idx)
         thread_idx = idx
         opt = _opt
         local dev = idx % cutorch.getDeviceCount()
         dev = (dev==0) and cutorch.getDeviceCount() or dev
         cutorch.setDevice(dev)
         _setup(dataset_name, proposals_path)
      end)
   else
      self.threads = {
         addjob = function(self, f1, f2)
            if f2 then
               return f2(f1())
            else
               f1()
            end
         end,
         synchronize = function() end,
      }
      require 'cutorch'
      _setup(dataset_name, proposals_path)
   end
   return self
end

-- go over all images in the dataset and the proposals and extract the
-- classes and bbox predictions
function module:computeBBoxes()
   local aboxes_t = {}
   local raw_output = tds.hash()
   local raw_bbox_pred = tds.hash()
   local timer = torch.Timer()
   for i=1, ds:size() do
      self.threads:addjob(
      function()
         return tester:testOne(i)
      end,
      function(res, raw_res)
         aboxes_t[i] = res
         if opt.test_save_raw ~= '' then
            raw_output[i] = raw_res[1]:float()
            raw_bbox_pred[i] = raw_res[2]:float()
         end
      end
      )
   end
   self.threads:synchronize()
   print("Finished with images in " .. timer:time().real .. " s")

   if opt.test_save_raw ~= '' then
      torch.save(opt.test_save_raw, {raw_output, raw_bbox_pred})
      print('Saved raw bboxes at: ' , opt.test_save_raw)
   end

   for i = 1,self.nThreads do
      self.threads:addjob(
      function() collectgarbage(); collectgarbage(); end)
   end
   self.threads:synchronize()
   self.threads = nil
   collectgarbage(); collectgarbage();
   print("Thread garbage collected")
   aboxes_t = tester:keepTopKPerImage(aboxes_t, 100) -- coco only accepts 100/image
   local aboxes = tester:transposeBoxes(aboxes_t)
   aboxes_t = nil
   collectgarbage(); collectgarbage();
   return aboxes
end

-- validation only function
function module:evaluateBoxes(aboxes)
   return tester:computeAP(aboxes)
end

function module:test()
   local aboxes = self:computeBBoxes()
   return self:evaluateBoxes(aboxes)
end

return module
