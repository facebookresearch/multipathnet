--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local tnt = require 'torchnet'
require 'donkey'

-- create an instance of DataSetJSON to make roidb and scoredb
-- that are passed to threads
local roidb, scoredb 
do
   local ds = loadDataSet(opt)
   ds:loadROIDB(opt.best_proposals_number)
   roidb, scoredb = ds.roidb, ds.scoredb
end

local loader = createTrainLoader(opt, roidb, scoredb, 1)
local bbox_regr = loader:setupData()
g_mean_std = bbox_regr

local opt = tnt.utils.table.clone(opt)

local function getIterator()
   return tnt.ParallelDatasetIterator{
      nthread = opt.nDonkeys,
      init = function(idx)
         require 'torchnet'
         require 'donkey'
         torch.manualSeed(opt.manualSeed + idx)
         g_donkey_idx = idx
      end,
      closure = function()
         local loaders = {}
         for i=1,(opt.integral and opt.nDonkeys or 1) do
            loaders[i] = createTrainLoader(opt, roidb, scoredb, i)
         end

         for i,v in ipairs(loaders) do
            v.bbox_regr = bbox_regr
         end

         return tnt.ListDataset{
            list = torch.range(1,opt.epochSize):long(),
            load = function(idx)
               return {loaders[torch.random(#loaders)]:sample()}
            end,
         }
      end,
   }
end

return getIterator
