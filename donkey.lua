--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local tnt = require 'torchnet'
local utils = paths.dofile'utils.lua'
require 'fbcoco'

function loadDataSet(opt)
   local dataset_name = opt.dataset..'_'..opt.train_set..opt.year
   local folder_name = opt.dataset == 'pascal' and ('VOC'..opt.year) or 'coco'
   local proposals_path = utils.makeProposalPath(opt.proposal_dir, folder_name, opt.proposals, opt.train_set, opt.imagenet_classes ~= '')

   local ds = paths.dofile'DataSetJSON.lua':create(dataset_name, proposals_path, opt.train_nsamples)
   if opt.imagenet_classes ~= '' then
      ds:allowMissingProposals(true) -- workaround
   end

   ds.sample_n_per_box = opt.sample_n_per_box
   ds.sample_sigma = opt.sample_n_per_box

   ds:setMinProposalArea(opt.train_min_proposal_size)
   -- ds:loadROIDB(opt.best_proposals_number)
   ds:setMinArea(opt.train_min_gtroi_size)
   return ds
end

function createTrainLoader(opt, roidb, scoredb, loader_idx)
   local ds = loadDataSet(opt)
   ds.roidb, ds.scoredb = roidb, scoredb
   local transformer = torch.load(opt.transformer)

   local fg_threshold, bg_threshold
   if opt.integral then
      local threshold = opt.bg_threshold_max + (loader_idx - 1) / 20
      bg_threshold = {opt.bg_threshold_min, threshold}
      fg_threshold = threshold
   else
      bg_threshold = {opt.bg_threshold_min, opt.bg_threshold_max}
      fg_threshold = opt.fg_threshold
   end

   local bp = fbcoco.BatchProviderROI(ds, opt.images_per_batch, opt.scale, opt.max_size, transformer, fg_threshold, bg_threshold) 

   bp.batch_size = opt.batchSize
   bp.class_specific = opt.train_class_specific

   return bp
end


