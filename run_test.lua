--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

require 'torch'
local json = require 'cjson'
local test_runner = paths.dofile('test_runner.lua')
local utils = paths.dofile'utils.lua'
local tds = require 'tds'

opt = {
   dataset = 'pascal',
   year = '2007',
   proposals = 'deepmask',
   proposal_dir = './data/proposals',
   transformer = 'RossTransformer',
   scale = 600,
   max_size = 1000,
   test_nGPU = 4,
   test_set = 'test',
   test_nsamples = -1, -- all samples
   test_data_offset = -1, -- ignore the first "offset" samples
   test_model = './data/models/caffenet_fast_rcnn_iter_40000.t7',
   test_best_proposals_number = 500,
   test_load_aboxes = '',
   test_save_res_prefix = '',
   test_save_res = '',
   test_save_raw = '',
   test_num_iterative_loc = 1,
   disable_memory_efficient_forward = false,
   test_add_nosoftmax = false, -- for backwards compatibility with szagoruyko's experiments ONLY
   test_use_rbox_scores = false,
   test_bbox_voting = false,
   test_bbox_voting_score_pow = 1,
   test_augment = false,
   test_just_save_boxes = false,
   test_min_proposal_size = 2,
   test_nms_threshold = 0.3,
   test_bbox_voting_nms_threshold = 0.5,
}
opt = xlua.envparams(opt)
print(opt)

local dataset_name = opt.dataset..'_'..opt.test_set..opt.year
local folder_name = opt.dataset == 'pascal' and ('VOC'..opt.year) or 'coco'
local proposals_path = utils.makeProposalPath(opt.proposal_dir, folder_name, opt.proposals, opt.test_set)

print('dataset:',dataset_name)
print('proposals_path:',proposals_path)

test_runner:setup(opt.test_nGPU, dataset_name, proposals_path)

local aboxes

if opt.test_load_aboxes == '' then
   aboxes = test_runner:computeBBoxes()
else
   aboxes = torch.load(opt.test_load_aboxes)
end

local dir = opt.test_save_res
if opt.test_data_offset ~= -1 then
   dir = opt.test_data_offset
   dir = opt.test_save_res_prefix .. dir
end


if dir ~= '' then
   print("Saving boxes to " .. dir)
   paths.mkdir(dir)
   torch.save(('%s/boxes.t7'):format(dir), aboxes)
end

if not opt.test_just_save_boxes then
   local res = test_runner:evaluateBoxes(aboxes)

   if dir ~= '' then
      torch.save(dir..'/results.t7', res)
   end
end
