--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local inn = require 'inn'
require 'fbcoco'

local utils = paths.dofile'utils.lua'

local mytester = torch.Tester()
local utiltest = torch.TestSuite()

function utiltest.bboxregression_parametrization()
   local A = torch.rand(2) * 100
   local B = torch.rand(2) * 100
   local bbox = torch.Tensor{A[1], A[2], A[1] + torch.random(40), A[2] + torch.random(40)}
   local tbox = torch.Tensor{B[1], B[2], B[1] + torch.random(40), B[2] + torch.random(40)}
   local out = torch.zeros(4)

   -- test 1-dim
   utils.convertTo(out, bbox, tbox)
   local out1 = torch.zeros(4)
   utils.convertFrom(out1, bbox, out)
   mytester:assertlt((out1 - tbox):abs():max(), 1e-8)

   -- test 2-dim
   local out2 = torch.zeros(1,4)
   utils.convertTo(out2, bbox:view(1,4), tbox:view(1,4))
   mytester:assertlt((out2:squeeze() - out):abs():max(), 1e-8)

   local out3 = torch.zeros(1,4)
   utils.convertFrom(out3, bbox:view(1,4), out:view(1,4))
   mytester:assertlt((out3 - out1):abs():max(), 1e-8)
end

function utiltest.boxoverlap()
   local a = torch.Tensor{
      {0,0,100,100},
      {0,50,100,150},
      {50,0,150,100},
      {50,50,150,150},
      {100,100,200,200}
   }
   local b = {50,50,150,150}

   local gt = torch.FloatTensor{1/7, 1/3, 1/3, 1, 1/7}
   mytester:assertlt((utils.boxoverlap(a,b) - gt):max(),5e-3)
end

function utiltest.attachProposals()
   local dataset_name = 'pascal_test2007'
   local proposals_path = 'data/proposals/VOC2007/selective_search/test.t7'

   local ds = dofile'DataSetJSON.lua':create(dataset_name, proposals_path)
   ds:loadROIDB(500)

   mytester:assertgt(ds:size(), 0)

   -- go over some annotations and check that they are in the right format
   for i=1,32 do
      local id = torch.random(ds:size())

      -- load an image and check that it has 3 channels
      local im = ds:getImage(id)
      mytester:asserteq(im:nDimension(), 3)

      -- annotation check
      local anno = ds:getAnnotation(1)
      local obj = anno[1]
      -- check that annotation has 'difficult' field
      mytester:assert(obj.difficult ~= nil)
      mytester:assertgt(obj.class_id, 0)
      -- check that the bbox is x1,y1,x2,y2
      mytester:assertgt(obj.bbox[3], obj.bbox[1])
      mytester:assertgt(obj.bbox[4], obj.bbox[2])

      -- check that proposals are in x1,y1,x2,y2 too
      local proposals = ds:getROIBoxes(id)
      mytester:assertgt(proposals:select(2,3):gt(proposals:select(2,1)):float():mean(), 0.9)
      mytester:assertgt(proposals:select(2,4):gt(proposals:select(2,2)):float():mean(), 0.9)
   end
end

function utiltest.merge_table()
   local t1, t2, t3 = {x = 1}, {y = 2}, {z = 3}
   local t = utils.merge_table{t1,t2,t3}
   assert(t.x == t1.x and t.y == t2.y and t.z == t3.z)
end

local precision = 1e-3

local nntest = torch.TestSuite()

local function criterionJacobianTest1D(cri, input, target)
   local eps = 1e-6
   local _ = cri:forward(input, target)
   local dfdx = cri:backward(input, target)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(dfdx)
   local input_s = input:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,input:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end

   -- compare centraldiff_dfdx with :backward()
   local err = (centraldiff_dfdx - dfdx):abs():max()
   mytester:assertlt(err, precision, 'error in difference between central difference and :backward')
end


function nntest.BBoxRegressionCriterion()
   local bs = torch.random(16,32)
   local input = torch.randn(bs, 84)
   local bbox_targets = torch.randn(bs, 84):zero()
   local bbox_labels = torch.Tensor(bs):random(2,21)
   for i=1,bs do
      bbox_targets[i]:narrow(1,(bbox_labels[i]-1)*4 + 1, 4)
   end
   local target = {bbox_labels, bbox_targets}
   local cri = nn.BBoxRegressionCriterion()
   criterionJacobianTest1D(cri, input, target)
end

function nntest.SequentialSplitBatch_ROIPooling()
   local input = {
      torch.randn(1,512,38,50):cuda(),
      torch.randn(40,5):cuda():mul(50),
   }
   input[2]:select(2,1):fill(1)

   local module = nn.SequentialSplitBatch(25)
   :add(inn.ROIPooling(7,7,1/16))
   :add(nn.View(-1):setNumInputDims(3))
   :add(nn.Linear(7*7*512,9))
   :cuda()

   local output_mod = module:forward(input):clone()
   output_mod = module:forward(input):clone()
   local output_ref = module:replace(function(x)
      if torch.typename(x) == 'nn.SequentialSplitBatch' then
         torch.setmetatable(x, 'nn.Sequential')
      end
      return x
   end):forward(input):clone()

   mytester:asserteq((output_mod - output_ref):abs():max(), 0, 'SequentialSplitBatch err')
end

function nntest.SequentialSplitBatch_Tensor()
   local input = torch.randn(40,512):cuda()
   local module = nn.SequentialSplitBatch(25):add(nn.Linear(512,9)):cuda()

   local output_mod = module:forward(input):clone()
   output_mod = module:forward(input):clone()
   local output_ref = module:replace(function(x)
      if torch.typename(x) == 'nn.SequentialSplitBatch' then
         torch.setmetatable(x, 'nn.Sequential')
      end
      return x
   end):forward(input):clone()

   mytester:asserteq((output_mod - output_ref):abs():max(), 0, 'SequentialSplitBatch err')
end

mytester:add(utiltest)
mytester:add(nntest)
mytester:run()
