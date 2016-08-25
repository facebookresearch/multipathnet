--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

require 'fbcoco'
require 'inn'

local mytester = torch.Tester()

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

mytester:add(nntest)
mytester:run()
