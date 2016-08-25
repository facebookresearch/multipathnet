--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local SelectBoxes,parent = torch.class('nn.SelectBoxes','nn.Module')

-- Input:
-- * SoftMax output (eg. 128 x 21)
-- * Bbox regresson output (eg. 128 x 84)
-- Output:
-- * boxes corresponding to the best classes (eg. 128 x 4)
-- Optionally renormalizes by multiplying by std and adding mean
-- if sigma_hat and sigma_mean are present in the self.

-- was lazy to finish CPU side
function SelectBoxes:__init()
   parent.__init(self)
   self.gradInput_classes = torch.Tensor()
   self.gradInput_boxes = torch.Tensor()
end

function SelectBoxes:updateOutput(input)
   local classes = input[1]
   local ys = input[2]

   local B = classes:size(1)
   self.maxvals = self.maxvals or classes.new()
   self.maxids = self.maxids or classes.new()
   self.ids = self.ids or classes.new()

   local maxvals = self.maxvals:resize(B,1)
   local maxids = self.maxids:resize(B,1)
   local ids = self.ids:resize(B,4)

   torch.max(maxvals, maxids, classes, 2)

   maxids:add(-1):mul(4)
   for i=1,4 do ids:select(2,i):fill(i) end
   ids:add(maxids:expand(B,4))
   self.output:resize(B,4):gather(ys, 2, ids)

   if not self.std then
      --print'dry run, using 0-1 mean-sigma in nn.SelectBoxes'
   else
      -- renormalize output
      local mu = self.mean:expandAs(self.output)
      local sigma = self.std:expandAs(self.output)
      self.output:cmul(sigma):add(mu)
   end

   return self.output
end

function SelectBoxes:updateGradInput(input,gradOutput)
   self.gradInput_classes:resizeAs(input[1]):zero()
   self.gradInput_boxes:resizeAs(input[2]):zero()
   self.gradInput = {self.gradInput_classes, self.gradInput2}
   return self.gradInput
end

function test()
   module = nn.SelectBoxes():cuda()
   local classes = torch.Tensor{
      {0,1,0},
      {1,0,0}
   }:cuda()
   local ys = torch.rand(2,3*4):cuda()

   local output = module:forward{classes, ys}:cuda()
   print(ys, output)
end

--test()
