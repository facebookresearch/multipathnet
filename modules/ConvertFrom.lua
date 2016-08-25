--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

require 'nn'

-- same as utils.ConvertFrom
-- reparametrization of source bbox -> target bbox relashionship

local module, parent = torch.class('nn.ConvertFrom','nn.Module')

function module:__init()
   parent.__init(self)
   self.gradInput1 = torch.Tensor()
   self.gradInput2 = torch.Tensor()
end

function module:updateOutput(input)
   local roi_boxes = input[1]
   local y = input[2]

   local bbox = roi_boxes:narrow(2,2,4)

   self.output:resizeAs(roi_boxes):copy(roi_boxes)
   local out = self.output:narrow(2,2,4)

   assert(bbox:size(2) == y:size(2))
   assert(bbox:size(2) == out:size(2))
   assert(bbox:size(1) == y:size(1))
   assert(bbox:size(1) == out:size(1))

   local xc = (bbox[{{},1}] + bbox[{{},3}]) * 0.5
   local yc = (bbox[{{},2}] + bbox[{{},4}]) * 0.5
   local w = bbox[{{},3}] - bbox[{{},1}]
   local h = bbox[{{},4}] - bbox[{{},2}]

   local xtc = torch.addcmul(xc, y[{{},1}], w)
   local ytc = torch.addcmul(yc, y[{{},2}], h)
   local wt = torch.exp(y[{{},3}]):cmul(w)
   local ht = torch.exp(y[{{},4}]):cmul(h)

   out[{{},1}] = xtc - wt * 0.5
   out[{{},2}] = ytc - ht * 0.5
   out[{{},3}] = xtc + wt * 0.5
   out[{{},4}] = ytc + ht * 0.5

   return self.output
end

function module:updateGradInput(input, gradOutput)
   self.gradInput1:resizeAs(input[1])
   self.gradInput2:resizeAs(input[2])
   self.gradInput = {self.gradInput1, self.gradInput2}
   return self.gradInput
end
