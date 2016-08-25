--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local Context, parent = torch.class('nn.ContextRegion','nn.Module')

-- Takes (Bx5) input in format {id,x1,x2,y1,y2}
-- and increases or decreases bounding boxes by 'scale' parameter

function Context:__init(scale)
   parent.__init(self)
   local a = (1 + scale) / 2
   local b = (1 - scale) / 2
   self.tr = torch.Tensor{
      {a, 0, b, 0},
      {0, a, 0, b},
      {b, 0, a, 0},
      {0, b, 0, a},
   }
end

function Context:updateOutput(input)
   assert(input:nDimension() == 2)
   assert(input:size(2) == 5)
   self.output:resizeAs(input):copy(input)
   self.output:narrow(2,2,4):mm(input:narrow(2,2,4), self.tr)
   return self.output
end

function Context:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   return self.gradInput
end

function Context:__tostring__()
   return torch.type(self)..'('..(self.tr[1][1]*2 - 1)..')'
end
