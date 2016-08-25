--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local ImageTransformer, parent = torch.class('fbcoco.ImageTransformer', 'nn.Module')

function ImageTransformer:__init(mean,std,scale,swap)
   parent.__init(self)
   self.mean = mean
   self.std = std
   self.scale = scale or 1
   self.swap = swap
end

function ImageTransformer:updateOutput(I)
   assert(I:nDimension() == 3)
   I = self.swap and I:index(1,torch.LongTensor(self.swap)) or I:clone()
   if self.scale ~= 1 then
      I:mul(self.scale)
   end
   for i=1,3 do
      I[i]:add(-self.mean[i])
      if self.std then
         I[i]:div(self.std[i])
      end
   end
   self.output = I
   return I
end
