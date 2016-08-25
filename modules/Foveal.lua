--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local Foveal, parent = torch.class('nn.Foveal','nn.Module')

function Foveal:__init()
   parent.__init(self)
end

function Foveal:updateOutput(input)
   assert(input:nDimension() == 2)
   assert(input:size(2) == 5)
   local N = 4
   self.output:resize(input:size(1) * N, input:size(2))

   local cinput = input:float()
   local coutput = self.output:float()

   local output_split = coutput:split(N)

   local function createRegion(id,x,y,w,h)
      return torch.FloatTensor{id,x,y,x+w,y+h}
   end

   for i=1,input:size(1) do
      local box = cinput[i]
      local id,x,y,x2,y2 = table.unpack(box:totable())
      local w = x2 - x
      local h = y2 - y
      local base = output_split[i]
      base[1]:copy(box)
      base[2]:copy(createRegion(id,x-w*.25,y-h*.25,w*1.5,h*1.5))
      base[3]:copy(createRegion(id,x-w*0.5,y-h*0.5,w*2.0,h*2.0))
      base[4]:copy(createRegion(id,x-w*1.5,y-h*1.5,w*4.0,h*4.0))
   end

   self.output:copy(coutput)
   return self.output
end
