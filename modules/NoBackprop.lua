--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local NoBackprop,parent = torch.class('nn.NoBackprop','nn.Container')

-- was lazy to finish CPU side
function NoBackprop:__init(inner)
   parent.__init(self)
   assert(inner)
   self.modules = {inner}
end

function NoBackprop:updateOutput(input)
   self.output = self.modules[1]:updateOutput(input)
   return self.output
end

function NoBackprop:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   return self.gradInput
end

function NoBackprop:__tostring()
   return 'NoBackprop: ' .. tostring(self.modules[1])
end

-- ugh, stupid temporary backwards-compatibility hack
NoBackprop.__version = 2
function NoBackprop:__read(file, version)
   -- do the normal read
   local var = file:readObject()
   for k, v in pairs(var) do
      self[k] = v
   end
   -- fixup module
   if version < 2 then
      self.modules = {self.inner}
      self.inner = nil
   end
end
