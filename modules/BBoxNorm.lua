--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local BBoxNorm, parent = torch.class('nn.BBoxNorm','nn.Module')

function BBoxNorm:__init(mean, std)
   assert(mean and std)
   parent.__init(self)
   self.mean = mean
   self.std = std
end

function BBoxNorm:updateOutput(input)
   assert(input:dim() == 2 and input:size(2) % 4 == 0)
   self.output:set(input)
   if not self.train then
      if not input:isContiguous() then
         self._output = self._output or input.new()
         self._output:resizeAs(input):copy(input)
         self.output = self._output
      end

      local output = self.output:view(-1, 4)
      output:cmul(self.std:expandAs(output)):add(self.mean:expandAs(output))
   end
   return self.output
end

function BBoxNorm:updateGradInput(input, gradOutput)
   assert(self.train, 'cannot updateGradInput in evaluate mode')
   self.gradInput = gradOutput
   return self.gradInput
end

function BBoxNorm:clearState()
   nn.utils.clear(self, '_output')
   return parent.clearState(self)
end
