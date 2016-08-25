--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local ModeSwitch, parent = torch.class('nn.ModeSwitch','nn.Container')

function ModeSwitch:__init(train_module, test_module)
   self.train = true
   self.modules = {train_module, test_module}
end

function ModeSwitch:updateOutput(input)
   local active = self.train and self.modules[1] or self.modules[2]
   self.output = active:updateOutput(input)
   return self.output
end

function ModeSwitch:updateGradInput(input, gradOutput)
   if self.train then
      self.gradInput = self.modules[1]:updateGradInput(input, gradOutput)
   else
      error'backprop not defined in evaluate mode'
   end
   return self.gradInput
end

function ModeSwitch:accGradParameters(input, gradOutput)
end

function ModeSwitch:__tostring__()
   return nn.ParallelTable.__tostring__(self)
end
