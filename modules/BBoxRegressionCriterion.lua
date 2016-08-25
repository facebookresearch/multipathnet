--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local BBoxRegressionCriterion, parent = torch.class('nn.BBoxRegressionCriterion', 'nn.SmoothL1Criterion')

function BBoxRegressionCriterion:updateOutput(inputs, targets)
   local target_classes = targets[1] -- B
   local target_boxes = targets[2]   -- Bx84
   -- inputs : Bx84

   self.sizeAverage = false

   target_classes = torch.type(target_classes) == 'torch.CudaTensor' and target_classes or target_classes:long()

   local B = inputs:size(1)
   local N = target_boxes:size(2)/4

   self._buffer1 = self._buffer1 or inputs.new()
   self._buffer2 = self._buffer2 or inputs.new()
   self._buffer1:resize(B,N):zero()
   self._buffer1:scatter(2,target_classes:view(B,1),1)
   self._buffer2:resizeAs(inputs):copy(self._buffer1:view(B,N,1):expand(B,N,4))
   self._buffer2:narrow(2,1,4):zero()
   self._buffer2:cmul(inputs)

   parent.updateOutput(self, self._buffer2, target_boxes)
   self.output = self.output / B
   return self.output
end

function BBoxRegressionCriterion:updateGradInput(inputs, targets)
   local B = inputs:size(1)
   parent.updateGradInput(self, self._buffer2, targets[2])
   return self.gradInput:div(B)
end
