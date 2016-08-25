--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'nn'
require 'engines.Optim'

local tnt = require 'torchnet'
local argcheck = require 'argcheck'

local FBOptimEngine, SGDEngine = torch.class('tnt.FBOptimEngine', 'tnt.SGDEngine', tnt)

FBOptimEngine.__init = argcheck{
   {name="self", type="tnt.FBOptimEngine"},
   call =
      function(self)
         SGDEngine.__init(self)
      end
}

FBOptimEngine.train = argcheck{
   {name="self", type="tnt.FBOptimEngine"},
   {name="network", type="nn.Module"},
   {name="criterion", type="nn.Criterion"},
   {name="iterator", type="tnt.DatasetIterator"},
   {name="maxepoch", type="number", default=1000},
   {name="optimMethod", type="function"},
   {name="config", type="table", opt=true},
   call =
      function(self, network, criterion, iterator, maxepoch, optimMethod, config)
         local state = {
            network = network,
            criterion = criterion,
            iterator = iterator,
            maxepoch = maxepoch,
            optimMethod = optimMethod,
            optimizer = nn.Optim(network, config),
            config = config,
            sample = {},
            epoch = 0, -- epoch done so far
            t = 0, -- samples seen so far
            training = true
         }

         self.hooks("onStart", state)
         while state.epoch < state.maxepoch do
            state.network:training()

            self.hooks("onStartEpoch", state)
            for sample in state.iterator() do
               state.sample = sample
               self.hooks("onSample", state)

               state.network:forward(sample.input)
               self.hooks("onForward", state)
               state.criterion:forward(state.network.output, sample.target)
               self.hooks("onForwardCriterion", state)

               state.network:zeroGradParameters()
               if state.criterion.zeroGradParameters then
                  state.criterion:zeroGradParameters()
               end

               state.criterion:backward(state.network.output, sample.target)
               self.hooks("onBackwardCriterion", state)
               state.network:backward(sample.input, state.criterion.gradInput)
               self.hooks("onBackward", state)

               state.optimizer:updateParameters(state.optimMethod, criterion.output)
               state.t = state.t + 1
               self.hooks("onUpdate", state)
            end
            state.epoch = state.epoch + 1
            self.hooks("onEndEpoch", state)
         end
         self.hooks("onEnd", state)
      end
}
