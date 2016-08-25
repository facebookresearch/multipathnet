--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

require 'cudnn'
require 'cunn'
require 'fbcoco'
require 'xlua'
require 'inn'
inn.utils = require 'inn.utils'
local utils = paths.dofile'model_utils.lua'

local model_opt = {
   resnet_path = './data/models/resnet/resnet-18.t7'
}
model_opt = xlua.envparams(model_opt)
print(model_opt)
if opt then for k,v in pairs(model_opt) do opt[k] = v end end

local function loadResNet(model_path)
   local net = torch.load(model_path)
   net:cuda():evaluate()

   local features = nn.Sequential()
   for i=1,7 do features:add(net:get(i)) end

   local input = torch.randn(1,3,224,224):cuda()

   utils.testSurgery(input, utils.disableFeatureBackprop, features, 5)
   utils.testSurgery(input, inn.utils.foldBatchNorm, features:findModules'nn.NoBackprop'[1])
   utils.testSurgery(input, inn.utils.BNtoFixed, features, true)
   utils.testSurgery(input, inn.utils.BNtoFixed, net, true)

   local classifier = nn.Sequential()
   for i=8,10 do classifier:add(net:get(i)) end

   local output_dim = classifier.output:size(2)

   local model = nn.Sequential()
      :add(nn.ParallelTable()
         :add(utils.makeDataParallel(features))
         :add(nn.Identity())
      )
      :add(inn.ROIPooling(14,14,1/16))
      :add(utils.makeDataParallel(classifier))
      :add(utils.classAndBBoxLinear(output_dim))

   model:cuda()

   utils.testModel(model)

   return {model, utils.ImagenetTransformer()}
end

return loadResNet(model_opt.resnet_path)
