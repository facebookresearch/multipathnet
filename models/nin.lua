--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

-- Fast Network-In-Network model from https://gist.github.com/szagoruyko/0f5b4c5e2d2b18472854

require 'inn'
require 'cudnn'
require 'fbcoco'
inn.utils = require 'inn.utils'
local utils = paths.dofile'model_utils.lua'

local net = utils.load'./data/models/model_bn_final.t7'
net:cuda():evaluate()
cudnn.convert(net, cudnn)

local input = torch.randn(1,3,224,224):cuda()

utils.testSurgery(input, utils.BNtoFixed, net, true)

local features = nn.Sequential()
local classifier = nn.Sequential()

for i=1,29 do features:add(net:get(i)) end
for i=31,40 do classifier:add(net:get(i)) end
classifier:add(nn.View(-1):setNumInputDims(3))

utils.testSurgery(input, utils.disableFeatureBackprop, features, 10)
utils.testSurgery(input, inn.utils.foldBatchNorm, features:findModules'nn.NoBackprop'[1])
utils.testSurgery(input, utils.BNtoFixed, features, true)
utils.testSurgery(input, utils.BNtoFixed, net, true)

local model = nn.Sequential()
  :add(nn.ParallelTable()
    :add(utils.makeDataParallel(features))
    :add(nn.Identity())
  )
  :add(inn.ROIPooling(7,7,1/16))
  :add(classifier)
  :add(utils.classAndBBoxLinear(1024))

model:cuda()

utils.testModel(model)

return {model, utils.ImagenetTransformer()}
