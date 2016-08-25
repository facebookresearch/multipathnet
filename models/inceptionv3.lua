--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

-- fine-tuning of pretrained inception-v3 trained by google and converted to
-- torch using https://github.com/Moodstocks/inception-v3.torch

require 'cudnn'
require 'cunn'
require 'inn'
require 'fbcoco'
inn.utils = require 'inn.utils'
local utils = paths.dofile'model_utils.lua'

local net = torch.load'./data/models/inceptionv3.t7'

local input = torch.randn(1,3,299,299):cuda()
local output1 = net:forward(input):clone()
utils.BNtoFixed(net, true)
local output2 = net:forward(input):clone()
assert((output1 - output2):abs():max() < 1e-5)

local features = nn.Sequential()
local classifier = nn.Sequential()

for i=1,25 do features:add(net:get(i)) end
for i=26,30 do classifier:add(net:get(i)) end

utils.testSurgery(input, utils.disableFeatureBackprop, features, 16)
utils.testSurgery(input, inn.utils.foldBatchNorm, features:findModules'nn.NoBackprop'[1])

local model = nn.Sequential()
  :add(nn.ParallelTable()
    :add(utils.makeDataParallel(features))
    :add(nn.Identity())
  )
  :add(inn.ROIPooling(17,17):setSpatialScale(17/299))
  :add(utils.makeDataParallel(classifier))
  :add(utils.classAndBBoxLinear(2048))

model:cuda()
model.input_size = 299 -- for utils.testModel

utils.testModel(model)

return {model, fbcoco.ImageTransformer({1,1,1},nil,2)}
