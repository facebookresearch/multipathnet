--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

require 'inn'
require 'cudnn'
require 'fbcoco'
local utils = paths.dofile'model_utils.lua'

local data = torch.load'data/models/imagenet_pretrained_alexnet.t7'

local model = nn.Sequential()
  :add(nn.ParallelTable()
    :add(utils.makeDataParallel(data.features:unpack()))
    :add(nn.Identity())
  )
  :add(inn.ROIPooling(6,6,1/16))
  :add(nn.View(-1):setNumInputDims(3))
  :add(data.top:unpack())
  :add(utils.classAndBBoxLinear(4096))

model:cuda()

utils.testModel(model)

return {model, utils.RossTransformer()}
