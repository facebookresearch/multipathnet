--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

require 'xlua'
require 'inn'
require 'cudnn'
require 'fbcoco'
local utils = paths.dofile'model_utils.lua'

local model_opt = xlua.envparams{
  model_conv345_norm = true,
  model_het = true,
  model_foveal_exclude = -1,
}

print("model_opt")
print(model_opt)

local N = 4

local data = torch.load'data/models/imagenet_pretrained_vgg.t7'
local features = utils.safe_unpack(data.features)
local classifier = utils.safe_unpack(data.top)

local model = nn.Sequential()

for k,v in ipairs(classifier:findModules'nn.Dropout') do v.inplace = true end

local skip_features = nn.Sequential()
for i = 1, 16 do
  skip_features:add(features:get(i))
end
local conv4 = nn.Sequential()
for i = 17, 23 do
  conv4:add(features:get(i))
end

local conv5 = nn.Sequential()
for i = 24, 30 do
  conv5:add(features:get(i))
end

skip_features:add(nn.ConcatTable()
  :add(conv4)
  :add(nn.Identity()))

skip_features:add(nn.ParallelTable()
  :add(nn.ConcatTable()
    :add(conv5)
    :add(nn.Identity()))
  :add(nn.Identity()))

skip_features:add(nn.FlattenTable())

model:add(nn.ParallelTable()
  :add(nn.NoBackprop(utils.makeDataParallel(skip_features)))
  :add(nn.Identity()))

model:add(nn.ParallelTable()
  :add(nn.Identity())
  :add(nn.Sequential()
    :add(nn.Foveal())
    :add(nn.View(-1,N,5))
    :add(nn.Transpose({1,2}))))

-- local towers = nn.ConcatTable()

local nGPU = opt and opt.train_nGPU or 4
local regions = nn.ModelParallelTable(2)
local oldDev = cutorch.getDevice()
local dev = 1
local Nreg = 0
for i=1,N do
  -- local dev = i % nGPU
  -- dev = (dev==0) and nGPU or dev
  if i ~= model_opt.model_foveal_exclude then
    cutorch.setDevice(dev)
    print('dev', i, dev)

    local region_instance = nn.Sequential()
    region_instance:add(nn.ParallelTable():add(nn.Identity()):add(nn.Select(1,i)))
    region_instance:add(utils.conv345Combine(
        model_opt.model_conv345_norm, i == 1, i <= 3, not model_opt.model_conv345_norm))
    region_instance:add(classifier:clone())
    region_instance:float():cuda()
    regions:add(region_instance, dev)

    dev = dev + 1
    dev = (dev > nGPU) and 1 or dev
    Nreg = Nreg +1
  end
end

if model_opt.model_het then
  -- ooh, doing something weird here to avoid OOM
  cutorch.setDevice(nGPU)
  local region_instance = nn.Sequential()
  region_instance:add(nn.ParallelTable():add(nn.Identity()):add(nn.Select(1,2)))
  region_instance:add(utils.conv345Combine(
    model_opt.model_conv345_norm, true, true, not model_opt.model_conv345_norm))

  region_instance:add(classifier:clone())
  region_instance:float():cuda()

  regions:add(region_instance, nGPU)
end
cutorch.setDevice(oldDev)
model:add(regions)

if model_opt.model_het then
  model:add(nn.ConcatTable():add(nn.Narrow(2, 1, Nreg*4096)):add(nn.Narrow(2, Nreg*4096+1, 4096)))
  model:add(utils.classAndBBoxLinear(Nreg*4096, 4096))
else
  model:add(utils.classAndBBoxLinear(Nreg*4096))
end
model:cuda()

model.phase = 1
model.setPhase2 = utils.vggSetPhase2_outer

utils.testModel(model)

return {model, utils.RossTransformer()}
