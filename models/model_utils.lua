--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local generateGraph = require 'optnet.graphgen'
local iterm = require 'iterm'
require 'iterm.dot'

local utils = {}

function utils.makeDataParallel(module, nGPU)
  nGPU = nGPU or ((opt and opt.train_nGPU) or 1)
  if nGPU > 1 then
    local dpt = nn.DataParallelTable(1) -- true?
    local cur_dev = cutorch.getDevice()
    for i = 1, nGPU do
      cutorch.setDevice(i)
      dpt:add(module:clone():cuda(), i)
    end
    cutorch.setDevice(cur_dev)
    return dpt
  else
    return nn.Sequential():add(module)
  end
end

function utils.makeDPParallelTable(module, nGPU)
  if nGPU > 1 then
    local dpt = nn.DPParallelTable()
    local cur_dev = cutorch.getDevice()
    for i = 1, nGPU do
      cutorch.setDevice(i)
      dpt:add(module:clone():cuda(), i)
    end
    cutorch.setDevice(cur_dev)
    return dpt
  else
    return nn.ParallelTable():add(module)
  end
end

-- returns a new Linear layer with less output neurons
function utils.compress(layer, n)
    local W = layer.weight
    local U,S,V = torch.svd(W:t():float())
    local new = nn.Linear(W:size(2), n):cuda()
    new.weight:t():copy(U:narrow(2,1,n) * torch.diag(S:narrow(1,1,n)) * V:narrow(1,1,n):narrow(2,1,n))
    new.bias:zero()
    return new
end

-- returns a Sequential of 2 Linear layers, one biasless with U*diag(S) and one
-- with V and original bias. L is the number of components to keep.
function utils.SVDlinear(layer, L)
  local W = layer.weight:double()
  local b = layer.bias:double()

  local K, N = W:size(1), W:size(2)

  local U, S, V = torch.svd(W:t(), 'A')

  local US = U:narrow(2,1,L) * torch.diag(S:narrow(1,1,L))
  local Vt = V:narrow(2,1,L)

  local L1 = nn.LinearNB(N, L)
  L1.weight:copy(US:t())

  local L2 = nn.Linear(L, K)
  L2.weight:copy(Vt)
  L2.bias:copy(b)

  return nn.Sequential():add(L1):add(L2)
end


function utils.testSurgery(input, f, net, ...)
   local output1 = net:forward(input):clone()
   f(net,...)
   local output2 = net:forward(input):clone()
   print((output1 - output2):abs():max())
   assert((output1 - output2):abs():max() < 1e-5)
end


function utils.removeDropouts(net)
   net:replace(function(x)
      return torch.typename(x):find'nn.Dropout' and nn.Identity() or x
   end)
end


function utils.disableFeatureBackprop(features, maxLayer)
  local noBackpropModules = nn.Sequential()
  for i = 1,maxLayer do
    noBackpropModules:add(features.modules[1])
    features:remove(1)
  end
  features:insert(nn.NoBackprop(noBackpropModules):cuda(), 1)
end

function utils.classAndBBoxLinear(N, N2)
  local class_linear = nn.Linear(N,opt and opt.num_classes or 21):cuda()
  class_linear.weight:normal(0,0.01)
  class_linear.bias:zero()

  local bbox_linear = nn.Linear(N2 or N,(opt and opt.num_classes or 21) * 4):cuda()
  bbox_linear.weight:normal(0,0.001)
  bbox_linear.bias:zero()

  if N2 then
    return nn.ParallelTable():add(class_linear):add(bbox_linear):cuda()
  else
    return nn.ConcatTable():add(class_linear):add(bbox_linear):cuda()
  end
end

function utils.testModel(model)
  input_size = model.input_size or 224
  print(model)
  model:training()
  local batchSz = opt and opt.images_per_batch or 2
  local boxes = torch.Tensor(batchSz, 5)
  for i = 1, batchSz do
    boxes[i]:copy(torch.Tensor({i,1,1,100,100}))
  end
  local input = {torch.CudaTensor(batchSz,3,input_size,input_size),boxes:cuda()}
  local output = model:forward(input)
  -- iterm.dot(generateGraph(model, input), opt and opt.save_folder..'/graph.pdf' or paths.tmpname()..'.pdf')
  print{output}
  print{model:backward(input,output)}
end

-- used in AlexNet and VGG models trained by Ross
function utils.RossTransformer()
  return fbcoco.ImageTransformer({102.9801,115.9465,122.7717}, nil, 255, {3,2,1})
end

-- used in ResNet and facebook inceptions
function utils.ImagenetTransformer()
  return fbcoco.ImageTransformer(
  { -- mean
    0.48462227599918,
    0.45624044862054,
    0.40588363755159,
  },
  { -- std
    0.22889466674951,
    0.22446679341259,
    0.22495548344775,
  })
end

function utils.normalizeBBoxRegr(model, meanstd)
  if #model:findModules('nn.BBoxNorm') == 0 then
    -- normalize the bbox regression
    local regression_layer = model:get(#model.modules):get(2)
    if torch.type(regression_layer) == 'nn.Sequential' then
      regression_layer = regression_layer:get(#regression_layer.modules)
    end
    assert(torch.type(regression_layer) == 'nn.Linear')

    local mean_hat = torch.repeatTensor(meanstd.mean,1,opt.num_classes):cuda()
    local sigma_hat = torch.repeatTensor(meanstd.std,1,opt.num_classes):cuda()

    regression_layer.weight:cdiv(sigma_hat:t():expandAs(regression_layer.weight))
    regression_layer.bias:add(-mean_hat):cdiv(sigma_hat)

    utils.addBBoxNorm(model, meanstd)
  end
end

function utils.addBBoxNorm(model, meanstd)
  if #model:findModules('nn.BBoxNorm') == 0 then
    model:add(nn.ParallelTable()
      :add(nn.Identity())
      :add(nn.BBoxNorm(meanstd.mean, meanstd.std)):cuda())
  end
end

function utils.vggSetPhase2(model)
  assert(model.phase == 1)
  local dpt = model.modules[1].modules[1]
  for i = 1, #dpt.modules do
    assert(torch.type(dpt.modules[i]) == 'nn.NoBackprop')
    dpt.modules[i] = dpt.modules[i].modules[1]
    utils.disableFeatureBackprop(dpt.modules[i], 10)
  end
  model.phase = phase
  print("Switched model to phase 2")
  print(model)
end

function utils.vggSetPhase2_outer(model)
  assert(model.phase == 1)
  model.modules[1].modules[1] = model.modules[1].modules[1].modules[1]
  local dpt = model.modules[1].modules[1]
  for i = 1, #dpt.modules do
    utils.disableFeatureBackprop(dpt.modules[i], 10)
  end
  model.phase = phase
  print("Switched model to phase 2")
  print(model)
end

function utils.conv345Combine(isNormalized, useConv3, useConv4, initCopyConv5)
  local totalFeat = 0

  local function make1PoolingLayer(idx, nFeat, spatialScale, normFactor)
    local pool1 = nn.Sequential()
      :add(nn.ParallelTable():add(nn.SelectTable(idx)):add(nn.Identity()))
      :add(inn.ROIPooling(7,7,spatialScale))
    if isNormalized then
      pool1:add(nn.View(-1, nFeat*7*7))
        :add(nn.Normalize(2))
        :add(nn.Contiguous())
        :add(nn.View(-1, nFeat, 7, 7))
    else
      pool1:add(nn.MulConstant(normFactor))
    end
    totalFeat = totalFeat + nFeat
    return pool1
  end

  local pooling_layer = nn.ConcatTable()
  pooling_layer:add(make1PoolingLayer(1, 512, 1/16, 1)) -- conv5
  if useConv4 then
    pooling_layer:add(make1PoolingLayer(2, 512, 1/8, 1/30)) -- conv4
  end
  if useConv3 then
    pooling_layer:add(make1PoolingLayer(3, 256, 1/4, 1/200)) -- conv3
  end
  local pooling_join = nn.Sequential()
    :add(pooling_layer)
    :add(nn.JoinTable(2))
  if isNormalized then
    pooling_join:add(nn.MulConstant(1000))
  end
  local conv_mix = cudnn.SpatialConvolution(totalFeat, 512, 1, 1, 1, 1)
  if initCopyConv5 then
    conv_mix.weight:zero()
    conv_mix.weight:narrow(2, 1, 512):copy(torch.eye(512)) -- initialize to just copy conv5
  end
  pooling_join:add(conv_mix)
  pooling_join:add(nn.View(-1):setNumInputDims(3))

  return pooling_join
end

function utils.load(path)
  local data = torch.load(path)
  return data.unpack and data:unpack() or data
end

-- takes a model, removes last classification layer and inserts integral loss
function utils.integral(model)
  local top_cat = model:get(#model.modules)
  model:remove(#model.modules)

  assert(torch.type(top_cat) == 'nn.ConcatTable' or
         torch.type(top_cat) == 'nn.ParallelTable')
  local is_parallel = torch.type(top_cat) == 'nn.ParallelTable'

  local new_cl = nn.ConcatTable()
  for i=1,opt.nDonkeys do
    new_cl:add(top_cat:get(1):clone())
  end
  local new_top = is_parallel and nn.ParallelTable() or nn.ConcatTable()
  new_top:add(new_cl):add(top_cat:get(2))
  model:add(new_top)

  integral_selector = nn.SelectTable(1)
  local train_branch = nn.ParallelTable()
    :add(integral_selector)
    :add(nn.Identity())

  local softmaxes = nn.ParallelTable()
  for i=1,opt.nDonkeys do
    softmaxes:add(
      nn.Sequential()
        :add(nn.SoftMax())
        :add(nn.View(1,-1,opt.num_classes))
    )
  end

  local eval_branch = nn.Sequential()
    :add(nn.ParallelTable()
      :add(nn.Sequential()
        :add(softmaxes)
        :add(nn.JoinTable(1))
        :add(nn.Mean(1))
      )
      :add(nn.Identity())
    )
  model.noSoftMax = true
  model:add(nn.ModeSwitch(train_branch, eval_branch))
  return {integral_selector}
end

return utils
