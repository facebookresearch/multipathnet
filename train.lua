--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

require 'torch'
require 'nn'
require 'optim'
require 'xlua'

local tnt = require 'torchnet'
require 'engines.fboptimengine'

require 'fbcoco'

local json = require 'cjson'
local utils = paths.dofile 'utils.lua'
local model_utils = paths.dofile 'models/model_utils.lua'

opt = {
   epoch = 1,
   dataset = 'pascal',
   train_set = 'trainval',
   test_set = 'test',
   model = 'alexnet',
   year = '2007',
   proposal_dir = 'data/proposals/',
   proposals = 'deepmask',
   images_per_batch = 2,
   scale = 600,
   max_size = 1000,
   learningRate = 1e-3,
   dampening = 0,
   weightDecay = 0.0005,
   momentum = 0.9,
   learningRateDecay = 0,
   nEpochs = 400,
   epochSize = 100,
   nDonkeys = 4,
   batchSize = 128,
   manualSeed = 555,
   step = 300,
   best_proposals_number = 1000,
   snapshot = 100,
   criterion = 'ce',
   decay = 0.1,
   bbox_regression = 1,
   retrain = 'no',
   train_min_gtroi_size = 0,
   train_remove_dropouts = false,
   retrain_mean_std = '',
   train_nGPU = 1,
   test_nGPU = 1,
   train_nsamples = -1, -- all samples
   test_nsamples = -1, -- all samples
   test_best_proposals_number = 500,
   disable_memory_efficient_forward=false,
   checkpoint=false,
   resume='',
   extra_proposals_file = '',
   method='sgd',
   sample_n_per_box = 0,
   sample_sigma = 1,
   train_min_proposal_size = 0,
   integral=false,
   imagenet_classes='',
   test_num_per_image=100,
   save_folder='',

   phase2_epoch=-1,
   phase2_learningRate=-1,
   phase2_step=-1,
   phase2_decay=-1,

   fg_threshold = -1,     -- if -1, then set to bg_threshold_max
   bg_threshold_min = 0.1,
   bg_threshold_max = 0.5,
}
opt = xlua.envparams(opt)

if opt.fg_threshold < 0 then
   opt.fg_threshold = opt.bg_threshold_max
end
if opt.manualSeed == -1 then --random
   opt.manualSeed = torch.random(10000)
end
print(opt)
model_opt = {}

require 'cutorch'
math.randomseed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)
torch.manualSeed(opt.manualSeed)

---------------------------------------------------------------------------------------
-- model
---------------------------------------------------------------------------------------
assert(opt.images_per_batch % opt.train_nGPU == 0, "images_per_batch must be a multiple of train_nGPU")
opt.num_classes = opt.dataset == 'pascal' and 21 or 81

local model_data = paths.dofile('models/'..opt.model..'.lua')
local model, transformer, info = table.unpack(model_data)

if opt.train_remove_dropouts then
   model_utils.removeDropouts(model)
end

-- serialize transformer for donkeys and to be loaded for testing
opt.transformer = paths.concat(opt.save_folder, 'transformer.t7')
torch.save(opt.transformer, transformer)

if opt.retrain ~= 'no' then
   print('Loading a retrain model:'..opt.retrain)
   model = torch.load(opt.retrain)
   transformer = torch.load(opt.transformer)
end

local getIterator = require 'data'
local iterator = getIterator()

local integral_switches
if opt.integral then
   if opt.retrain == 'no' then
      integral_switches = model_utils.integral(model)
   else
      local switch = model:findModules'nn.ModeSwitch'[1]
      integral_switches = switch:get(1):findModules'nn.SelectTable'
   end
end

model:cuda()

if not opt.bbox_mask_1d then
   model_utils.addBBoxNorm(model, g_mean_std)
end

model_utils.testModel(model)

-- set up testing
local test_year = (opt.year == '2007,2012') and '2007' or opt.year
local dataset_name = opt.dataset..'_'..opt.test_set..test_year
local test_folder_name = opt.dataset == 'pascal' and ('VOC'..test_year) or 'coco'
local test_proposals_path = utils.makeProposalPath(opt.proposal_dir, test_folder_name, opt.proposals, opt.test_set)

--------------------------------------------------------------------------
-- training
--------------------------------------------------------------------------

local samples = {}

local function createCriterion()
   criterion = nn.ParallelCriterion()
   :add(nn.CrossEntropyCriterion(), 1)
   :add(nn.BBoxRegressionCriterion(), opt.bbox_regression)
   return criterion:cuda()
end

local dataTimer = tnt.TimeMeter()
local timer, batchTimer = tnt.TimeMeter({ unit = true }), tnt.TimeMeter()
local trainLoss = tnt.AverageValueMeter()
local primary_loss = tnt.AverageValueMeter()
local bboxregr_loss = tnt.AverageValueMeter()


local engine = tnt.FBOptimEngine()


local function json_log(t) print('json_stats: '..json.encode(t)) end

-----------------------------------------------------------------------------

local function log(state, extra)
   local info = {
      epoch = state.epoch + 1,
      learningRate = state.learningRate,
      decay = state.decay,
      train_time = timer.timer:time().real,
      train_loss = trainLoss:value(),
      primary_loss = primary_loss:value(),
      bboxregr_loss = bboxregr_loss:value(),
   }
   json_log(utils.merge_table{opt, model_opt, extra, info})
end

local function save(model, state, epoch)
   opt.test_model = 'model_'..epoch..'.t7'
   opt.test_state = 'optimState_'..epoch..'.t7'
   local model_path = paths.concat(opt.save_folder, opt.test_model)
   local state_path = paths.concat(opt.save_folder, opt.test_state)

   print("Saving model to "..model_path)
   torch.save(model_path, utils.checkpoint(model))
   print("Saving state to "..state_path)
   torch.save(state_path, state)
end

local function validate(model)
   if opt.test_nGPU > 1 then
      print("test_nGPU > 1, running tester in separate threads")
      local test_runner = paths.dofile'test_runner.lua'
      test_runner:setup(opt.test_nGPU, dataset_name, test_proposals_path)
      local res = test_runner:test()
      test_runner = nil
      tester = nil -- global var
      return res
   else
      print("test_nGPU == 1, running tester in main thread")
      model:evaluate()
      local ds = paths.dofile'DataSetJSON.lua':create(dataset_name, test_proposals_path, opt.test_nsamples)
      ds:loadROIDB(opt.test_best_proposals_number)
      local tester = fbcoco.Tester_FRCNN(model,transformer,ds,{opt.scale}, opt.max_size, opt)
      local res = tester:test()
      model:training()
      return res
   end
end

engine.hooks.onStart = function(state)
   state.learningRate = opt.learningRate
   state.decay = opt.decay
   state.step = opt.step
   utils.cleanupOptim(state)
   if opt.checkpoint then
      local filename = checkpoint.resume(state)
      if filename then
         print("WARNING: restarted from checkpoint:", filename)
      elseif opt.resume ~= '' then
         print("resuming from checkpoint:", opt.resume)
         checkpoint.apply(state, opt.resume)
      end
   end
end

engine.hooks.onStartEpoch = function(state)
   local epoch = state.epoch + 1
   if epoch == opt.phase2_epoch then
      print("switching to phase 2")
      if state.network.setPhase2 then
         state.network:setPhase2()
      end
      if opt.phase2_learningRate >= 0 then
         print("setting learning rate to " .. opt.phase2_learningRate)
         state.learningRate = opt.phase2_learningRate

         local optimizer = state.optimizer
         for k,v in pairs(optimizer.modulesToOptState) do if v[1] then
            for i,u in ipairs(v) do
               if u.dfdx then
                  local curdev = cutorch.getDevice()
                  cutorch.setDevice(u.dfdx:getDevice())
                  u.dfdx:zero()
                  cutorch.setDevice(curdev)
                  u.learningRate = state.learningRate
               end
            end
         end end
      end
      if opt.phase2_step >= 0 then
         print("setting step to " .. opt.phase2_step)
         state.step = opt.phase2_step
      end
      if opt.phase2_decay >= 0 then
         print("setting decay to " .. opt.phase2_decay)
         state.decay = opt.phase2_decay
      end
   end

   if opt.checkpoint and epoch % opt.snapshot == 0 then
      checkpoint.checkpoint(state, opt)
   end
   print("Training epoch " .. epoch .. "/" .. opt.nEpochs)
   trainLoss:reset()
   primary_loss:reset()
   bboxregr_loss:reset()
   timer:reset()
   state.n = 0
end

engine.hooks.onSample = function(state)
   cutorch.synchronize(); collectgarbage();
   dataTimer:stop()

   utils.recursiveCast(samples, state.sample, 'torch.CudaTensor')

   if opt.integral then
      assert(samples[2][3])
      for i,v in ipairs(integral_switches) do
         v.index = samples[2][3]
         v.gradInput = {}
      end
   end

   state.sample.input = samples[1]
   state.sample.target = samples[2]
end

engine.hooks.onUpdate = function(state)
   cutorch.synchronize(); collectgarbage();
   state.n = state.n + 1

   local err = state.criterion.output
   trainLoss:add(err)
   primary_loss:add(state.criterion.criterions[1].output)
   bboxregr_loss:add(state.criterion.criterions[2].output)

   timer:incUnit()

   print(('Epoch: [%d][%d/%d]\tTime %.3f (%.3f) DataTime %.3f Err %.4f'):format(
   state.epoch + 1, state.n, opt.epochSize, batchTimer:value(), timer:value(), dataTimer:value(), err))

   dataTimer:reset()
   dataTimer:resume()
   batchTimer:reset()
end

engine.hooks.onEndEpoch = function(state)
   local epoch = state.epoch + 1
   if epoch % state.step == 0 then
      print('Dropping learning rate')
      state.learningRate = state.learningRate * state.decay
      local optimizer = state.optimizer
      for k,v in pairs(optimizer.modulesToOptState) do if v[1] then
         for i,u in ipairs(v) do
            if u.dfdx then
               local curdev = cutorch.getDevice()
               cutorch.setDevice(u.dfdx:getDevice())
               u.dfdx:mul(state.decay)
               cutorch.setDevice(curdev)
               u.learningRate = u.learningRate * state.decay
            end
         end
      end end
   end
   log(state, {finished = 0, voc_metric = 0, coco_metric = 0})
   if epoch % opt.snapshot == 0 then
      save(state.network, state.optimizer, epoch)
      local res = validate(state.network)
      log(state, {
         voc_metric = res[2],
         coco_metric = res[1],
      })
   end
end

engine.hooks.onEnd = function(state)
   print("Done training. Running final validation")

   save(state.network, state.optimizer, 'final')

   opt.test_nsamples = 4952

   local res = validate(state.network)
   log(state, {
      voc_metric = res[2],
      coco_metric = res[1],
   })
end


engine:train{
   network = model,
   criterion = createCriterion(),
   config = opt,
   maxepoch = opt.nEpochs,
   optimMethod = optim[opt.method],
   iterator = iterator,
}
