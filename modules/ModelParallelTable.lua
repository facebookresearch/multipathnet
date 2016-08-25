--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local gpuLocalCopyBuffers = {}
local baseModuleIndex = 1  -- A constant

-- *****************************************************************************
-- Helper Functions
-- *****************************************************************************
-- queryGPUDeviceId - Function to query a tensor or table for the
-- GPUID.  For tables we will search the table for CudaTensors, query their
-- device and make sure the deviceIds of ALL CudaTensors are on the same GPU.
local function queryGPUDeviceId(object)
   if torch.type(object) == 'torch.CudaTensor' then
      return object:getDevice()
   end

   local deviceId

   -- Try finding a parameter
   local stack = {}  -- explicit stack to recurse on tables
   for key, param in pairs(object) do
      if key ~= 'modules' then
         stack[#stack+1] = param  -- Push onto the stack
      end
   end
   while #stack > 0 do
      local param = stack[#stack]; stack[#stack] = nil  -- Pop the stack
      if (torch.type(param) == 'table') then
         for i = 1, #param do stack[#stack+1] = param[i] end  -- Push onto stack
      elseif (torch.type(param) == 'torch.CudaTensor') then
         if (torch.numel(param) > 0) then
            -- Empty tensors are always on GPU "0"
            local curId = param:getDevice()
            if deviceId == nil then
               deviceId = curId
            else
               assert(deviceId == curId,
               'Found CudaTensor instances from different devices')
            end
         end
      end
   end

   return deviceId
end

-- Get an available GPU buffer for asyncGPUCopy.  It is used when the GPU tensor
-- is not contiguous.
local function getBuffer()
   local device = cutorch.getDevice()
   if not gpuLocalCopyBuffers[device] then
      gpuLocalCopyBuffers[device] = torch.CudaTensor()
   end
   return gpuLocalCopyBuffers[device]
end

-- setDeviceSafe - Avoid redundant calls to setDevice
local function setDevice(gpuid)
   if (cutorch.getDevice() ~= gpuid) then
      cutorch.setDevice(gpuid)
   end
end

local function equalSize(sizeTable1, sizeTable2)
   if (#sizeTable1 ~= #sizeTable2) then
      return false
   end
   for i = 1, #sizeTable1 do
      if sizeTable1[i] ~= sizeTable2[i] then return false end
   end
   return true
end

local function equalSize(sizeTable1, sizeTable2)
   if (#sizeTable1 ~= #sizeTable2) then
      return false
   end
   for i = 1, #sizeTable1 do
      if sizeTable1[i] ~= sizeTable2[i] then return false end
   end
   return true
end

-- deepTensorsCopy - perform an elementwise copy of the tensors in the nested
-- table. We assume that the tables are properly initialized (ie same size and
-- structure), although we will assert it.
local function deepTensorsCopy(dst, src)
   if (torch.type(src) == 'table') then
      assert(torch.type(dst) == 'table' and #src == #dst)
      for i = 1, #src do deepTensorsCopy(dst[i], src[i]) end
   elseif torch.type(src):find('torch%..+Tensor') then
      assert(torch.type(dst):find('torch%..+Tensor'))
      assert(dst:isSameSizeAs(src))
      dst:copy(src)
   else
      error('input must be a nested table of tensors!')
   end
end

-- deepTensorsAdd - perform an elementwise add of the tensors in the nested
-- table. We assume that the tables are properly initialized (ie same size and
-- structure), although we will assert it.
--
-- Note: this is necessary because add() will malloc new memory on the cuda
-- driver side every time we want to get new memory!  Therefore, we actually
-- need to copy src to the dst gpu
local function deepTensorsAdd(dst, src)
   if (torch.type(src) == 'table') then
      assert(torch.type(dst) == 'table' and #src == #dst)
      for i = 1, #src do deepTensorsAdd(dst[i], src[i]) end
   elseif torch.type(src):find('torch%..+Tensor') then
      assert(torch.type(dst):find('torch%..+Tensor'))
      assert(dst:isSameSizeAs(src))

      local dstGpuid = dst:getDevice()
      local srcGpuid = src:getDevice()
      local curGpuid = cutorch:getDevice()
      setDevice(dstGpuid)

      -- Copy src over to a buffer on the dst GPU
      local srcBufferOnDstGpu = src
      if (dstGpuid ~= srcGpuid) then
         srcBufferOnDstGpu = getBuffer()
         srcBufferOnDstGpu:resizeAs(src)
         assert(src:isContiguous())
         srcBufferOnDstGpu:copy(src)
      end

      -- Perform the actual add
      dst:add(srcBufferOnDstGpu)
      if (dstGpuid ~= srcGpuid) then
         -- Ensures we get to keep the buffer for the duration of the add
         cutorch.synchronize()
      end

      setDevice(curGpuid)  -- Put the GPU id back to what it was
   else
      error('input must be a nested table of tensors!')
   end
end

-- *****************************************************************************
-- ModelParallelTable
-- *****************************************************************************
local ModelParallelTable, parent = torch.class('nn.ModelParallelTable',
'nn.Container')

function ModelParallelTable:__init(dimension, noGradInput)
   parent.__init(self)
   if not dimension then
      error "must specify a dimension!"
   end

   self.dimension = dimension
   self.modules = {}
   self.gpuAssignments = {}  -- Which gpuid each module sits on
   self.inputGpu = {}  -- inputs for each gpu
   self.gradOutputGpu = {} -- gradOutputs for each gpu
   self.outputGpu = {} -- outputs for each gpu
   self.gradInputGpu = {} -- gradInput for each gpu
   self.gradInputAddBuffer  = {}
   self.noGradInput = noGradInput or false
end

-- NOTE: The input should be on the FIRST added GPU device, and similarly the
-- output will be on the FIRST GPU device.
function ModelParallelTable:add(module, gpuid)
   local parameters = module:parameters()
   for _, param in ipairs(parameters) do
     assert(param:getDevice() == gpuid, param:getDevice() .. "~=" .. gpuid)
   end
   assert(gpuid <= cutorch.getDeviceCount() and gpuid >= 1)
   assert(#self.modules == #self.gpuAssignments)

   self.modules[#self.modules + 1] = module
   self.gpuAssignments[#self.gpuAssignments + 1] = gpuid

   return self
end

function ModelParallelTable:__tostring()
   return 'ModelParallelTable: ' .. #self.modules .. ' x ' .. tostring(self.modules[1])
end

function ModelParallelTable:get(index)
   return self.modules[index]
end

function ModelParallelTable:updateOutput(input)
   local baseGpuid = self.gpuAssignments[baseModuleIndex]
   -- cutorch.withDevice(baseGpuid, function() print('input', input[1]:mean()) end)
   assert(queryGPUDeviceId(input) == baseGpuid, 'Input is not on gpu ' ..
   baseGpuid)

   local prevGpuid = cutorch.getDevice()

   -- distribute the input to GPUs
   for i = 1, #self.modules do
      local gpuid = self.gpuAssignments[i]
      -- Copy the tensors in the input nested table to the GPU with gpuid
      self.inputGpu[i] = self:_copyTensorRecursive(
         input, self.inputGpu[i],
         baseGpuid, gpuid
      )
     -- cutorch.withDevice(gpuid, function() print('inputGpu', i, self.inputGpu[gpuid][1]:mean()) end)
   end

   cutorch.synchronize()

   -- update output for each module asynchronously
   for i, module in ipairs(self.modules) do
      local gpuid = self.gpuAssignments[i]
      setDevice(gpuid)
      self.outputGpu[i] = module:updateOutput(self.inputGpu[i])
      -- cutorch.withDevice(gpuid, function() print('outputGpu', i, self.outputGpu[gpuid]:mean()) end)
   end

   cutorch.synchronize()

   -- concatenate the outputs to the base GPU
   for i = 1, #self.modules do
      local gpuid = self.gpuAssignments[i]
      -- Merge the tensors in the input nested table to the GPU with gpuid
      self.output = self:_concatTensorRecursive(
         self.outputGpu[i], self.output,
         gpuid, i, baseGpuid, baseModuleIndex,
         #self.modules
      )
   end
   cutorch.synchronize()

   setDevice(prevGpuid)

   -- cutorch.withDevice(baseGpuid, function() print('output', self.output:mean()) end)
   return self.output
end

function ModelParallelTable:updateGradInput(input, gradOutput)
   -- We assume that updateOutput has already been called (therefore inputGpu
   -- has been populated)
   local baseGpuid = self.gpuAssignments[baseModuleIndex]
   -- cutorch.withDevice(baseGpuid, function() print('gradOutput', gradOutput:mean()) end)
   assert(queryGPUDeviceId(gradOutput) == baseGpuid,
   'gradOutput is not on gpu ' .. baseGpuid)

   local prevGpuid = cutorch.getDevice()

   -- distribute the gradOutput to GPUs
   for i = 1, #self.modules do
      local gpuid = self.gpuAssignments[i]
      -- Split the tensors in the input nested table to the GPU with gpuid
      -- _distributeTensorRecursive(src,dst,srcGpuid,srcInd,dstGpuid,dstInd)
      self.gradOutputGpu[i] = self:_distributeTensorRecursive(gradOutput,
         self.gradOutputGpu[i], baseGpuid, baseGpuIndex, gpuid, i, #self.modules)
   end

   cutorch.synchronize()

   -- update gradInput for each module asynchronously
   for i, module in ipairs(self.modules) do
      local gpuid = self.gpuAssignments[i]
      setDevice(gpuid)
      self.gradInputGpu[i] = module:updateGradInput(self.inputGpu[i],
      self.gradOutputGpu[i])
   end

   -- concatenate the outputs to the base GPU
   for i = 1, #self.modules do
      local gpuid = self.gpuAssignments[i]
      -- Merge the tensors in the input nested table to the GPU with gpuid
      self.gradInputAddBuffer[i] = self:_copyTensorRecursive(self.gradInputGpu[i],
         self.gradInputAddBuffer[i], gpuid, baseGpuid)
   end

   cutorch.synchronize()
   setDevice(baseGpuid)
   self.gradInput = self:_zeroTensorRecursive(self.gradInputGpu[baseGpuid], self.gradInput)
   for i = 1, #self.modules do
     self:_accumulateTensorRecursive(self.gradInputAddBuffer[i], self.gradInput)
   end

   setDevice(prevGpuid)

   return self.gradInput
end

function ModelParallelTable:accGradParameters(input, gradOutput, scale)
   -- We assume updateGradInput has already been called (so gradOutput has
   -- already been populated)
   local prevGpuid = cutorch.getDevice()
   local baseGpuid = self.gpuAssignments[baseModuleIndex]

   scale = scale or 1
   -- Calculate the gradWeight + gradBias on each sub-module
   for i, module in ipairs(self.modules) do
      local gpuid = self.gpuAssignments[i]
      setDevice(gpuid)
      module:accGradParameters(self.inputGpu[i], self.gradOutputGpu[i],
          scale)
   end

   cutorch.synchronize()  -- We have to wait until accGradParameters has finished

   setDevice(prevGpuid)
end

function ModelParallelTable:accUpdateGradParameters(input, gradOutput, lr)
   error("accUpdateGradParameters not supported for ModelParallelTable.")
end

function ModelParallelTable:zeroGradParameters()
   local prevGpuid = cutorch.getDevice()
   for i, module in ipairs(self.modules) do
      setDevice(self.gpuAssignments[i])
      module:zeroGradParameters()
   end
   setDevice(prevGpuid)
end

function ModelParallelTable:updateParameters(learningRate)
   error("updateParameters not supported for ModelParallelTable.")
end

function ModelParallelTable:share(mlp,...)
   error("Share not supported for ModelParallelTable.")
end

function ModelParallelTable:clone()
   error("clone not supported for ModelParallelTable.")
end

function ModelParallelTable:reset(stdv)
   local prevGpuid = cutorch.getDevice()
   for i, module in ipairs(self.modules) do
      setDevice(self.gpuAssignments[i])
      module:reset(stdv)
   end
   setDevice(prevGpuid)
end

function ModelParallelTable:name()
   return 'ModelParallelTable'
end

function ModelParallelTable:type(typeStr)
   if typeStr == "torch.CudaTensor" then
      for i, m in ipairs(self.modules) do
         m:type(typeStr)
      end
   else
      error("ModelParallelTable only supports CudaTensor, not " .. typeStr)
   end
end

function ModelParallelTable:_getSliceRange(tensor, id, total)
   local outerDim = tensor:size(self.dimension)
   assert(outerDim % total == 0) -- FIXME get rid of this restriction
   local eltsPerMod = outerDim / total
   local rangeStart = (id - 1) * eltsPerMod + 1
   local rangeEnd   = id * eltsPerMod

   return tensor:narrow(self.dimension, rangeStart, rangeEnd-rangeStart+1)
end

function ModelParallelTable:_copyTensorRecursive(src, dst, srcGpuid, dstGpuid)
   if (torch.type(src) == 'table') then
      if torch.type(dst) ~= 'table' or #src ~= #dst then
         dst = {}
      end

      -- Recurse on the table
      for i = 1, #src do
         dst[i] = self:_copyTensorRecursive(src[i], dst[i], srcGpuid, dstGpuid)
      end

   elseif torch.type(src):find('torch%..+Tensor') then
      if (dst == nil or torch.type(dst) ~= 'torch.CudaTensor') then
         -- Allocate only on startup or when input table structure changes.
         -- Otherwise we will just resize the tensor below.
         setDevice(dstGpuid)
         dst = torch.CudaTensor()
      end

      -- Split the tensor
      assert(torch.typename(src) == 'torch.CudaTensor')

      if not dst:isSameSizeAs(src) then
         setDevice(dstGpuid)
         dst:resizeAs(src)
      end

      dst:copy(src)
   else
      error('input must be a nested table of tensors!')
   end

   return dst
end

-- _distributeTensorRecursive - if the src is a tensor then the function slices
-- it long self.dimension and copies each portion into each child module.
-- Otherwise it does a recursive call on tables.
function ModelParallelTable:_distributeTensorRecursive(src, dst,
    srcGpuid, srcIndex, dstGpuid, dstIndex, nModules)
   if (torch.type(src) == 'table') then
      if torch.type(dst) ~= 'table' or #src ~= #dst then
         dst = {}
      end

      -- Recurse on the table
      for i = 1, #src do
         dst[i] = self:_distributeTensorRecursive(src[i], dst[i], srcGpuid,
         srcIndex, dstGpuid, dstIndex, nModules)
      end

   elseif torch.type(src):find('torch%..+Tensor') then
      if (dst == nil or torch.type(dst) ~= 'torch.CudaTensor') then
         -- Allocate only on startup or when input table structure changes.
         -- Otherwise we will just resize the tensor below.
         setDevice(dstGpuid)
         dst = torch.CudaTensor()
      end

      -- Split the tensor
      assert(torch.typename(src) == 'torch.CudaTensor')
      local slice = self:_getSliceRange(src, dstIndex, nModules)

      if not dst:isSameSizeAs(slice) then
         setDevice(dstGpuid)
         dst:resizeAs(slice)
      end

      dst:copy(slice)
   else
      error('input must be a nested table of tensors!')
   end

   return dst
end

-- _concatTensorRecursive - if the src is a tensor then the function copies it
-- into the dst slice along self.dimension.
-- Otherwise it does a recursive call on tables.
function ModelParallelTable:_concatTensorRecursive(src, dst, srcGpuid,
   srcIndex, dstGpuid, dstIndex, nModules)
   if (torch.type(src) == 'table') then
      if torch.type(dst) ~= 'table' or #src ~= #dst then
         dst = {}
      end

      -- Recurse on the table
      for i = 1, #src do
         dst[i] = self:_concatTensorRecursive(src[i], dst[i], srcGpuid,
            srcIndex, dstGpuid, dstIndex, nModules)
      end

   elseif torch.type(src):find('torch%..+Tensor') then
      if (dst == nil or torch.type(dst) ~= 'torch.CudaTensor') then
         -- Allocate only on startup or when input table structure changes.
         -- Otherwise we will just resize the tensor below.
         setDevice(dstGpuid)
         dst = torch.CudaTensor()
      end

      if (torch.numel(src) > 0) then
         -- Some modules return empty gradInputs if they don't actually return
         -- anything.
         local dstSize = src:size():totable()
         dstSize[self.dimension] = dstSize[self.dimension] * nModules
         if not (equalSize(dst:size():totable(), dstSize)) then
            assert(srcIndex == 1)
            setDevice(dstGpuid)
            dst:resize(unpack(dstSize))
         end

         -- Split the tensor
         assert(torch.typename(src) == 'torch.CudaTensor')
         local slice = self:_getSliceRange(dst, srcIndex, nModules)
         slice:copy(src)
      end
   else
      error('input must be a nested table of tensors!')
   end

   return dst
end

function ModelParallelTable:_zeroTensorRecursive(src, dst)
   if (torch.type(src) == 'table') then
      if torch.type(dst) ~= 'table' or #src ~= #dst then
         dst = {}
      end

      -- Recurse on the table
      for i = 1, #src do
         dst[i] = self:_zeroTensorRecursive(src[i], dst[i])
      end

   elseif torch.type(src):find('torch%..+Tensor') then
      if (dst == nil or torch.type(dst) ~= 'torch.CudaTensor') then
         dst = torch.CudaTensor()
      end

      -- Split the tensor
      assert(torch.typename(src) == 'torch.CudaTensor')

      if not dst:isSameSizeAs(src) then
         dst:resizeAs(src)
      end
      dst:zero()
   else
      error('input must be a nested table of tensors!')
   end
   return dst
end

function ModelParallelTable:_accumulateTensorRecursive(src, dst)
   if (torch.type(src) == 'table') then
      -- Recurse on the table
      for i = 1, #src do
         dst[i] = self:_accumulateTensorRecursive(src[i], dst[i])
      end
   elseif torch.type(src):find('torch%..+Tensor') then
      dst:add(src)
   else
      error('input must be a nested table of tensors!')
   end
   return dst
end


-- Backward compatibility purposes
ModelParallelTable.__version = 2

-- ModelParallelTable.deserializeNGPUs controls how many GPUs to deserialize
-- upon, otherwise will deserialize to as many GPUs as serialized and error
-- out if it doesn't have enough available
function ModelParallelTable:__read(file, version)
   -- backwards compatibility
   -- TEMPORARY HACK: remove before checking into OSS
   if version < 2 then
      local var = file:readObject()
      for k, v in pairs(var) do
         self[k] = v
      end
      -- hope we didn't run out of memory :)
      local gpu = cutorch.getDevice()
      for i = 1, #self.gpuAssignments do
         -- move each branch to the correct gpu
         cutorch.setDevice(self.gpuAssignments[i])
         self.modules[i]:float():cuda()
      end
      cutorch.setDevice(gpu)
      return
   end

   self.gpuAssignments = file:readObject()

   if ModelParallelTable.deserializeNGPUs then
      if ModelParallelTable.deserializeNGPUs > cutorch.getDeviceCount() then
         error('Deserialization requested on too many GPUs: ' ..
                  ModelParallelTable.deserializeNGPUs .. ' vs ' ..
                  cutorch.getDeviceCount() .. ' available')
      end
      -- round-robin branches
      for i = 1, #self.gpuAssignments do
        self.gpuAssignments[i] = i % ModelParallelTable.deserializeNGPUs
        self.gpuAssignments[i] = (self.gpuAssignments[i]==0) and
            ModelParallelTable.deserializeNGPUs or self.gpuAssignments[i]
      end
   end

   -- If ModelParallelTable.deserializeNGPUs, deserialization overrides
   -- gpu assignments anyway. If not, we need as many GPUs as the max,
   -- there may be holes.
   local nGPUs = math.max(unpack(self.gpuAssignments))
   if nGPUs > cutorch.getDeviceCount() then
      error('Model was serialized on ' ..
               math.max(unpack(self.gpuAssignments)) ..
               ' nGPUs, but you are running on ' .. cutorch.getDeviceCount() ..
               ' please set ModelParallelTable.deserializeNGPUs to ignore ' ..
               ' serialized tower-GPU assignments')
   end

   local gpu = cutorch.getDevice()
   self.modules = {}
   -- deserialize each of the branches on the correct gpu
   for i = 1, #self.gpuAssignments do
      cutorch.setDevice(self.gpuAssignments[i])
      self.modules[i] = file:readObject()
   end

   -- finally deserialize everything else
   cutorch.setDevice(gpu)
   local var = file:readObject()
   for k, v in pairs(var) do
      self[k] = v
   end
end

function ModelParallelTable:__write(file)
   file:writeObject(self.gpuAssignments)

   -- Write all the branches
   local modules = self.modules
   local gpuAssignments = self.gpuAssignments
   self.modules = nil
   self.gpuAssignments = nil
   for _, m in ipairs(modules) do
       file:writeObject(m)
   end

   -- Write everything else as a table
   local t = {}
   for k, v in pairs(self) do
      t[k] = v
   end
   file:writeObject(t)

   self.gpuAssignments = gpuAssignments
   self.modules = modules
end

function ModelParallelTable:clearState()
   self.inputGpu = {}
   self.gradOutputGpu = {}
   self.outputGpu = {}
   self.gradInputGpu = {}

   return parent.clearState(self)
end
