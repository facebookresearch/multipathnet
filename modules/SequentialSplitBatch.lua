--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local Sequential, parent = torch.class('nn.SequentialSplitBatch', 'nn.Sequential')

function Sequential:__init(size)
   parent.__init(self)
   self.batch_size = size
end

function Sequential:updateOutput(input)
   if torch.type(input) ~= 'table' then
      assert(input:dim() == 2 or input:dim() == 4)
      local batch_size = input:size(1)
      if batch_size <= self.batch_size then
         return parent.updateOutput(self, input)
      else
         -- propagate small batch to determine output size
         local output_size = parent.updateOutput(self,input:narrow(1,1,1)):size()
         output_size[1] = batch_size
         self.output_ = self.output_ or input.new()
         self.parent_output = self.parent_output or input[1].new()
         self.output:set(self.parent_output)
         self.output_:resize(output_size)
         local input_split = input:split(self.batch_size,1)
         local output_split = self.output_:split(self.batch_size,1)
         for i,v in ipairs(input_split) do
            output_split[i]:copy(parent.updateOutput(self,v))
         end
         self.output:set(self.output_)
         return self.output
      end
   elseif torch.type(input) == 'table' then
      -- only 1-nested tables supported
      -- only tensor output so far
      local input_sizes = {}
      for i,v in ipairs(input) do
         assert(v:dim() == 2 or v:dim() == 4)
         input_sizes[i] = v:size(1)
      end
      --assert(torch.Tensor(input_sizes):std() == 0, 'different sizes on input')
      local batch_size = input_sizes[2]

      if batch_size <= self.batch_size then
         return parent.updateOutput(self, input)
      else
         -- propagate small batch
         local subinput = {}
         for i,v in ipairs(input) do subinput[i] = v:narrow(1,1,1) end
         local output_size = parent.updateOutput(self, subinput):size()
         output_size[1] = batch_size
         self.output_ = self.output_ or input[1].new()
         self.parent_output = self.parent_output or input[1].new()
         self.output:set(self.parent_output)
         self.output_:resize(output_size)
         local output_split = self.output_:split(self.batch_size,1)
         local per_input_splits = {}
         for i,v in ipairs(input) do per_input_splits[i] = v:split(self.batch_size,1) end

         assert(self.output_:storage() ~= self.output:storage())
         for k,u in ipairs(output_split) do
            local subinput = {input[1], per_input_splits[2][k]}
            u:copy(parent.updateOutput(self,subinput))
         end
         self.output:set(self.output_)
         return self.output
      end
   end
end


-- for updateGradInput, accGradParameters, etc do not do anything.
