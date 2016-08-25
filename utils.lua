--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

stringx = require('pl.stringx') -- must be global or threads will barf :(

local tnt = require 'torchnet'

local utils = {}

local ffi = require 'ffi'
ffi.cdef[[
void bbox_vote(THFloatTensor *res, THFloatTensor *nms_boxes, THFloatTensor *scored_boxes, float threshold);
void NMS(THFloatTensor *keep, THFloatTensor *scored_boxes, float overlap);
]]

local ok, C = pcall(ffi.load, './libnms.so')
if not ok then
   os.execute'make'
   ok, C = pcall(ffi.load, './libnms.so')
   assert(ok, 'run make and check what is wrong')
end


function utils.nms(boxes, overlap)
   local keep = torch.FloatTensor()
   C.NMS(keep:cdata(), boxes:cdata(), overlap)
   return keep
end

function utils.bbox_vote(nms_boxes, scored_boxes, overlap)
   local res = torch.FloatTensor()
   C.bbox_vote(res:cdata(), nms_boxes:cdata(), scored_boxes:cdata(), overlap)
   return res
end


--------------------------------------------------------------------------------
-- utility functions for the evaluation part
--------------------------------------------------------------------------------

function utils.joinTable(input,dim)
   local size = torch.LongStorage()
   local is_ok = false
   for i=1,#input do
      local currentOutput = input[i]
      if currentOutput:numel() > 0 then
         if not is_ok then
            size:resize(currentOutput:dim()):copy(currentOutput:size())
            is_ok = true
         else
            size[dim] = size[dim] + currentOutput:size(dim)
         end
      end
   end
   local output = input[1].new():resize(size)
   local offset = 1
   for i=1,#input do
      local currentOutput = input[i]
      if currentOutput:numel() > 0 then
         output:narrow(dim, offset,
         currentOutput:size(dim)):copy(currentOutput)
         offset = offset + currentOutput:size(dim)
      end
   end
   return output
end

--------------------------------------------------------------------------------

function utils.keep_top_k(boxes,top_k)
   local X = utils.joinTable(boxes,1)
   if X:numel() == 0 then
      return boxes, 0
   end
   local scores = X[{{},-1}]:sort(1,true)
   local thresh = scores[math.min(scores:numel(),top_k)]
   for i=1,#boxes do
      local bbox = boxes[i]
      if bbox:numel() > 0 then
         local idx = torch.range(1,bbox:size(1)):long()
         local keep = bbox[{{},-1}]:ge(thresh)
         idx = idx[keep]
         if idx:numel() > 0 then
            boxes[i] = bbox:index(1,idx)
         else
            boxes[i]:resize()
         end
      end
   end
   return boxes, thresh
end

--------------------------------------------------------------------------------
-- evaluation
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------

function utils.boxoverlap(a,b)
   local b = b.xmin and {b.xmin,b.ymin,b.xmax,b.ymax} or b
   local x1 = a:select(2,1):clone()
   x1[x1:lt(b[1])] = b[1]
   local y1 = a:select(2,2):clone()
   y1[y1:lt(b[2])] = b[2]
   local x2 = a:select(2,3):clone()
   x2[x2:gt(b[3])] = b[3]
   local y2 = a:select(2,4):clone()
   y2[y2:gt(b[4])] = b[4]

   local w = x2-x1+1;
   local h = y2-y1+1;
   local inter = torch.cmul(w,h):float()
   local aarea = torch.cmul((a:select(2,3)-a:select(2,1)+1) ,
   (a:select(2,4)-a:select(2,2)+1)):float()
   local barea = (b[3]-b[1]+1) * (b[4]-b[2]+1);

   -- intersection over union overlap
   local o = torch.cdiv(inter , (aarea+barea-inter))
   -- set invalid entries to 0 overlap
   o[w:lt(0)] = 0
   o[h:lt(0)] = 0
   return o
end


function utils.intersection(a,b)
   local b = b.xmin and {b.xmin,b.ymin,b.xmax,b.ymax} or b
   local x1 = a:select(2,1):clone()
   x1[x1:lt(b[1])] = b[1]
   local y1 = a:select(2,2):clone()
   y1[y1:lt(b[2])] = b[2]
   local x2 = a:select(2,3):clone()
   x2[x2:gt(b[3])] = b[3]
   local y2 = a:select(2,4):clone()
   y2[y2:gt(b[4])] = b[4]

   local w = x2-x1+1;
   local h = y2-y1+1;
   local inter = torch.cmul(w,h):float()
   local aarea = torch.cmul((a:select(2,3)-a:select(2,1)+1) ,
   (a:select(2,4)-a:select(2,2)+1)):float()
   return torch.cdiv(inter, aarea)
end
--------------------------------------------------------------------------------

function utils.flipBoxes(boxes, image_width)
   local flipped = boxes:clone()
   flipped:select(2,1):copy( - boxes:select(2,3) + image_width + 1 )
   flipped:select(2,3):copy( - boxes:select(2,1) + image_width + 1 )
   return flipped
end

--------------------------------------------------------------------------------

function utils.merge_table(elements)
   local t = {}
   for i,u in ipairs(elements) do
      for k,v in pairs(u) do
         t[k] = v
      end
   end
   return t
end

-- bbox, tbox: [x1,y1,x2,y2]
local function convertTo(out, bbox, tbox)
   if torch.type(out) == 'table' or out:nDimension() == 1 then
      local xc = (bbox[1] + bbox[3]) * 0.5
      local yc = (bbox[2] + bbox[4]) * 0.5
      local w = bbox[3] - bbox[1]
      local h = bbox[4] - bbox[2]
      local xtc = (tbox[1] + tbox[3]) * 0.5
      local ytc = (tbox[2] + tbox[4]) * 0.5
      local wt = tbox[3] - tbox[1]
      local ht = tbox[4] - tbox[2]
      out[1] = (xtc - xc) / w
      out[2] = (ytc - yc) / h
      out[3] = math.log(wt / w)
      out[4] = math.log(ht / h)
   else
      local xc = (bbox[{{},1}] + bbox[{{},3}]) * 0.5
      local yc = (bbox[{{},2}] + bbox[{{},4}]) * 0.5
      local w = bbox[{{},3}] - bbox[{{},1}]
      local h = bbox[{{},4}] - bbox[{{},2}]
      local xtc = (tbox[{{},1}] + tbox[{{},3}]) * 0.5
      local ytc = (tbox[{{},2}] + tbox[{{},4}]) * 0.5
      local wt = tbox[{{},3}] - tbox[{{},1}]
      local ht = tbox[{{},4}] - tbox[{{},2}]
      out[{{},1}] = (xtc - xc):cdiv(w)
      out[{{},2}] = (ytc - yc):cdiv(h)
      out[{{},3}] = wt:cdiv(w):log()
      out[{{},4}] = ht:cdiv(h):log()
   end
end

function utils.convertTo(...)
   local arg = {...}
   if #arg == 3 then
      convertTo(...)
   else
      local x = arg[1]:clone()
      convertTo(x, arg[1], arg[2])
      return x
   end
end

function utils.convertFrom(out, bbox, y)
   if torch.type(out) == 'table' or out:nDimension() == 1 then
      local xc = (bbox[1] + bbox[3]) * 0.5
      local yc = (bbox[2] + bbox[4]) * 0.5
      local w = bbox[3] - bbox[1]
      local h = bbox[4] - bbox[2]

      local xtc = xc + y[1] * w
      local ytc = yc + y[2] * h
      local wt = w * math.exp(y[3])
      local ht = h * math.exp(y[4])

      out[1] = xtc - wt/2
      out[2] = ytc - ht/2
      out[3] = xtc + wt/2
      out[4] = ytc + ht/2
   else
      assert(bbox:size(2) == y:size(2))
      assert(bbox:size(2) == out:size(2))
      assert(bbox:size(1) == y:size(1))
      assert(bbox:size(1) == out:size(1))
      local xc = (bbox[{{},1}] + bbox[{{},3}]) * 0.5
      local yc = (bbox[{{},2}] + bbox[{{},4}]) * 0.5
      local w = bbox[{{},3}] - bbox[{{},1}]
      local h = bbox[{{},4}] - bbox[{{},2}]

      local xtc = torch.addcmul(xc, y[{{},1}], w)
      local ytc = torch.addcmul(yc, y[{{},2}], h)
      local wt = torch.exp(y[{{},3}]):cmul(w)
      local ht = torch.exp(y[{{},4}]):cmul(h)

      out[{{},1}] = xtc - wt * 0.5
      out[{{},2}] = ytc - ht * 0.5
      out[{{},3}] = xtc + wt * 0.5
      out[{{},4}] = ytc + ht * 0.5
   end
end

-- WARNING: DO NOT USE
-- this function is WIP, it doesn't seem to work yet
function utils.setDataParallelN(model, nGPU)
   assert(nGPU)
   assert(nGPU >= 1 and nGPU <= cutorch.getDeviceCount())
   for _,m in ipairs(model:listModules()) do
      if torch.type(m) == 'nn.DataParallelTable' then
         if #m.modules ~= nGPU then
            assert(#m.modules >= 1)
            local inner = m.modules[1]
            inner:float()
            m:__init(m.dimension, m.noGradInput) -- reinitialize
            for i = 1, nGPU do
               cutorch.withDevice(i, function()
                  m:add(inner:clone():cuda(), i)
               end)
            end
         end
      end
   end
   collectgarbage(); collectgarbage();
end

function utils.removeDataParallel(model)
   for _,m in ipairs(model:listModules()) do
      if m.modules then
         for j,inner in ipairs(m.modules) do
            if torch.type(inner) == 'nn.DataParallelTable' then
               assert(#inner.modules >= 1)
               m.modules[j] = inner.modules[1]:float():cuda() -- maybe move to the right GPU
            end
         end
      end
   end
   -- model:float():cuda() -- maybe move to the right GPU
end

-- Deletes entries in modulesToOptState for modules that don't have parameters
-- in the network. This includes modules in DataParallelTable that aren't on
-- the primary GPU.
function utils.cleanupOptim(state)
   local params, gradParams = state.network:parameters()
   local map = {}
   for _,param in ipairs(params) do
      map[param] = true
   end

   local optimizer = state.optimizer
   for module, _ in pairs(optimizer.modulesToOptState) do
      if not map[module.weight] and not map[module.bias] then
         optimizer.modulesToOptState[module] = nil
      end
   end
end

function utils.makeProposalPath(proposal_dir, dataset, proposals, set, imagenet)
   local res = {}
   if set == 'val5k' then set = 'val' end
   if set == 'val35k' then set = 'val' end
   proposals = stringx.split(proposals, ',')
   for i = 1, #proposals do
      if dataset=='coco' and set=='trainval' then
         table.insert(res, paths.concat(proposal_dir, dataset,  proposals[i], 'train.t7'))
         table.insert(res,  paths.concat(proposal_dir, dataset, proposals[i], 'val.t7'))
      elseif dataset=='VOC2007,2012' then
         table.insert(res, paths.concat(proposal_dir, 'VOC2007', proposals[i], set .. '.t7'))
         table.insert(res, paths.concat(proposal_dir, 'VOC2012', proposals[i], set .. '.t7'))
      else
         table.insert(res, paths.concat(proposal_dir, dataset, proposals[i], set .. '.t7'))
      end
   end

   if opt and opt.extra_proposals_file ~= '' then
      table.insert(res, opt.extra_proposals_file)
   end

   if imagenet then
      -- deepmask, cuz that's all we got
      table.insert(res, paths.concat(proposal_dir, 'imagenet', 'deepmask', 'train.t7'))
   end


   return res
end

function utils.saveResults(aboxes, dataset, res_file)

   nClasses = #aboxes
   nImages = #aboxes[1]

   local size = 0
   for class, rc in pairs(aboxes) do
      for i, data in pairs(rc) do
         if data:nElement() > 0 then
            size = size + data:size(1)
         end
      end
   end

   local out = {}
   out.dataset = dataset
   out.images = torch.range(1,nImages):float()
   local det = {}
   out.detections = det
   det.boxes = torch.FloatTensor(size, 4)
   det.scores = torch.FloatTensor(size)
   det.categories = torch.FloatTensor(size)
   det.images = torch.FloatTensor(size)
   local off = 1
   for class = 1, #aboxes do
      for i = 1, #aboxes[class] do
         local data = aboxes[class][i]
         if data:nElement() > 0 then
            det.boxes:narrow(1, off, data:size(1)):copy(data:narrow(2,1,4))
            det.scores:narrow(1, off, data:size(1)):copy(data:select(2,5))
            det.categories:narrow(1, off, data:size(1)):fill(class)
            det.images:narrow(1, off, data:size(1)):fill(i)
            off = off + data:size(1)
         end
      end
   end
   torch.save(res_file, out)
end

-- modified nn.utils
-- accepts different types and numbers
function utils.recursiveCopy(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = utils.recursiveCopy(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resize(t2:size()):copy(t2)
   elseif torch.type(t2) == 'number' then
      t1 = t2
   else
      error("expecting nested tensors or tables. Got "..
      torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

function utils.recursiveCast(dst, src, type)
   if #dst == 0 then
      tnt.utils.table.copy(dst, nn.utils.recursiveType(src, type))
   end
   utils.recursiveCopy(dst, src)
end

-- another version of nms that returns indexes instead of new boxes
function utils.nms_dense(boxes, overlap)
  local n_boxes = boxes:size(1)

  if n_boxes == 0 then
    return torch.LongTensor()
  end

  -- sort scores in descending order
  assert(boxes:size(2) == 5)
  local vals, I = torch.sort(boxes:select(2,5), 1, true)

  -- sort the boxes
  local boxes_s = boxes:index(1, I):t():contiguous()

  local suppressed = torch.ByteTensor():resize(boxes_s:size(2)):zero()

  local x1 = boxes_s[1]
  local y1 = boxes_s[2]
  local x2 = boxes_s[3]
  local y2 = boxes_s[4]
  local s  = boxes_s[5]

  local area = torch.cmul((x2-x1+1), (y2-y1+1))

  local pick = torch.LongTensor(s:size(1)):zero()

  -- these clones are just for setting the size
  local xx1 = x1:clone()
  local yy1 = x1:clone()
  local xx2 = x1:clone()
  local yy2 = x1:clone()
  local w = x1:clone()
  local h = x1:clone()

  local pickIdx = 1
  for c = 1, n_boxes do
    if suppressed[c] == 0 then
      pick[pickIdx] = I[c]
      pickIdx = pickIdx + 1

      xx1:copy(x1):clamp(x1[c], math.huge)
      yy1:copy(y1):clamp(y1[c], math.huge)
      xx2:copy(x2):clamp(0, x2[c])
      yy2:copy(y2):clamp(0, y2[c])

      w:add(xx2, -1, xx1):add(1):clamp(0, math.huge)
      h:add(yy2, -1, yy1):add(1):clamp(0, math.huge)
      local inter = w
      inter:cmul(h)
      local union = xx1
      union:add(area, -1, inter):add(area[c])
      local ol = h
      torch.cdiv(ol, inter, union)

      suppressed:add(ol:gt(overlap)):clamp(0,1)
    end
  end

  pick = pick[{{1,pickIdx-1}}]
  return pick
end

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k,v in pairs(tbl) do
      -- will skip all DPTs. it also causes stack overflow, idk why
      if torch.typename(v) == 'nn.DataParallelTable' then
         v = v:get(1)
      end
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

utils.deepCopy = deepCopy

function utils.checkpoint(net)
   return deepCopy(net):float():clearState()
end


return utils
