--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local class = require 'class'
local py = require 'fb.python'

local Coco = class('coco')

function Coco:__init(annFile)
py.exec('import sys')
py.exec('from pycocotools.coco import COCO')
py.exec('from pycocotools.cocoeval import COCOeval')
py.exec([=[
global cocoGt
cocoGt = COCO(annFile)
]=], {annFile=annFile})
end

function Coco:evaluate(res)
py.exec([=[
global stats
cocoDt = cocoGt.loadRes(res)
imgIds=sorted(cocoDt.imgToAnns.keys())
imgIds=imgIds[0:len(imgIds)]
cocoEval = COCOeval(cocoGt,cocoDt)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
stats = cocoEval.stats
]=], {res=res})
return py.eval('stats')
end

return Coco
