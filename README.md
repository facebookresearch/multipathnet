MultiPath Network training code
==========

The code provides functionality to train Fast R-CNN and MultiPath Networks in [Torch-7](http://torch.ch).<br>
Corresponding paper: **A MultiPath Network for Object Detection** http://arxiv.org/abs/1604.02135

![sheep](https://cloud.githubusercontent.com/assets/4953728/17826153/442d027a-666e-11e6-9a1e-2fac95a2d3ba.jpg)

If you use MultiPathNet in your research, please cite the relevant papers:

```
@INPROCEEDINGS{Zagoruyko2016Multipath,
    author = {S. Zagoruyko and A. Lerer and T.-Y. Lin and P. O. Pinheiro and S. Gross and S. Chintala and P. Doll{\'{a}}r},
    title = {A MultiPath Network for Object Detection},
    booktitle = {BMVC}
    year = {2016}
}
```

## Requirements

* Linux
* NVIDIA GPU with compute capability 3.5+

## Installation

The code depends on Torch-7, fb.python and several other easy-to-install torch packages.<br>
To install Torch, follow http://torch.ch/docs/getting-started.html<br>
Then install additional packages:

```bash
luarocks install inn
luarocks install torchnet
luarocks install fbpython
luarocks install class
```

Evaluation relies on COCO API calls via python interface, because lua interface doesn't support it.
Lua API is used to load annotation files in \*json to COCO API data structures. This doesn't work for proposal
files as they're too big, so we provide converted proposals for sharpmask and selective search in torch format.

First, clone https://github.com/pdollar/coco:

```
git clone https://github.com/pdollar/coco
```

Then install LuaAPI:

```
cd coco
luarocks make LuaAPI/rocks/coco-scm-1.rockspec
```

And PythonAPI:

```
cd coco/PythonAPI
make
```

You might need to install Cython for this:

```
sudo apt-get install python-pip
sudo pip install Cython
```

You will have to add the path to PythonAPI to `PYTHONPATH`. Note that this won't work with anaconda as it ships
with it's own libraries which conflict with torch.

### EC2 installation script

Thanks to @DeegC there is [scripts/ec2-install.sh](scripts/ec2-install.sh) script for quick EC2 setup.

## Data preparation

The root folder should have a folder `data` with the following subfolders:

```
models/
annotations/
proposals/
```

`models` folder should contain AlexNet and VGG pretrained imagenet files downloaded from [here](#training). ResNets can resident in other places specified by `resnet_path` env variable.

`annotations` should contain \*json files downloaded from http://mscoco.org/external. There are \*json annotation files for
PASCAL VOC, MSCOCO, ImageNet and other datasets.

`proposals` should contain \*t7 files downloaded from here
We provide selective search VOC 2007 and VOC 2012 proposals converted from https://github.com/rbgirshick/fast-rcnn and SharpMask proposals for COCO 2015 converted from https://github.com/facebookresearch/deepmask, which can be used to compute proposals for new images as well.

Here is an example structure:

```
data
|-- annotations
|   |-- instances_train2014.json
|   |-- instances_val2014.json
|   |-- pascal_test2007.json
|   |-- pascal_train2007.json
|   |-- pascal_train2012.json
|   |-- pascal_val2007.json
|   `-- pascal_val2012.json
|-- models
|   |-- caffenet_fast_rcnn_iter_40000.t7
|   |-- imagenet_pretrained_alexnet.t7
|   |-- imagenet_pretrained_vgg.t7
|   `-- vgg16_fast_rcnn_iter_40000.t7
`-- proposals
    |-- VOC2007
    |   `-- selective_search
    |       |-- test.t7
    |       |-- train.t7
    |       |-- trainval.t7
    |       `-- val.t7
    `-- coco
        `-- sharpmask
            |-- train.t7
            `-- val.t7
```

Download selective_search proposals for VOC2007:

```bash
wget https://dl.fbaipublicfiles.com/multipathnet/proposals/VOC2007/selective_search/train.t7
wget https://dl.fbaipublicfiles.com/multipathnet/proposals/VOC2007/selective_search/val.t7
wget https://dl.fbaipublicfiles.com/multipathnet/proposals/VOC2007/selective_search/trainval.t7
wget https://dl.fbaipublicfiles.com/multipathnet/proposals/VOC2007/selective_search/test.t7
```

Download sharpmask proposals for COCO:

```bash
wget https://dl.fbaipublicfiles.com/multipathnet/proposals/coco/sharpmask/train.t7
wget https://dl.fbaipublicfiles.com/multipathnet/proposals/coco/sharpmask/val.t7
```

As for the images themselves, provide paths to VOCDevkit and COCO in [config.lua](config.lua)

## Running DeepMask with MultiPathNet on provided image

We provide an example of how to extract DeepMask or SharpMask proposals from an image and run recognition MultiPathNet
to classify them, then do non-maximum suppression and draw the found objects.

1. Clone DeepMask project into the root directory:

  ```bash
  git clone https://github.com/facebookresearch/deepmask
  ```

2. Download DeepMask or SharpMask network:

  ```bash
  cd data/models
  # download SharpMask based on ResNet-50
  wget https://dl.fbaipublicfiles.com/deepmask/models/sharpmask/model.t7 -O sharpmask.t7
  ```

3. Download recognition network:

  ```bash
  cd data/models
  # download ResNet-18-based model trained on COCO with integral loss
  wget https://dl.fbaipublicfiles.com/multipathnet/models/resnet18_integral_coco.t7
  ```

4. Make sure you have COCO validation .json files in `data/annotations/instances_val2014.json`

5. Pick some image and run the script:

  ```bash
  th demo.lua -img ./deepmask/data/testImage.jpg
  ```

And you should see this image:

![iterm2 4jpuod lua_khbaaq](https://cloud.githubusercontent.com/assets/4953728/17951035/69d6cb2e-6a5f-11e6-83b8-767c2ae0ae64.png)

See file [demo.lua](demo.lua) for details.

## Training

The repository supports training Fast-RCNN and MultiPath networks with data and model multi-GPU paralellism.
Supported base models are the following:

* AlexNet trained in [caffe](https://github.com/bvlc/caffe) by Ross Girshick, [imagenet_pretrained_alexnet.t7](https://dl.fbaipublicfiles.com/multipathnet/models/imagenet_pretrained_alexnet.t7)
* VGG trained in [caffe](https://github.com/bvlc/caffe) by Ross Girshick, [imagenet_pretrained_vgg.t7](https://dl.fbaipublicfiles.com/multipathnet/models/imagenet_pretrained_vgg.t7)
* ResNets trained in torch with [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) by Sam Gross
* inception-v3 trained in [tensorflow](https://github.com/tensorflow/tensorflow) by Google
* Network-In-Network trained in torch with [imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch) by Sergey Zagoruyko

### PASCAL VOC

To train Fast-RCNN on VOC2007 trainval with VGG base model and selective search proposals do:

```bash
test_nsamples=1000 model=vgg ./scripts/train_fastrcnn_voc2007.sh
```

The resulting mAP is slightly (~2 mAP) higher than original Fast-RCNN number. We should mention that the code is not exactly the same
as we improved ROIPooling by fixing a few bugs, see https://github.com/szagoruyko/imagine-nn/pull/17

### COCO

To train MultiPathNet with VGG-16 base model on 4 GPUs run:

```bash
train_nGPU=4 test_nGPU=1 ./scripts/train_multipathnet_coco.sh
```

Here is a graph visualization of the network (click to enlarge):

<a href="https://dl.fbaipublicfiles.com/multipathnet/extra/multipathnet.pdf">
<img width="1109" alt="multipathnet" src="https://cloud.githubusercontent.com/assets/4953728/17974712/70baf428-6ae7-11e6-9881-e0dea8ab9c18.png">
</a>

To train ResNet-18 on COCO do:

```bash
train_nGPU=4 test_nGPU=1 model=resnet resnet_path=./data/models/resnet/resnet-18.t7 ./scripts/train_coco.sh
```

## Evaluation

### PASCAL VOC

We provide original models from Fast-RCNN paper converted to torch format here:
* [caffenet_fast_rcnn_iter_40000.t7](https://dl.fbaipublicfiles.com/multipathnet/models/caffenet_fast_rcnn_iter_40000.t7)
* [vgg16_fast_rcnn_iter_40000.t7](https://dl.fbaipublicfiles.com/multipathnet/models/vgg16_fast_rcnn_iter_40000.t7)

To evaluate these models run:

```bash
model=data/models/caffenet_fast_rcnn_iter_40000.t7 ./scripts/eval_fastrcnn_voc2007.sh
model=data/models/vgg_fast_rcnn_iter_40000.t7 ./scripts/eval_fastrcnn_voc2007.sh
```

### COCO

Evaluate fast ResNet-18-based network trained with integral loss on COCO val5k split ([resnet18_integral_coco.t7](https://dl.fbaipublicfiles.com/multipathnet/models/resnet18_integral_coco.t7) 89MB):

```bash
test_nGPU=4 test_nsamples=5000 ./scripts/eval_coco.sh
```

It achieves 24.4 mAP using 400 SharpMask proposals per image:

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.244
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.402
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.268
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.078
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.266
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.394
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.249
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.368
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.377
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.135
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.444
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.561
```
