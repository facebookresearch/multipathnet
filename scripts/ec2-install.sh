#!/bin/bash

# This sets up a new AWS EC2 server to run Facebook's multipathnet.
#
# Requires:
#   - Amazon EC2 instance with GPU (g2.2xlarge or g2.8xlarge) running AWS Linux.
#   - 30 GB of disk space.

sudo pip install numpy
sudo yum -y install boost-devel

sudo yum -y install git
sudo yum -y install automake

wget https://github.com/google/glog/archive/v0.3.3.zip
unzip v0.3.3.zip 
cd glog-0.3.3/
./configure
make
sudo make install

# Install torch
cd ~
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh -b
. ~/torch/install/bin/torch-activate

luarocks install inn
luarocks install torchnet
luarocks install fbpython
luarocks install class
luarocks install optnet

# Install COCO
cd ~
git clone https://github.com/pdollar/coco.git
cd coco
luarocks make LuaAPI/rocks/coco-scm-1.rockspec

cd PythonAPI
make
export PYTHONPATH=$PYTHONPATH:~/coco/PythonAPI

# Install nVidia CUDA
cd ~
wget http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-7.5-linux-x64-v5.1.tgz
tar xvf cudnn-7.5-linux-x64-v5.1.tgz
cd cuda/lib64
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH

# Install Multipathnet
cd ~
git clone https://github.com/facebookresearch/multipathnet.git

cd /tmp
wget http://mscoco.org/static/annotations/PASCAL_VOC.zip
wget http://mscoco.org/static/annotations/ILSVRC2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip

export MPROOT=~/multipathnet
mkdir -p $MPROOT/data/annotations
cd $MPROOT/data/annotations
unzip -j /tmp/PASCAL_VOC.zip
unzip -j /tmp/ILSVRC2014.zip
unzip -j /tmp/instances_train-val2014.zip

mkdir -p $MPROOT/data/proposals/VOC2007/selective_search
cd $MPROOT/data/proposals/VOC2007/selective_search
wget https://s3.amazonaws.com/multipathnet/proposals/VOC2007/selective_search/train.t7
wget https://s3.amazonaws.com/multipathnet/proposals/VOC2007/selective_search/val.t7
wget https://s3.amazonaws.com/multipathnet/proposals/VOC2007/selective_search/trainval.t7
wget https://s3.amazonaws.com/multipathnet/proposals/VOC2007/selective_search/test.t7

mkdir -p $MPROOT/data/proposals/coco/sharpmask
cd $MPROOT/data/proposals/coco/sharpmask
wget https://s3.amazonaws.com/multipathnet/proposals/coco/sharpmask/train.t7
wget https://s3.amazonaws.com/multipathnet/proposals/coco/sharpmask/val.t7

mkdir -p $MPROOT/data/models
cd $MPROOT/data/models
wget https://s3.amazonaws.com/multipathnet/models/imagenet_pretrained_alexnet.t7
wget https://s3.amazonaws.com/multipathnet/models/imagenet_pretrained_vgg.t7
wget https://s3.amazonaws.com/multipathnet/models/vgg16_fast_rcnn_iter_40000.t7
wget https://s3.amazonaws.com/multipathnet/models/caffenet_fast_rcnn_iter_40000.t7

if [ ! -f ~/multipathnet/config.lua.backup ]; then
  cp ~/multipathnet/config.lua ~/multipathnet/config.lua.backup
fi

echo "
-- put your paths to VOC and COCO containing subfolders with images here
local VOCdevkit = '$MPROOT/data/proposals'
local coco_dir = '$MPROOT/data/proposals/coco'

return {
   pascal_train2007 = paths.concat(VOCdevkit, 'VOC2007/selective_search'),
   pascal_val2007 = paths.concat(VOCdevkit, 'VOC2007/selective_search'),
   pascal_test2007 = paths.concat(VOCdevkit, 'VOC2007/selective_search'),
   pascal_train2012 = paths.concat(VOCdevkit, 'VOC2007/selective_search'),
   pascal_val2012 = paths.concat(VOCdevkit, 'VOC2007/selective_search'),
   pascal_test2012 = paths.concat(VOCdevkit, 'VOC2007/selective_search'),
   coco_train2014 = paths.concat(coco_dir, 'sharpmask'),
   coco_val2014 = paths.concat(coco_dir, 'sharpmask'),
}" > ~/multipathnet/config.lua

cd $MPROOT
git clone https://github.com/facebookresearch/deepmask.git

cd $MPROOT/data/models
# download SharpMask based on ResNet-50
wget https://s3.amazonaws.com/deepmask/models/sharpmask/model.t7 -O sharpmask.t7
wget https://s3.amazonaws.com/multipathnet/models/resnet18_integral_coco.t7

echo
echo 'Add the following to your .bashrc:
export PYTHONPATH=~/coco/PythonAPI
export LD_LIBRARY_PATH=~/cuda/lib64:$LD_LIBRARY_PATH'
