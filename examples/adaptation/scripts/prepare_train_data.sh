#!/usr/bin/env bash
# Create the office-caltech lmdb inputs
#author : Debjeet Majumdar

EXAMPLE=examples/adaptation/datasets
DATA=~/thesis/domain_adaptation_images
TOOLS=build/tools
subfolder=("amazon" "webcam" "dslr")
# TRAIN_DATA_ROOT=/path/to/imagenet/train/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi



for d in "${subfolder[@]}";do
  rm -rf $EXAMPLE/$d/
  echo "Creating $d lmdb..."
  TRAIN_DATA_ROOT=$DATA/$d/images/
  if [ ! -d "$TRAIN_DATA_ROOT" ]; then
    echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
    echo "Set the TRAIN_DATA_ROOT variable in prepare_train_data.sh to the path" \
         "where the adaptation training data is stored."
    exit 1
  fi
  echo "Train data root : $TRAIN_DATA_ROOT"
  echo "data file $DATA/$d/train.txt"
  echo "target lmdb $EXAMPLE/$d"
  GLOG_logtostderr=1 $TOOLS/convert_imageset \
      --resize_height=$RESIZE_HEIGHT \
      --resize_width=$RESIZE_WIDTH \
      --shuffle \
      $TRAIN_DATA_ROOT \
      $DATA/$d/train.txt \
      $EXAMPLE/$d/$d_0_lmdb
  
done

