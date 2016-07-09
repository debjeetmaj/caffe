#!/bin/bash

OFFICE_DIR=$1
 
ROOT_DIR="$( cd "$(dirname "$0")"/../../.. ; pwd -P )"
cd $ROOT_DIR

#Download AlexNet reference model.
echo "[*] Downloading AlexNet reference model..."
python ./scripts/download_model_binary.py ./models/bvlc_alexnet >/dev/null 2>/dev/null

#Download ImageNet aux data.
echo "[*] Downloading ImageNet aux data..."
./data/ilsvrc12/get_ilsvrc_aux.sh >/dev/null 2>/dev/null

Prepare lmdb databases for the Office dataset.
echo "[*] Preparing datasets..."
mkdir ./data/office
for DOMAIN in amazon webcam dslr; do
    python ./examples/adaptation/scripts/convert_data.py \
        -s $OFFICE_DIR/ \
        -t ./data/office/ \
        -d $DOMAIN -i 1 
done

# Prepare directories for the experiments.
echo "[*] Preparing directories for experiments..."
for MODE in amazon_to_webcam dslr_to_webcam webcam_to_dslr; do
    python ./examples/adaptation/scripts/prepare_dirs.py \
        -m $MODE \
        -t ./models/adaptation \
        -d ./data/office \
        -a ./models/bvlc_alexnet/bvlc_alexnet.caffemodel \
        -i ./data/ilsvrc12/imagenet_mean.binaryproto \
        -p ./examples/adaptation/protos 
done
