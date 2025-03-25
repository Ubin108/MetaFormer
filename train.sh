#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12321 basicsr/train.py -opt Aberration_Correction/Options/Train_Aberration_Transformers.yml --launcher pytorch --name $2