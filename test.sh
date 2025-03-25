#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12321 basicsr/test.py -opt Aberration_Correction/Options/Eval_Aberration_Transformers.yml --launcher pytorch --name $2