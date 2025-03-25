# Aberration Correcting Vision Transformers for High-Fidelity Metalens Imaging

Byeonghyeon Lee, Youbin Kim, Yongjae Jo, Hyunsu Kim, Hyemi Park, Yangkyu Kim, Debabrata Mandal, Praneeth Chakravarthula, Inki Kim, and Eunbyung Park

[Project Page](https://benhenryl.github.io/Metalens_Transformer/) &nbsp; [Paper](https://arxiv.org/abs/2412.04591) 


We ran the experiments in the following environment:
```
- ubuntu: 20.04
- python: 3.10.13
- cuda: 11.8
- pytorch: 2.2.0
- GPU: 4x A6000 ada
```

Our code is based on [Restormer](https://github.com/swz30/Restormer), [X-Restormer](https://github.com/Andrew0613/X-Restormer), and [Neural Nano Optics](https://github.com/princeton-computational-imaging/Neural_Nano-Optics). We appreciate their works.

## 1. Environment Setting
### 1-1. Pytorch
Note: pytorch >= 2.2.0 is required for Flash Attention.

### 1-2. [Flash Attention](https://github.com/Dao-AILab/flash-attention)
cf. Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100) are supported now.
```
pip install packaging ninja
pip install flash-attn --no-build-isolation
```

### 1-3. Other required packages
```
pip install -r requirements.txt
```

### 1-4. Basicsr
```
python setup.py develop --no_cuda_ext
```

## 2. Dataset & Pre-trained weights
You can download train/test dataset [here](https://drive.google.com/drive/folders/1e2wJwmcjXFvblVs0l5OXwpIkTqxd1Fhq?usp=drive_link) and pre-trained weights [here](https://drive.google.com/drive/folders/1q5pKE1Z0RJjHVmJlNq7nPSWcaGd9bDb7?usp=drive_link).
Please move the pre-trained weights to experiments/.  
Note: The model creates aberrated images on the fly using clean (gt) images during training.  
In case of validation, it also produces the aberrated images in the same manner, where the aberrated images can have different noises to what we used for our validation. 
There will be only negligible difference in the results as it still uses the same noise distributions, but if you want a precise comparison with the validation set we used for our experiments, please contact us.


## 3. Training
Please set dataset path in ```./Aberration_Correction/Options/Train_Aberration_Transformers.yml``` 
```
bash train.sh GPU_IDS FOLDER_NAME 
// ex. bash train.sh 0,1,2,3 training
// where it uses gpu 0 to 3 and make a directory experiments/training where log, weights and others will be stored.
```

## 4. Inference
Please set dataset path in ```./Aberration_Correction/Options/Test_Aberration_Transformers.yml```  
If you want to run a inference using the pre-trained model, you can use a command
```
bash test.sh GPU_ID FOLDER_NAME
// ex. bash test.sh 0 pretrained
```
Or you can designate the FOLDER_NAME with your weight path. 

## BibTeX
```
@article{lee2024aberration,
  title={Aberration Correcting Vision Transformers for High-Fidelity Metalens Imaging},
  author={Lee, Byeonghyeon and Kim, Youbin and Jo, Yongjae and Kim, Hyunsu and Park, Hyemi and Kim, Yangkyu and Mandal, Debabrata and Chakravarthula, Praneeth and Kim, Inki and Park, Eunbyung},
  journal={arXiv preprint arXiv:2412.04591},
  year={2024}
}
```