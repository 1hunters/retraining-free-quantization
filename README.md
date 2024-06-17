# RFQuant: Retraining-free Model Quantization via One-Shot Weight-Coupling Learning, CVPR (2024) 

Official implementation for paper "Retraining-free Model Quantization via One-Shot Weight-Coupling Learning" (RFQuant). 
RFQuant presents a new training-then-searching pipeline for performing quantization-aware training. 

## Abstract
Quantization is of significance for compressing the over-parameterized deep neural models and deploying them on resource-limited devices. Fixed-precision quantization suffers from performance drop due to the limited numerical representation ability. Conversely, mixed-precision quantization (MPQ) is advocated to compress the model effectively by allocating heterogeneous bit-width for layers. MPQ is typically organized into a searching-retraining two-stage process. Previous works only focus on determining the optimal bit-width configuration in the first stage efficiently, while ignoring the considerable time costs in the second stage. However, retraining always consumes hundreds of GPU-hours on the cutting-edge GPUs, thus hindering deployment efficiency significantly. In this paper, we devise a one-shot training-searching paradigm for mixed-precision model compression. Specifically, in the first stage, all potential bit-width configurations are coupled and thus optimized simultaneously within a set of shared weights. However, our observations reveal a previously unseen and severe bit-width interference phenomenon among highly coupled weights during optimization, leading to considerable performance degradation under a high compression ratio. To tackle this problem, we first design a bit-width scheduler to dynamically freeze the most turbulent bit-width of layers during training, to ensure the rest bit-widths converged properly. Then, taking inspiration from information theory, we present an information distortion mitigation technique to align the behaviour of the bad-performing bit-widths to the well-performing ones. In the second stage, an inference-only greedy search scheme is devised to evaluate the goodness of configurations without introducing any additional training costs. Extensive experiments on three representative models and three datasets demonstrate the effectiveness of the proposed method.

## Environment Setup and Data Preparation
You can use the following command to setup the training/evaluation environment: 

```
git clone https://github.com/1hunters/retraining-free-quantization.git
cd 1hunters/retraining-free-quantization
conda create -n RFQuant python=3.9
conda activate RFQuant
pip install -r requirements.txt
```

We use the ImageNet dataset at http://www.image-net.org/. The training set is moved to /path_to_imagenet/imagenet/train and the validation set is moved to /path_to_imagenet/imagenet/val: 
```
/path_to_imagenet/imagenet/
  train/
    class0/
      img0.jpeg
      ...
    class1/
      img0.jpeg
      ...
    ...
  val/
    class0/
      img0.jpeg
      ...
    class1/
      img0.jpeg
      ...
    ...
```

## Usage

### Training
See the yaml files in ``configs/training/`` folder. Please make sure the path of imagenet dataset is set properly. For example, please use the following command: 

- ResNet18

  ``python -m torch.distributed.launch --nproc_per_node=4 main.py configs/training/train_resnet18_w2to6_a2to6.yaml`` for training the ResNet18 model with bit-width candidates [2, 3, 4, 5, 6] and, 

- MobileNetv2

  ``python -m torch.distributed.launch --nproc_per_node=4 main.py configs/training/train_mobilenetv2_w2to6_a2to6.yaml`` for training the MobileNetv2 model with bit-width candidates [2, 3, 4, 5, 6].

We perform all experiments on NVIDIA A100 (80G) GPUs. If you don't have enough GPU memory, please consider lowering the batchsize or disabling the ``information distortion mitigation`` technique. 

### Search 
We search on a subset of imagenet with our greedy search algorithm. 

## Acknowledgement
The authors would like to thank the following insightful open-source projects & papers, this work cannot be done without all of them:

- LSQ implementation: https://github.com/zhutmost/lsq-net 
- Once-for-all: https://github.com/mit-han-lab/once-for-all

## Citation

```
@inproceedings{tang2024retraining,
  title={Retraining-free Model Quantization via One-Shot Weight-Coupling Learning},
  author={Tang, Chen and Meng, Yuan and Jiang, Jiacheng and Xie, Shuzhao and Lu, Rongwei and Ma, Xinzhu and Wang, Zhi and Zhu, Wenwu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```