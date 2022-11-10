# Code for "ConvReLU++: Reference-based Lossless Acceleration of Conv-ReLU Operations for Mobile Deep Vision"


This repo contains three frameworks code, including Pytorch, TFlite-micro and NCNN. Please download pre-trained models and datasets according to our paper. After downloading all codes and assests, compile NCNN and TFlite-micro for reproduce the experiment results.

# Directory Structure

```
.
├── LICENSE
├── README.md
├── ncnn_src    # code in NCNN
├── pytorch_src    # code in Pytorch
└── tflite-micro_src    # code in TFlite-micro
```

# Usage
1. compile [NCNN](https://github.com/Tencent/ncnn) and [TFlite-micro](https://github.com/tensorflow/tflite-micro) according to the official tutorial. 
2. Perform inference with compiled framework with pretrained models and datasets.


# Main Experiment Results
![](results.png)