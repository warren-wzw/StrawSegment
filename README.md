# Introduce
This repository contains the source code for our paper:

[U-Net Semantic Segmentation-Based Calorific Value Estimation of Straw Multifuels for Combined Heat and Power Generation Processes](https://www.mdpi.com/1996-1073/17/20/5143)

## Functions
* It introduces a self-attention mechanism in the skip connections to enhance the extraction of key information from deep features;
* It replaces traditional convolutions with depthwise separable convolutions to reduce the model’s computational complexity and improve inference speed;
* It substitutes the bottleneck layer with a Transformer encoder to leverage the global modeling capabilities of Transformers, allowing the model to better understand contextual information within the image.

| Model           | PA(%)↑ | mIoU(%)↑ | Speed(ms)↓ | Size(MB) |
|-----------------|--------|----------|------------|-----------|
| Unet            | 95.56  | 86.8     | 5.11       | 118.76    |
| SegNET          | 94.554 | 84.8     | 4.26       | 112.325   |
| DDRNet          | 87.655 | 71.8     | 2.62       | 21.7247   |
| SmaAt-UNet      | 93.713 | 83.0     | 2.66       | 15.3824   |
| Our model-large | 96.309 | 89       | 5.45       | 129.69    |
| Our model-small | 95.359 | 86.6     | 3.44       | 31.176    |

## Model Architecture
![](./images/1.png)

## Results
![](./images/2.png)
## Env set
* cuda               11.8
* python             3.8.0
* conda create --name StrawSeg python=3.8.0
* Detail please reference requierment.txt

## About author
* warren@伟
* Blog：[CSDN](https://blog.csdn.net/warren103098?type=blog)