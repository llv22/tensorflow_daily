# SSGAN

* Concept: Using generator's capacity of generating sample to improve performance of image classification, furthermore, to enhance generalization capacity of discriminator.
  Paper: [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf)
* Discriminator/Classification Network
  1. use image as input, classify real image into the first N classes, generated image into the last N+1 class.
  2. loss includes:  
     2.1 loss of supervised learning  
     $$ \mathcal{L}_{supervised} = -\mathbb{E}_{x,y \sim P_{data}(x,y)} log [p_{model}(y | x, y<n+1)] $$
     2.2 loss of GAN discriminator  
     $$ \mathcal{L}_{GAN} = -\left\{ \mathbb{E}_{x \sim p_{data}(x)}log[1-p_{model}(y=n+1|x)] + \mathbb{E}_{x \sim Generator}log[1-p_{model}(y=n+1|x)] \right\} $$  
* Tensorflow source code: https://github.com/gitlimlab/SSGAN-Tensorflow
* Package Required: must install progressbar2 package

## 1. Data preparation
```bash
    $ cd SSGAN-Tensorflow
    $ python download.py --dataset CIFAR10
```
## 2. Train model
```bash
    $ python trainer.py --dataset CIFAR10
```
## 3. Test model
```bash
    $ python evaler.py --dataset CIFAR10 --checkpoint ckpt_dir
```