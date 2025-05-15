# Posterior Sampling using Proximal Stochastic Gradient Langevin Algorithm (PSGLA)

This repository contains the code to sample posterior distribution in 2D or for images using PSGLA and comparable methods.



## Environment definition

Our code that is based on Pytorch and DeepInv libraries, you need to create a conda environment with our libraries version

```
conda env create -f environment.yml
```

## Deep Neural Networks Weights 

You need to create a folder named "Pretrained_models" and download to follow weights and put them in the folder "Pretrained_models" :
- For DnCNN with control Lipschitz constant proposed by [Learning Maximum Monotone](https://github.com/matthieutrs/LMMO_lightning) follow this [Link](https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_color.pth?download=true)
- For the original DRUNet proposed by [DPIR](https://github.com/cszn/DPIR) follow this [Link](https://huggingface.co/deepinv/drunet/resolve/main/drunet_deepinv_color.pth?download=true)
- For the Gradient-Step DRUNet proposed by [GSDRUNet](https://github.com/samuro95/GSPnP) follow this [Link](https://huggingface.co/deepinv/gradientstep/resolve/main/GSDRUNet.ckpt)
- For the Prox-DRUNet proposed by [Prox-DRUNet](https://github.com/samuro95/Prox-PnP) follow this [Link](https://plmbox.math.cnrs.fr/f/faf7d62213e449fa9c8a/?dl=1)