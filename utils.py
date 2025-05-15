import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
import os
import sys
import utils
import argparse
import imageio
import cv2
from deepinv.optim.data_fidelity import L2

def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img

def psnr(img1,img2) :
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    # img1 = np.float64(img1)
    # img2 = np.float64(img2)
    mse = np.mean((img1 - img2)**2)
    return 20 * np.log10(1. / np.sqrt(mse))

def single2uint(img):
    return np.uint8((img*255.).round())

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def array2tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

def tensor2array(img):
    img = img.cpu()
    img = img.squeeze(0).detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

def imsave(img_path,img):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def load_image_gray(path_img, img):
    """
    Load the image img from path_image and crop it to have a gray scale image 256*256

    Parameters
    ----------
    path_img : str
        path where img is saved
    img : str
        name of the image

    Returns
    -------
    im : ndarray, shape (256, 256)
        The load image
    """
    im_total = plt.imread(path_img+img)
    if img == "duck.png":
        im = np.mean(im_total, axis = 2)
        im = im[400:656,550:806]
    if img == "painting.png" or img == 'cancer.png' or img == 'cells.png':
        im = np.mean(im_total, axis = 2)
        im = cv2.resize(im, dsize=(256, 256))
    if img == "castle.png":
        im = np.mean(im_total, axis = 2)
        im = im[100:356,0:256]
    if img == "simpson_nb512.png" or img == "goldhill.png":
        im = im_total[100:356,0:256]
    if img in set(["cameraman.png", '01.png', '02.png', '03.png', '04.png', '05.png', '06.png', '07.png']):
        im = im_total
    if img == '09.png':
        im = im_total[:256,256:]
    if img == '10.png':
        im = im_total[100:356,256:]
    if img == '11.png':
        im = im_total[:256,:256]
    if img == '12.png':
        im = im_total[100:356,100:356]
    return im

def torch_denoiser(x,model):
    """
    pytorch_denoiser for a denoiser train to predict the noise
    Inputs:
        xtilde      noisy tensor
        model       pytorch denoising model
   
    Output:
        x           denoised tensor
    """

    # denoise
    with torch.no_grad():
        #xtorch = xtilde.unsqueeze(0).unsqueeze(0)
        r = model(x)
        #r = np.reshape(r, -1)
        x_ = x - r
        out = torch.squeeze(x_)
    return out