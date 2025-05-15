import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
import sys
import os
from natsort import os_sorted
from utils import *
from restoration_algorithms import *
import argparse
import imageio
import deepinv as deepinv

###
# Parser arguments
###

parser = argparse.ArgumentParser()
parser.add_argument("--n_iter", type=int, default=10000, help='number of iteration')
parser.add_argument("--alpha", type=float, default=1., help='relaxation parameter of the denoiser')
parser.add_argument("--s", type=float, default=5., help='denoiser parameter')
parser.add_argument("--img", type = str, default = 'castle.png', help = 'image in the set20 dataset to reconstruct')
parser.add_argument("--dataset_name", type = str, default = 'set1c', help = 'dataset of images to reconstruct')
parser.add_argument("--path_result", type=str, default='result', help='path to save the results : it will be save in results/path_result')
parser.add_argument("--model_name", type=str, default = None, help='name of the model for our models')
parser.add_argument("--gpu_number", type=int, default = 0, help='gpu number use')
parser.add_argument("--Lip", type=bool, default = False, help='True : the network is 1-Lip, False : no constraint')
parser.add_argument("--blur_type", type=str, default = 'uniform', help='uniform : uniform blur, gaussian : gaussian blur')
parser.add_argument("--l", type=int, default = 4, help='(2*l+1)*(2*l+1) is the size of the blur kernel. Need to verify 2l+1 < 128')
parser.add_argument("--si", type=float, default = 1., help='variance of the blur kernel in case of gaussian blur')
parser.add_argument("--prop", type=float, default = 0.5, help='proportion of masked pixels in random inpainting')
parser.add_argument("--num_of_layers", type=int, default = 17, help='numbers of layers in the deep neural network')
parser.add_argument("--delta", type=float, default = 3e-5, help='step-size for the data-fidelity')
parser.add_argument("--lambd", type=float, default = 1., help='regularization weights')
parser.add_argument("--zeta", type=float, default = 0.8, help='regularization weights for DiffPIR')
parser.add_argument("--t_start", type=int, default = 200, help='time of start for DiffPIR')
parser.add_argument("--seed_ip", type=int, default = 0, help='seed for the inverse problem')
parser.add_argument("--seed_alg", type=int, default = 0, help='seed for the algorithm running')
parser.add_argument("--Pb", type=str, default = 'inpainting', help="Type of problem, possible : 'deblurring', 'inpainting'")
parser.add_argument('--grayscale', dest='grayscale', action='store_true')
parser.set_defaults(grayscale=False)
parser.add_argument('--save_images_online', dest='save_images_online', action='store_true')
parser.set_defaults(save_images_online=False)
parser.add_argument("--alg", type=str, default = 'psgla', help="Choice of algorithm, implemented alg : 'psgla', 'pnp_ula', 'pnp'")
parser.add_argument("--den", type=str, default = 'Prox_DRUNet', help="Choice of denoiser with pretrained weights on color natural images, implemented alg : 'Prox_DRUNet', 'DnCNN'")
parser.add_argument("--den_TV_it", type=int, default = 10, help="Number of iteration to estimate the Prox TV at each iteration of the algorithm")
parser.add_argument("--indx_start", type = int, default = 0, help = "Indice of image to start to restore inside the dataset")
pars = parser.parse_args()

###
# PARAMETERS
###

# Parameters for PnP-ULA
n_iter = pars.n_iter #1000
n_burn_in = int(n_iter/10)
n_inter = int(n_iter/1000)
n_inter_mmse = np.copy(n_inter)

# Denoiser parameters
s = pars.s

# Regularization parameters
alpha = pars.alpha # 1 or 0.3
c_min = 0 #-1
c_max = 1 #2

# Inverse problem prameters
sigma = 1
l = pars.l # size of the blurring kernel

# Path to save the results
path_result = 'results/' + pars.path_result
os.makedirs(path_result, exist_ok = True)

path_result = os.path.join(path_result, pars.Pb)
os.makedirs(path_result, exist_ok = True)
if '--prop' in sys.argv:
    path_result = os.path.join(path_result, 'prop_'+str(pars.prop))
    os.makedirs(path_result, exist_ok = True)
path_result = os.path.join(path_result, pars.dataset_name)
os.makedirs(path_result, exist_ok = True)
path_result = os.path.join(path_result, pars.alg)
os.makedirs(path_result, exist_ok = True)
path_result = os.path.join(path_result, pars.den)
os.makedirs(path_result, exist_ok = True)
if '--s' in sys.argv:
    path_result = os.path.join(path_result, 's_'+str(pars.s))
    os.makedirs(path_result, exist_ok = True)
if '--delta' in sys.argv:
    path_result = os.path.join(path_result, 'delta_'+str(pars.delta))
    os.makedirs(path_result, exist_ok = True)
if '--lambd' in sys.argv:
    path_result = os.path.join(path_result, 'lambd_'+str(pars.lambd))
    os.makedirs(path_result, exist_ok = True)
if '--alpha' in sys.argv:
    path_result = os.path.join(path_result, 'alpha_'+str(pars.alpha))
    os.makedirs(path_result, exist_ok = True)
if '--n_iter' in sys.argv:
    path_result = os.path.join(path_result, 'n_iter_'+str(pars.n_iter))
    os.makedirs(path_result, exist_ok = True)
if '--seed_alg' in sys.argv:
    path_result = os.path.join(path_result, 'seed_alg_'+str(pars.seed_alg))
    os.makedirs(path_result, exist_ok = True)
if '--zeta' in sys.argv:
    path_result = os.path.join(path_result, 'zeta_'+str(pars.zeta))
    os.makedirs(path_result, exist_ok = True)
if '--t_start' in sys.argv:
    path_result = os.path.join(path_result, 't_start_'+str(pars.t_start))
    os.makedirs(path_result, exist_ok = True)
if '--den_TV_it' in sys.argv:
    path_result = os.path.join(path_result, 'den_TV_it_'+str(pars.den_TV_it))
    os.makedirs(path_result, exist_ok = True)

###
# Harware Parameters
###

# GPU device selection
cuda = True
device = "cuda:"+str(pars.gpu_number)
# Type
dtype = torch.float32
tensor = torch.FloatTensor
# Seed of the algorithm
seed = pars.seed_alg

# Prior regularization parameter
alphat = torch.tensor(alpha, dtype = dtype, device = device)
# Normalization of the standard deviation noise distribution
sigma1 = sigma/255.0
sigma2 = sigma1**2
sigma2t = torch.tensor(sigma2, dtype = dtype, device = device)
# Normalization of the denoiser noise level
s1 = s/255.
s2 = (s1)**2
s2t = torch.tensor(s2, dtype = dtype, device = device)

###
# Algorithm Parameters
###

if pars.alg == "pnp_ula":
    # Parameter strong convexity in the tails
    lambd = 0.5/(2/sigma2 + alpha/s2)
    lambdt = torch.tensor(lambd, dtype = dtype, device = device)

    # Discretization step-size
    delta_float = 1/3/(1/sigma2 + 1/lambd + alpha/s2)
    deltat = torch.tensor(delta_float, dtype = dtype, device = device)

elif pars.alg == "psgla":
    lambd = pars.lambd
    lambdt = torch.tensor(lambd, dtype = dtype, device = device)
    sig_float = pars.s / 255.
    delta_float = sig_float**2

elif pars.alg == "baseline":
    lambd = delta_float = None

elif pars.alg == "pnp" or pars.alg == "red":
    lambd = pars.lambd
    lambdt = torch.tensor(lambd, dtype = dtype, device = device)
    sig_float = pars.s / 255.
    delta_float = pars.delta

elif pars.alg == "diffpir":
    sig_float = delta_float = None
    n_iter = 20
    if '--lambd' in sys.argv:
        lambd = pars.lambd
    else:
        lambd = .13
    zeta = pars.zeta
    t_start = pars.t_start
    sigma_noise = pars.s / 255.

    if (pars.den != "GSDRUNet") and (pars.den != "DRUNet") and (pars.den != "DiffUNet"):
        raise ValueError("DiffPIR is only implemented with DRUNet architecture.")

# Set input image paths
if '--img' in sys.argv : # if a specific image path is given
    input_paths = ['../datasets/set20/'+pars.img]
else : # if not given, we aply on the whole dataset name given in argument 
    input_path = os.path.join('../datasets',pars.dataset_name)
    input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

for i in range(pars.indx_start, len(input_paths)):
    path_result_im = os.path.join(path_result, 'im_'+str(i))
    os.makedirs(path_result_im, exist_ok = True)

    ###
    # IMAGE
    ###

    # Image loading
    im_path = input_paths[i]
    im_int = imread_uint(im_path)
    # im_int = im_int[:256,:256,:]

    # Image normalization
    im = np.float32(im_int/255.)

    if pars.grayscale:
        im_t = torch.from_numpy(np.ascontiguousarray(im)).float().unsqueeze(0).unsqueeze(0).to(device)
    else:
        im_t = torch.from_numpy(np.transpose(np.ascontiguousarray(im),(2,0,1))).float().unsqueeze(0).to(device)

    ###
    # Prior Fidelity
    ###

    # Add the root directory to sys.path
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # if pars.Lip:
    #     from models.model_dncnn.realSN_models import DnCNN
    # else:
    #     from models.model_dncnn.models import DnCNN
    # path_model = "../Pretrained_models/" + pars.model_name + ".pth"
    # net = DnCNN(channels=1, num_of_layers=pars.num_of_layers)
    # model = nn.DataParallel(net,device_ids=[int(str(device)[-1])],output_device=device)#.cuda()
    # dicti = torch.load(path_model, map_location=torch.device(device if torch.cuda.is_available() else "cpu"))
    # dicti_ = {}
    # for keys, values in dicti.items():
    #     dicti_["module."+keys] = values.to(device)
    # model.load_state_dict(dicti_)
    # model.eval()

    if pars.den == 'DnCNN':
        denoiser = deepinv.models.DnCNN(in_channels=3, out_channels=3, pretrained='../Pretrained_models/dncnn_sigma2_lipschitz_color.pth', device = device)
    elif pars.den == 'Prox_DRUNet':
        denoiser = deepinv.models.GSDRUNet(in_channels=3, out_channels=3, pretrained='../Pretrained_models/Prox-DRUNet.ckpt', device = device, act_mode="s")
    elif pars.den == 'GSDRUNet':
        denoiser = deepinv.models.GSDRUNet(in_channels=3, out_channels=3, pretrained='../Pretrained_models/GSDRUNet.ckpt', device = device)
    elif pars.den == 'DRUNet':
        denoiser = deepinv.models.DRUNet(in_channels=3, out_channels=3, pretrained='../Pretrained_models/drunet_color.pth', device=device)
    elif pars.den == 'TV':
        denoiser = deepinv.models.TVDenoiser(n_it_max=pars.den_TV_it).to(device)
    else:
        raise ValueError("Denoiser not implemented.")

    Ds = lambda x : denoiser.forward(x, s1)
    prior_grad = lambda x : alphat*(Ds(x) - x)/s2t

    ###
    # Data Fidelity
    ###

    if pars.Pb == 'inpainting':
        gen = torch.Generator(device=device)
        gen.manual_seed(pars.seed_ip) # for reproductivity
        mask = torch.rand((im_t.shape[2],im_t.shape[3]), generator=gen, device = device)
        prop = pars.prop
        mask_2d = 1*(mask > prop)
        neg_mask_2d = 1*(mask <= prop)
        mask = (torch.ones(3)[None,:,None,None].to(device))*mask_2d[None,None,:,:]
        neg_mask = 1 - mask

        y_t = mask * im_t + torch.normal(torch.zeros(*im_t.size()).to(device), std = sigma1*torch.ones(*im_t.size()).to(device),generator=gen)
        data_grad = lambda x: -mask*(x - y_t)/(sigma2t)

        y = y_t.cpu().detach().numpy()
        y = np.transpose(y[0,:,:,:], (1,2,0))
        plt.imsave(path_result_im + '/observation.png', np.clip(y,0,1)) #save the missing pixel image

        #initialization at the Markov Chain
        init_torch = mask * y_t  + neg_mask * 0.5 * torch.ones(y_t.shape).to(device)

    if pars.Pb == 'deblurring':
        # Definition of the convolution operator
        if pars.blur_type == 'uniform':
            l_h = 2*l+1
            h = np.ones((1, l_h))
        if pars.blur_type == 'gaussian':
            si = pars.si
            h = np.array([[np.exp(-i**2/(2*si**2)) for i in range(-l,l+1)]])
        h = h/np.sum(h)
        h_= np.dot(h.T,h)
        h_conv = np.flip(h_) # Definition of Data-grad
        h_conv = np.copy(h_conv) #Useful because pytorch cannot handle negatvie strides
        hconv_torch = torch.from_numpy(h_conv).type(tensor).to(device)
        hcorr_torch = torch.from_numpy(h_).type(tensor).to(device)
        if pars.grayscale:
            hconv_torch = hconv_torch.unsqueeze(0).unsqueeze(0)
            hcorr_torch = hcorr_torch.unsqueeze(0).unsqueeze(0)
        else:
            ones_torch = torch.ones(3,hconv_torch.shape[0],hconv_torch.shape[1]).to(device)
            hconv_torch = hconv_torch.unsqueeze(0)
            hconv_torch = hconv_torch[None,:,:,:] * ones_torch[:,None,:,:]
            hcorr_torch = hcorr_torch.unsqueeze(0)
            hcorr_torch = hcorr_torch[None,:,:,:] * ones_torch[:,None,:,:]

        #forward model definition
        A = lambda x: torch.nn.functional.conv2d(torch.nn.functional.pad(x, [l,l,l,l], mode = 'circular'), hconv_torch, groups=x.size(1), padding = 0)
        AT = lambda x: torch.nn.functional.conv2d(torch.nn.functional.pad(x, [l,l,l,l], mode = 'circular'), hcorr_torch, groups=x.size(1), padding = 0)

        #blur the blur image in torch
        gen = torch.Generator(device=device)
        gen.manual_seed(pars.seed_ip) #for reproductivity
        y_t = A(im_t) + torch.normal(torch.zeros(*im_t.size()).to(device), std = sigma1*torch.ones(*im_t.size()).to(device),generator=gen)

        # DATA-GRAD FOR THE DEBLURRING
        data_grad = lambda x: -AT(A(x) - y_t)/(sigma2t)

        #initialization at the Markov Chain
        init_torch = y_t

    ###
    # Restoration
    ###

    # Name for data storage
    name = 'sigma{}_s{}'.format(sigma, s)

    if pars.alg == "psgla":
        Samples_t, Mmse_t, Mmse2_t = psgla(init = init_torch, data_grad = data_grad, denoiser = denoiser, alpha = alphat, lambd = lambdt, sig_float = sig_float, delta = delta_float, seed = seed, device = device, n_iter = n_iter, n_inter = n_inter, n_inter_mmse = n_inter_mmse, path = path_result_im, save_images_online = pars.save_images_online, name = name)
    elif pars.alg == "baseline":
        if pars.Pb == "inpainting":
            Samples_t, Mmse_t, Mmse2_t = baseline_restoration_inpainting(y_t, neg_mask_2d, device)
        else:
            raise ValueError("Method only implemented for inpainting.")
    elif pars.alg == "pnp_ula":
        Samples_t, Mmse_t, Mmse2_t = pnpula(init = init_torch, data_grad = data_grad, prior_grad = prior_grad, delta = deltat, lambd = lambdt, seed = seed, device = device, n_iter = n_iter, n_inter = n_inter, n_inter_mmse = n_inter_mmse, path = path_result_im, save_images_online = pars.save_images_online, name = name)
    elif pars.alg == "pnp":
        Samples_t, Mmse_t, Mmse2_t = pnp(init = init_torch, data_grad = data_grad, Pb = pars.Pb, denoiser = denoiser, alpha = alphat, lambd = lambdt, sig_float = sig_float, delta = delta_float, seed = seed, device = device, n_iter = n_iter, n_inter = n_inter, n_inter_mmse = n_inter_mmse, path = path_result_im, save_images_online = pars.save_images_online, name = name)
    elif pars.alg == "red":
        Samples_t, Mmse_t, Mmse2_t = red(init = init_torch, data_grad = data_grad, Pb = pars.Pb, denoiser = denoiser, alpha = alphat, lambd = lambdt, sig_float = sig_float, delta = delta_float, seed = seed, device = device, n_iter = n_iter, n_inter = n_inter, n_inter_mmse = n_inter_mmse, path = path_result_im, save_images_online = pars.save_images_online, name = name)
    elif pars.alg == "diffpir":
        Samples_t, Mmse_t, Mmse2_t = diffpir(y = y_t, mask = mask, device = device, denoiser = denoiser, n_iter = n_iter, lambda_ = lambd, zeta = zeta, t_start = t_start, sigma_noise = sigma_noise)


    #convert object in numpy array for analyse
    Samples, Mmse, Mmse2, Psnr_sample, SIM_sample, Min_sample, Max_sample = [], [], [], [], [], [], []

    for i, sample in enumerate(Samples_t):
        samp = sample.cpu().detach().numpy()
        if not(pars.grayscale):
            samp = np.transpose(samp,(1,2,0))
        Psnr_sample.append(PSNR(im, samp, data_range = 1))
        if pars.grayscale:
            SIM_sample.append(ssim(im, samp, data_range = 1))
        else:
            SIM_sample.append(ssim(im, samp, data_range = 1, channel_axis = 2))
        Samples.append(samp)
        Min_sample.append(np.min(samp))
        Max_sample.append(np.max(samp))

    for m in Mmse_t:
        im_ = m.cpu().detach().numpy()
        if not(pars.grayscale):
            im_ = np.transpose(im_,(1,2,0))
        Mmse.append(im_)
    for m in Mmse2_t:
        im_ = m.cpu().detach().numpy()
        if not(pars.grayscale):
            im_ = np.transpose(im_,(1,2,0))
        Mmse2.append(im_)

    #save the observation
    y = y_t.cpu().detach().numpy()
    if pars.grayscale:
        y = y[0,0,:,:]
        ssb = ssim(im, y, data_range = 1)
    else:
        y = y[0,:,:,:]
        y = np.transpose(y,(1,2,0))
        ssb = ssim(im, y, data_range = 1, channel_axis = 2)
    psb = PSNR(im, y, data_range = 1)

    # Compute PSNR and SIM for the online MMSE
    n = len(Mmse)
    PSNR_list = []
    SIM_list = []
    Mmse = np.array(Mmse)

    if pars.grayscale:
        mean_list = np.cumsum(Mmse, axis = 0) / np.arange(1,n+1)[:,None,None]
    else:
        mean_list = np.cumsum(Mmse, axis = 0) / np.arange(1,n+1)[:,None,None,None]

    for i in range(1,n):
        mean = mean_list[i]
        PSNR_list.append(PSNR(im, mean, data_range = 1))
        if pars.grayscale:
            SIM_list.append(ssim(im, mean, data_range = 1))
        else:
            SIM_list.append(ssim(im, mean, data_range = 1, channel_axis = 2))

    # Computation of the mean and std of the whole Markov chain
    xmmse = np.mean(Mmse, axis = 0)
    pmmse = PSNR(im, xmmse, data_range = 1)
    if pars.grayscale:
        smmse = ssim(im, xmmse, data_range = 1)
    else:
        smmse = ssim(im, xmmse, data_range = 1, channel_axis = 2)

    # Computation of the std of the Markov chain
    xmmse2 = np.mean(Mmse2, axis = 0)
    var = xmmse2 - xmmse**2
    var = var*(var>=0) + 0*(var<0)
    std = np.sqrt(var)
    diff = np.abs(im-xmmse)
    init = init_torch.cpu().detach().numpy()
    init = np.transpose(init[0,:,:,:], (1,2,0))

    #save the result of the experiment
    dict = {
            # 'Samples' : Samples,
            # 'Mmse' : Mmse,
            # 'Mmse2' : Mmse2,
            'PSNR_sample' : Psnr_sample,
            'SIM_sample' : SIM_sample,
            'PSNR_mmse' : PSNR_list,
            'SIM_list' : SIM_list,
            'observation' : y,
            'init' : init,
            'PSNR_y' : psb,
            'SIM_y' : ssb,
            'ground_truth' : im,
            'MMSE' : xmmse,
            'PSNR_MMSE' : pmmse,
            'SIM_MMSE' : smmse,
            'std' : std,
            'diff' : diff,
            'n_iter' : n_iter,
            's' : s,
            'alpha' : alpha,
            'c_min' : c_min,
            'c_max' : c_max,
            'sigma' : sigma,
            'l' : l,
            'lambda' : lambd,
            'delta' : delta_float,
        }

    np.save(path_result_im+'/'+ name +'_result.npy', dict)

    ###
    # PLOTS
    ###

    # #creation of a video of the samples
    # writer = imageio.get_writer(os.path.join(path_result,"samples_video"+name+".mp4"), fps=100)
    # for im_ in Samples:
    #     im_uint8 = np.clip(im_ * 255, 0, 255).astype(np.uint8)
    #     writer.append_data(im_uint8)
    # writer.close()

    # PSNR plots
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(Psnr_sample, "+")
    ax.set_title("PSNR between samples and GT")
    fig.savefig(path_result_im +"/PSNR_between_samples_and_GT_n_iter{}".format(n_iter)+".png")
    plt.show()

    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(PSNR_list, "+")
    ax.set_title("PSNR between online MMSE and GT")
    fig.savefig(path_result_im +"/PSNR_between_online_MMSE_and_GT_n_iter{}".format(n_iter)+".png")
    plt.show()

    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(SIM_sample, "+")
    ax.set_title("SIM between samples and GT")
    fig.savefig(path_result_im +"/SIM_between_samples_and_GT_n_iter{}".format(n_iter)+".png")
    plt.show()

    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(Psnr_sample, "+")
    ax.set_title("SIM between online MMSE and GT")
    fig.savefig(path_result_im +"/SIM_between_online_MMSE_and_GT_n_iter{}".format(n_iter)+".png")
    plt.show()

    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(Max_sample, "+")
    ax.set_title("Maximum value of samples")
    fig.savefig(path_result_im +"/Max_values_samples_n_iter{}".format(n_iter)+".png")
    plt.show()

    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(Min_sample, "+")
    ax.set_title("Minimum value of samples")
    fig.savefig(path_result_im +"/Min_values_samples_n_iter{}".format(n_iter)+".png")
    plt.show()

    if pars.grayscale:
        cmap = 'gray'
    else:
        cmap = None

    #save the observation and the ground-truth
    plt.imsave(path_result_im + '/observation.png', np.clip(y,0,1), cmap = cmap)
    plt.imsave(path_result_im + '/ground_truth.png', np.clip(im,0,1), cmap = cmap)
    plt.imsave(path_result_im + '/init.png', np.clip(init,0,1), cmap = cmap)

    # Saving of the MMSE of the sample
    plt.imsave(path_result_im + '/mmse_' + name + '_psnr{:.2f}_ssim{:.2f}.png'.format(pmmse, smmse), np.clip(xmmse,0,1), cmap = cmap)
    print("The output PSNR : {:.2f} dB / output SSIM : {:.2f}".format(pmmse,smmse))

    # Save the error for inpainting
    if pars.Pb == 'inpainting':
        mask_np = mask.cpu().detach().numpy()
        mask_np = np.transpose(mask_np[0,:,:,:],(1,2,0))
        plt.imsave(path_result_im + '/error.png', np.clip(mask_np*(xmmse-im),0,1), cmap = cmap)

    # Saving of the MMSE compare to the original and observation
    fig = plt.figure(figsize = (10, 10))
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(xmmse, cmap = 'gray')
    ax1.axis('off')
    ax1.set_title("MMSE (PSNR={:.2f}/SSIM={:.2f})".format(pmmse, smmse))
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(im, cmap = 'gray')
    ax2.axis('off')
    ax2.set_title("GT")
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(y, cmap = 'gray')
    ax3.axis('off')
    ax3.set_title("Obs (PSNR={:.2f}/SSIM={:.2f})".format(psb, ssb))
    fig.savefig(path_result_im+'/MMSE_and_Originale_and_Observation_n_iter{}'.format(n_iter)+'.png')
    plt.show()

    # Saving of the standard deviation and the difference between MMSE and Ground-Truth (GT)
    fig = plt.figure(figsize = (10, 5))
    ax1 = fig.add_subplot(1,2,1)
    std_gray = np.sum(std, axis = -1)
    ax1.imshow((std_gray - np.min(std_gray))/(np.max(std_gray) - np.min(std_gray)), cmap = 'gray')
    ax1.axis('off')
    ax1.set_title("Std of the Markov Chain, min = {:.2f}, max = {:.2f}".format(np.min(std_gray), np.max(std_gray)))
    ax2 = fig.add_subplot(1,2,2)
    error = np.abs(im-xmmse)
    ax2.imshow((error - np.min(error))/(np.max(error) - np.min(error)), cmap = 'gray')
    ax2.axis('off')
    ax2.set_title("Diff MMSE-GT, min = {:.2f}, max = {:.2f}".format(np.min(error), np.max(error)))
    fig.savefig(path_result_im+'/Std_of_the_Markov_Chain_n_iter{}'.format(n_iter)+'.png')
    plt.show()

    # Saving of the Fourier transforme of the standard deviation, to detect possible artecfact of sampling
    plt.imsave(path_result_im+"/Fourier_transform_std_MC_n_iter{}".format(n_iter)+".png",np.clip(np.fft.fftshift(np.log(np.abs(np.fft.fft2(std))+1e-10)),0,1))