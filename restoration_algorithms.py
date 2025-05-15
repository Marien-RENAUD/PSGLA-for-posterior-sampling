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


def baseline_restoration_inpainting(img_t, mask_t, device, R = 3):
    """
    Implementation in OpenCV of the method proposed in "An Image Inpainting Technique Based on the Fast Marching Method" by Alexandru Telea.
    """
    img_np = img_t.cpu().detach().numpy()
    img_np = np.transpose(img_np[0,:,:,:], (1,2,0))
    img_np = (255*img_np).astype(np.uint8)
    print(mask_t.shape)
    print(img_np.shape)
    mask_np = mask_t.cpu().detach().numpy()
    mask_np = (mask_np).astype(np.uint8)
    print(torch.sum(torch.abs(img_t * mask_t)))
    
    inpaint_img_np = cv2.inpaint(img_np, mask_np, R, cv2.INPAINT_NS)
    inpaint_img_np = inpaint_img_np.astype(np.float32) / 255.0
    inpaint_img_t = torch.from_numpy(np.transpose(np.ascontiguousarray(inpaint_img_np),(2,0,1))).float().to(device)

    return [], [inpaint_img_t], []




def pnpula(init, data_grad, prior_grad, delta, lambd, n_iter = 5000, n_inter = 1000, n_inter_mmse = 1000, seed = None, device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu"), c_min = -1, c_max = 2, path = None, save_images_online = False, name = None):
    """
    PnP-ULA sampling algorithm 

    Inputs:
        init        Initialization of the Markov chain (torch tensor)
        prior_grad  Gradient of the log-prior (already multiplied by the regularization parameter)
        data_grad   Gradient of the likelihood
        delta       Discretization step-size (torch tensor)
        lambd       Moreau-Yoshida regularization parameter (torch tensor)
        n_iter      Number of ULA iterations
        n_inter     Number of iterations before saving of a sample
        n_inter_mmse Number of iterations for a mean computation
        device      cuda device used to store samples
        seed        int, seed used
        path        str : where to store the data
        name        name of the stored dictionnary
        c_min       To ensure strong convexity
        c_max       To ensure strong convexity
    Outputs:
        Xlist       Samples stored every n_inter iterations
        Xlist_mmse  Mmse computed over n_inter_mmse iterations
        Xlist_mmse2 Average X**2 computed over n_inter_mmse iterations
    """
    # Type
    dtype = torch.float32
    tensor = torch.FloatTensor
    # Shape of the image
    im_shape = init.shape
    # Markov chain init
    X = torch.zeros(im_shape, dtype = dtype, device = device)
    X = init.clone().detach()
    # To compute the empirical average over n_inter_mmse
    xmmse = torch.zeros(im_shape, dtype = dtype, device = device)
    # To compute the empirical variance over n_inter_mmse
    xmmse2 = torch.zeros(im_shape, dtype = dtype, device = device)
    # 
    One = torch.ones(xmmse2.shape)
    One = One.to(device)

    # 
    brw = torch.sqrt(2*delta)
    brw = brw.to(device)
    print("delta = {}".format(delta.float()))
 
    if path == None:
        path = str()   
    if seed != None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed) #for reproductivity
    if name == None:
        name = str()
    if n_inter_mmse == None:
        n_inter_mmse = np.copy(n_inter)

    #To store results
    Xlist = []
    Xlist_mmse = []
    Xlist_mmse2 = []
    iter_mmse = 0

    # Frequency at which we save samples
    K = int(n_iter/10)
    
    with torch.no_grad():
        for i in tqdm(range(n_iter)):
            Z = torch.randn(im_shape, generator = gen, dtype = dtype, device = device)
            # grad G : Tweedie's formula
            grad_log_prior = prior_grad(X)
            # grad F : gaussian Data fit
            grad_log_data = data_grad(X) # T comment if we want to sample from the prior
            # Projection
            out = torch.where(X>c_min, X, c_min*One)
            proj = torch.where(out<c_max, out, c_max*One)
            # grad log-posterior
            gradPi = grad_log_prior - (X-proj)/lambd  + grad_log_data
            # Langevin update
            X = X + delta*gradPi + brw*Z

            # To save samples of the Markov chain after the burn-in every n_inter iterations.
            if i%n_inter == 0:
                X_ = torch.squeeze(X)
                # Sample Storage
                Xlist.append(X_)
            
            if i % K == 0 and save_images_online:
                X_numpy = X.detach().cpu().numpy()[0,:,:,:]
                X_numpy = np.transpose(X_numpy, (1,2,0))
                plt.imsave(path+'/x_'+str(i)+'.png', np.clip(X_numpy,0,1), cmap = None)

            # Computation online of E[X] and E[X**2]
            if iter_mmse <= n_inter_mmse-1:
                xmmse = iter_mmse/(iter_mmse + 1)*xmmse + 1/(iter_mmse + 1)*X
                xmmse2 = iter_mmse/(iter_mmse + 1)*xmmse2 + 1/(iter_mmse + 1)*X**2
                iter_mmse += 1
            else:
                xmmse = iter_mmse/(iter_mmse + 1)*xmmse + 1/(iter_mmse + 1)*X
                xmmse2 = iter_mmse/(iter_mmse + 1)*xmmse2 + 1/(iter_mmse + 1)*X**2
                z = torch.squeeze(xmmse)
                z2 = torch.squeeze(xmmse2)
                Xlist_mmse.append(z)
                Xlist_mmse2.append(z2)
                del xmmse
                del xmmse2
                xmmse = torch.zeros(im_shape, dtype = dtype, device = device)
                xmmse2 = torch.zeros(im_shape, dtype = dtype, device = device)
                iter_mmse = 0
            # Saving the data on the disk during the process
            if i%K == 0 and save_images_online:
                #save the result of the experiment
                dict = {
                        'Samples' : Xlist,
                        'Mmse' : Xlist_mmse,
                        'Mmse2' : Xlist_mmse2,
                        'n_iter' : n_iter,
                        'c_min' : c_min,
                        'c_max' : c_max,
                        'lambda' : lambd,
                        'delta' : delta,
                    }
                torch.save(dict, path+'/'+ name +'_sampling.pth')

    return Xlist, Xlist_mmse, Xlist_mmse2


def psgla(init, data_grad, denoiser, alpha, lambd, sig_float = 0.0055, delta = 4e-5, n_iter = 5000, n_inter = 1000, n_inter_mmse = 1000, seed = None, device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu"), c_min = -1, c_max = 2, path = None, save_images_online = False, name = None):
    """
    Implementation of the Proximal Stochastic Gradient Langevin Algorithm (PSGLA) 

    Inputs:
        init        Initialization of the Markov chain (torch tensor)
        denoiser    Denoiser
        data_grad   Gradient of the likelihood
        delta       Discretization step-size (torch tensor)
        lambd       Moreau-Yoshida regularization parameter (torch tensor)
        n_iter      Number of ULA iterations
        n_inter     Number of iterations before saving of a sample
        n_inter_mmse Number of iterations for a mean computation
        device      cuda device used to store samples
        seed        int, seed used
        path        str : where to store the data
        name        name of the stored dictionnary
        c_min       To ensure strong convexity
        c_max       To ensure strong convexity
    Outputs:
        Xlist       Samples stored every n_inter iterations
        Xlist_mmse  Mmse computed over n_inter_mmse iterations
        Xlist_mmse2 Average X**2 computed over n_inter_mmse iterations
    """
    # Type
    dtype = torch.float32
    tensor = torch.FloatTensor
    # Shape of the image
    im_shape = init.shape
    # Markov chain init
    X = torch.zeros(im_shape, dtype = dtype, device = device)
    X = init.clone().detach()
    # To compute the empirical average over n_inter_mmse
    xmmse = torch.zeros(im_shape, dtype = dtype, device = device)
    # To compute the empirical variance over n_inter_mmse
    xmmse2 = torch.zeros(im_shape, dtype = dtype, device = device)
    # 
    One = torch.ones(xmmse2.shape)
    One = One.to(device)

    # Parameters setting
    delta_float = delta
    delta = torch.tensor(delta_float).to(device).to(torch.float32)
    sig_noised = sig_float
    sig = torch.tensor(sig_noised).to(device).to(torch.float32)
    
    print("delta = {}, sigma = {}".format(delta_float, sig_noised))
 
    if path == None:
        path = str()   
    if seed != None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed) #for reproductivity
    if name == None:
        name = str()
    if n_inter_mmse == None:
        n_inter_mmse = np.copy(n_inter)

    #To store results
    Xlist = []
    Xlist_mmse = []
    Xlist_mmse2 = []
    iter_mmse = 0

    # Frequency at which we save samples
    K = int(n_iter/10)
    sig_den = sig
    noise_ratio = torch.tensor(np.sqrt(2)).to(device).to(torch.float32)

    with torch.no_grad():
        for i in tqdm(range(n_iter)):
            Z = torch.randn(im_shape, generator=gen, dtype = dtype, device = device)
            # grad F : gaussian Data fit
            grad_log_data = data_grad(X) # T comment if we want to sample from the prior
            # Langevin update
            Y = X + noise_ratio*sig*Z #+ (delta/lambd)*grad_log_data + noise_ratio*sig*Z
            # Denoiser update
            X = (1- alpha) * Y + alpha * denoiser.forward(Y, sig_den)

            # To save samples of the Markov chain after the burn-in every n_inter iterations.
            if i%n_inter == 0:
                X_ = torch.squeeze(X)
                # Sample Storage
                Xlist.append(X_)

            if i % K == 0 and save_images_online:
                X_numpy = X.detach().cpu().numpy()[0,:,:,:]
                X_numpy = np.transpose(X_numpy, (1,2,0))
                # print(np.min(X_numpy), np.max(X_numpy))
                plt.imsave(path+'/x_'+str(i)+'.png', np.clip(X_numpy,0,1), cmap = None)
                Y_numpy = Y.detach().cpu().numpy()[0,:,:,:]
                Y_numpy = np.transpose(Y_numpy, (1,2,0))
                plt.imsave(path+'/y_'+str(i)+'.png', np.clip(Y_numpy,0,1), cmap = None)


            # Computation online of E[X] and E[X**2]
            if iter_mmse <= n_inter_mmse-1:
                xmmse = iter_mmse/(iter_mmse + 1)*xmmse + 1/(iter_mmse + 1)*X
                xmmse2 = iter_mmse/(iter_mmse + 1)*xmmse2 + 1/(iter_mmse + 1)*X**2
                iter_mmse += 1
            else:
                xmmse = iter_mmse/(iter_mmse + 1)*xmmse + 1/(iter_mmse + 1)*X
                xmmse2 = iter_mmse/(iter_mmse + 1)*xmmse2 + 1/(iter_mmse + 1)*X**2
                z = torch.squeeze(xmmse)
                z2 = torch.squeeze(xmmse2)
                Xlist_mmse.append(z)
                Xlist_mmse2.append(z2)
                del xmmse
                del xmmse2
                xmmse = torch.zeros(im_shape, dtype = dtype, device = device)
                xmmse2 = torch.zeros(im_shape, dtype = dtype, device = device)
                iter_mmse = 0
            # Saving the data on the disk during the process
            if i%K == 0 and save_images_online:
                #save the result of the experiment
                dict = {
                        'Samples' : Xlist,
                        'Mmse' : Xlist_mmse,
                        'Mmse2' : Xlist_mmse2,
                        'n_iter' : n_iter,
                        'c_min' : c_min,
                        'c_max' : c_max,
                        'lambda' : lambd,
                        'delta' : delta,
                    }
                torch.save(dict, path+'/'+ name +'_sampling.pth')

    return Xlist, Xlist_mmse, Xlist_mmse2

def diffpir(y, mask, device, denoiser, lambda_ = .13, t_start = 200, n_iter = 20, zeta = 0.8, sigma_noise = 10. / 255.0):
    T = 1000  # Number of timesteps used during training

    def get_alphas(beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=T):
        betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        betas = torch.from_numpy(betas).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)  # This is \overline{\alpha}_t
        return torch.tensor(alphas_cumprod)

    alphas = get_alphas()

    sigmas = torch.sqrt(1.0 - alphas) / alphas.sqrt()

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def data_fidelity_prox_inpainting(x, y, mask, stepsize):
        return (stepsize*mask*y + x)/(stepsize*mask+1)

    data_fidelity = L2()

    rhos = lambda_ * (sigma_noise**2) / (sigmas**2)

    # get timestep sequence
    seq = np.sqrt(np.linspace(0, t_start**2, n_iter))
    seq = [int(s) for s in list(seq)]
    seq[-1] = seq[-1] - 1

    # Initialization
    x = 2 * y - 1
    x = torch.clip(torch.sqrt(alphas[t_start]) * x + torch.sqrt(1 - alphas[t_start]) * torch.randn_like(x), 0, 1)

    X_list = [torch.squeeze(x)]

    with torch.no_grad():
        for i in tqdm(range(len(seq))):
            # Current and next noise levels
            curr_sigma = sigmas[t_start - 1 - seq[i]].cpu().numpy()

            # 1. Denoising step
            # print('sigma :', curr_sigma)
            # print("current psnr : {:.2f}dB".format(psnr(np.clip(tensor2array(x_true), 0, 1), np.clip(tensor2array(x), 0, 1))))
            # plt.imsave(exp_out_path+'img_x_'+str(i)+'input.png', single2uint(np.clip(tensor2array(x), 0, 1)))
            x0 = 2*denoiser.forward((x+1)/2, curr_sigma*1.)-1
            # plt.imsave(exp_out_path+'img_x_'+str(i)+'_den_input.png', single2uint(np.clip(tensor2array(x0), 0, 1)))
            # print("current psnr : {:.2f}dB".format(psnr(np.clip(tensor2array(x_true), 0, 1), np.clip(tensor2array(x0), 0, 1))))

            if not seq[i] == seq[-1]:
                # 2. Data fidelity step
                t_i = find_nearest(sigmas.cpu(), curr_sigma)

                x0 = data_fidelity_prox_inpainting(x0, y, mask, stepsize = 1 / (2 * rhos[t_i]))

                # Normalize data for sampling
                x0 = 2 * x0 - 1
                x = 2 * x - 1

                # 3. Sampling step
                next_sigma = sigmas[t_start - 1 - seq[i + 1]].cpu().numpy()
                t_im1 = find_nearest(
                    sigmas, next_sigma
                )  # time step associated with the next noise level

                eps = (x - alphas[t_i].sqrt() * x0) / torch.sqrt(
                    1.0 - alphas[t_i]
                )  # effective noise

                x = alphas[t_im1].sqrt() * x0 + torch.sqrt(1.0 - alphas[t_im1]) * (
                    np.sqrt(1 - zeta) * eps + np.sqrt(zeta) * torch.randn_like(x)
                )

                # Rescale the output in [0, 1]
                x = (x + 1) / 2
                X_list.append(torch.squeeze(x))

    return X_list, [torch.squeeze(x)], []


def pnp(init, data_grad, Pb, denoiser, alpha, lambd, sig_float = 0.0055, delta = 4e-5, n_iter = 5000, n_inter = 1000, n_inter_mmse = 1000, seed = None, device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu"), c_min = -1, c_max = 2, path = None, save_images_online = False, name = None):
    """
    Implementation of the Plug and Play algorithm, Forward-Backward scheme

    Inputs:
        init        Initialization of the Markov chain (torch tensor)
        denoiser    Denoiser
        data_grad   Gradient of the likelihood
        delta       Discretization step-size (torch tensor)
        lambd       Moreau-Yoshida regularization parameter (torch tensor)
        n_iter      Number of ULA iterations
        n_inter     Number of iterations before saving of a sample
        n_inter_mmse Number of iterations for a mean computation
        device      cuda device used to store samples
        seed        int, seed used
        path        str : where to store the data
        name        name of the stored dictionnary
        c_min       To ensure strong convexity
        c_max       To ensure strong convexity
    Outputs:
        Xlist       Samples stored every n_inter iterations
        Xlist_mmse  Mmse computed over n_inter_mmse iterations
        Xlist_mmse2 Average X**2 computed over n_inter_mmse iterations
    """
    # Type
    dtype = torch.float32
    tensor = torch.FloatTensor
    # Shape of the image
    im_shape = init.shape
    # Markov chain init
    X = torch.zeros(im_shape, dtype = dtype, device = device)
    X = init.clone().detach()
    # To compute the empirical average over n_inter_mmse
    xmmse = torch.zeros(im_shape, dtype = dtype, device = device)
    # To compute the empirical variance over n_inter_mmse
    xmmse2 = torch.zeros(im_shape, dtype = dtype, device = device)
    # 
    One = torch.ones(xmmse2.shape)
    One = One.to(device)

    # Parameters setting
    delta_float = delta
    delta = torch.tensor(delta_float).to(device).to(torch.float32)
    sig_noised = sig_float
    sig = torch.tensor(sig_noised).to(device).to(torch.float32)
    
    print("delta = {}, sigma = {}".format(delta_float, sig_noised))
 
    if path == None:
        path = str()   
    if seed != None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed) #for reproductivity
    if name == None:
        name = str()
    if n_inter_mmse == None:
        n_inter_mmse = np.copy(n_inter)

    #To store results
    Xlist = []

    # Frequency at which we save samples
    K = int(n_iter/10)

    with torch.no_grad():
        for i in tqdm(range(n_iter)):
            if Pb == 'inpainting' and i < n_iter // 10:
                sig_den = 40. / 255.
            else:
                sig_den = sig
            grad_log_data = data_grad(X) # T comment if we want to sample from the prior
            Y = X + (delta/lambd)*grad_log_data
            # Denoiser update
            X = (1- alpha) * Y + alpha * denoiser.forward(Y, sig_den)

            if i % K == 0 and save_images_online:
                X_numpy = X.detach().cpu().numpy()[0,:,:,:]
                X_numpy = np.transpose(X_numpy, (1,2,0))
                # print(np.min(X_numpy), np.max(X_numpy))
                plt.imsave(path+'/x_'+str(i)+'.png', np.clip(X_numpy,0,1), cmap = None)
                Y_numpy = Y.detach().cpu().numpy()[0,:,:,:]
                Y_numpy = np.transpose(Y_numpy, (1,2,0))
                plt.imsave(path+'/y_'+str(i)+'.png', np.clip(Y_numpy,0,1), cmap = None)
            Xlist.append(torch.squeeze(X))

    return Xlist, [torch.squeeze(X)], []

def red(init, data_grad, Pb, denoiser, alpha, lambd, sig_float = 0.0055, delta = 4e-5, n_iter = 5000, n_inter = 1000, n_inter_mmse = 1000, seed = None, device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu"), c_min = -1, c_max = 2, path = None, save_images_online = False, name = None):
    """
    Implementation of the Regularization by Denoising algorithm

    Inputs:
        init        Initialization of the Markov chain (torch tensor)
        denoiser    Denoiser
        data_grad   Gradient of the likelihood
        delta       Discretization step-size (torch tensor)
        lambd       Moreau-Yoshida regularization parameter (torch tensor)
        n_iter      Number of ULA iterations
        n_inter     Number of iterations before saving of a sample
        n_inter_mmse Number of iterations for a mean computation
        device      cuda device used to store samples
        seed        int, seed used
        path        str : where to store the data
        name        name of the stored dictionnary
        c_min       To ensure strong convexity
        c_max       To ensure strong convexity
    Outputs:
        Xlist       Samples stored every n_inter iterations
        Xlist_mmse  Mmse computed over n_inter_mmse iterations
        Xlist_mmse2 Average X**2 computed over n_inter_mmse iterations
    """
    # Type
    dtype = torch.float32
    tensor = torch.FloatTensor
    # Shape of the image
    im_shape = init.shape
    # Markov chain init
    X = torch.zeros(im_shape, dtype = dtype, device = device)
    X = init.clone().detach()

    # Parameters setting
    delta_float = delta
    delta = torch.tensor(delta_float).to(device).to(torch.float32)
    sig_noised = sig_float
    sig = torch.tensor(sig_noised).to(device).to(torch.float32)
    
    print("delta = {}, sigma = {}".format(delta_float, sig_noised))
 
    #To store results
    Xlist = []
    # Frequency at which we save samples
    K = int(n_iter/10)
    sig_den = sig

    with torch.no_grad():
        for i in tqdm(range(n_iter)):
            if i < 10 and Pb == 'inpainting':
                sig_den = 50./255.
            else:
                sig_den = sig

            grad_log_data = data_grad(X) # T comment if we want to sample from the prior
            X = X + delta*grad_log_data - delta * lambd * (X - denoiser.forward(X, sig_den))

            X_ = torch.squeeze(X)
            Xlist.append(X_)

            if i % K == 0 and save_images_online:
                X_numpy = X.detach().cpu().numpy()[0,:,:,:]
                X_numpy = np.transpose(X_numpy, (1,2,0))
                # print(np.min(X_numpy), np.max(X_numpy))
                plt.imsave(path+'/x_'+str(i)+'.png', np.clip(X_numpy,0,1), cmap = None)

    return Xlist, [torch.squeeze(X)], []
