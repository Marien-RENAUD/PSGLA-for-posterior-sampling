import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
from lpips import LPIPS
from argparse import ArgumentParser
import os
import argparse
import cv2
import imageio
from utils_GMM_2D import *
from utils import array2tensor


parser = ArgumentParser()
parser.add_argument('--fig_number', type=int, default=-1)
parser.add_argument('--table_number', type=int, default=-1)
pars = parser.parse_args()

path_figure = "results/figure/"
path_result = "results/result_GMM/"

if pars.fig_number == 0:
    fig = plt.figure(figsize = (20, 10))
    gs = gridspec.GridSpec(2, 4, wspace=0.1, hspace=0.15)
    
    label_size = 20
    alpha_scatter = 0.1
    
    Wass_dist_list_ULA = []
    Wass_dist_list_PSGLA = []

    name_list = ['symetric_gaussians', 'disymmetric_gaussians', 'cross']
    N = 10000
    for i in range(3):
        dict = np.load(path_result + "Sample_PnP_SnoPnP_ULA_"+name_list[i]+"_N"+str(N)+"_result.npy", allow_pickle=True).item()
        A = dict['A'] ; Y = dict['Y'] ; sigma = dict['sigma'] ; mu_list = dict['mu_list'] ; sigma_list = dict['sigma_list'] ; pi_list = dict['pi_list']
        
        Sample_ULA = dict["Sample_PnP_ULA"] ; Wass_PnP_ULA = dict['Wass_PnP_ULA'] ; MMSE_PnP_ULA = dict['MMSE_PnP_ULA']
        Wass_dist_list_ULA.append(dict["Wass_dist_ULA_list"][i])
        Wass_dist_list_PSGLA.append(dict["Wass_dist_PSGLA_list"][i])

        ax2 = plt.subplot(gs[0,i])
        sample_ULA = Sample_ULA[i]
        sample_ULA = np.random.permutation(sample_ULA)[:1000] #take a sample of 1000 points
        y = Y[i]
        mu_cond_list, sigma_cond_list, p_list = constantes_conditionnal_prob(A, y, sigma, mu_list, sigma_list, pi_list)
        alpha_list = Alpha(p_list)
        draw_gaussian_mixture(ax2, mu_cond_list, sigma_cond_list, alpha_list, rbox = 5, color = 'k',  label = r'$p(x|y)$', linewidth = 2)
        # X_0, X_1 = np.mgrid[-10:10:150j, -10:10:150j]
        # positions = np.vstack([X_0.ravel(), X_1.ravel()])
        # posterior_density = gaussian_mixture_density(positions, mu_cond_list, sigma_cond_list, p_list)
        # Z_post = np.reshape(posterior_density.T, X_0.shape)
        # Z_post = np.log(Z_post / np.sum(Z_post) + 1e-6)
        # ax2.pcolormesh(X_0, X_1, Z_post, cmap='gray',  label = 'x_data|y')
        ax2.scatter(sample_ULA[:, 0], sample_ULA[:, 1], alpha=alpha_scatter, c='g', label = r'$X_k$') #sample of PnP-ULA
        draw_gaussian_mixture(ax2, mu_list, sigma_list, pi_list, rbox = 9, color = 'b',  label = r'$p(x)$', linewidth = 2)
        ax2.scatter(y[0], y[1], c='r', label = 'y')
        ax2.grid(False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("Wasserstein distance = {:.2f} \n Density distance = {:.3f}".format(Wass_PnP_ULA[i], MMSE_PnP_ULA[i]))
        if i == 0:
            ax2.set_ylabel('PnP-ULA', fontsize = label_size)
            ax2.legend(loc = 'upper left')

            x_0, x_1, width, height = -8.4, 7.8, 1.4, 0.5  # Position et taille du carré

            # Définir le nombre de segments pour le dégradé horizontal
            n_steps = 100  # Nombre de segments du dégradé

            # Créer un dégradé de couleur horizontal (du blanc au noir)
            for j in range(n_steps):
                t = (n_steps-1-j) / ((1+0.3)*n_steps - 1)
                t_min = 0
                t_max = (n_steps - 1) / ((1+0.3)*n_steps - 1)
                color = (t, t, t)
                # upper part
                ax2.plot([x_0 + (j/n_steps) * width, x_0 + ((j+1)/n_steps) * width], 
                        [x_1, x_1], color=color, lw=2, zorder=10)
                # bottom part
                ax2.plot([x_0 + (j/n_steps) * width, x_0 + ((j+1)/n_steps) * width], 
                        [x_1 + height, x_1 + height], color=color, lw=2, zorder=10)
                # left part
                ax2.plot([x_0, x_0], 
                        [x_1 + (j/n_steps) * height, x_1 + ((j+1)/n_steps) * height], color=(t_max,t_max,t_max), lw=2, zorder=10)
                # right part
                ax2.plot([x_0 + width, x_0 + width], 
                        [x_1 + (j/n_steps) * height, x_1 + ((j+1)/n_steps) * height], color=(t_min,t_min,t_min), lw=2, zorder=10)


        
        Sample_SnoPnP_ULA = dict["Sample_SnoPnP_ULA"] ; Wass_SnoPnP_ULA = dict['Wass_SnoPnP_ULA'] ; MMSE_SnoPnP_ULA = dict['MMSE_SnoPnP_ULA']
        ax2 = plt.subplot(gs[1,i])
        sample_SnoPnP_ULA = Sample_SnoPnP_ULA[i]
        sample_SnoPnP_ULA = np.random.permutation(sample_SnoPnP_ULA)[:1000] #take a sample of 1000 points
        draw_gaussian_mixture(ax2, mu_cond_list, sigma_cond_list, alpha_list, rbox = 5, color = 'k',  label = 'x_data|y', linewidth = 2)
        # ax2.pcolormesh(X_0, X_1, Z_post, cmap='gray',  label = 'x_data|y')
        ax2.scatter(sample_SnoPnP_ULA[:, 0], sample_SnoPnP_ULA[:, 1], alpha=alpha_scatter, c='g', label = 'x_data|y sample') #sample of PnP-ULA
        draw_gaussian_mixture(ax2, mu_list, sigma_list, pi_list, rbox = 9, color = 'b',  label = 'x_data', linewidth = 2)
        ax2.scatter(y[0], y[1], c='r', label = 'y')
        ax2.grid(False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("Wasserstein distance = {:.2f} \n Density distance = {:.3f}".format(Wass_SnoPnP_ULA[i], MMSE_SnoPnP_ULA[i]))
        if i == 0:
            ax2.set_ylabel('PnP-PSGLA', fontsize = label_size)
    

    Wass_dist_list_ULA = np.array(Wass_dist_list_ULA)
    Wass_dist_list_PSGLA = np.array(Wass_dist_list_PSGLA)
    Wass_dist_list_ULA = np.mean(Wass_dist_list_ULA, axis = 0)
    Wass_dist_list_PSGLA = np.mean(Wass_dist_list_PSGLA, axis = 0)
    X = np.linspace(0, N, len(Wass_dist_list_ULA))

    text_size = 12

    ax2 = plt.subplot(gs[0,-1])
    ax2.semilogx(X, Wass_dist_list_ULA)
    ax2.set_title("Mean Wasserstein distance")
    ax2.set_ylim(0,15)
    ax2.set_yticks([0, 15])
    ax2.set_yticklabels([r'$0$', r'$15$'], fontsize=text_size)
    ax2.set_xticks([100, 1000, 10000])
    ax2.set_xticklabels([r'$10^2$', r'$10^3$', r'$10^4$'], fontsize=text_size)

    ax2 = plt.subplot(gs[1,-1])
    ax2.semilogx(X, Wass_dist_list_PSGLA)
    ax2.set_ylim(0,15)
    ax2.set_yticks([0, 15])
    ax2.set_yticklabels([r'$0$', r'15'], fontsize=text_size)
    ax2.set_xticks([100, 1000, 10000])
    ax2.set_xticklabels([r'$10^2$', r'$10^3$', r'$10^4$'], fontsize=text_size)

    fig.savefig(path_figure+"/figure_paper_GMM_2D"+str(N)+".png", dpi = 300)


if pars.table_number == 0:
    #generate the result of inpainting for table of result of PSGLA, PnP-ULA, DiffPIR on CBSD10 dataset.

    path_result = "results/result/inpainting/CBSD10/"
    n = 10
    output_psnr = [[],[],[],[],[]]
    output_ssim = [[],[],[],[],[]]
    for i in tqdm(range(n)):
        dic_PnP_ULA = np.load(path_result + "/pnp_ula/DnCNN/s_2.0/n_iter_100000/im_"+str(i)+"/sigma1_s2.0_result.npy", allow_pickle=True).item()
        dic_PSGLA_DnCNN = np.load(path_result + "/psgla/DnCNN/s_2.0/lambd_5.0/im_"+str(i)+"/sigma1_s2.0_result.npy", allow_pickle=True).item()
        dic_PSGLA_TV = np.load(path_result + "/psgla/TV/s_10.0/lambd_10.0/n_iter_1000/im_"+str(i)+"/sigma1_s10.0_result.npy", allow_pickle=True).item()
        dic_DiffPIR = np.load(path_result + "/diffpir/GSDRUNet/lambd_0.05/zeta_0.999/im_"+str(i)+"/sigma1_s5.0_result.npy", allow_pickle=True).item()
        dic_PnP = np.load(path_result + "/pnp/GSDRUNet/s_10.0/delta_1e-05/n_iter_500/im_"+str(i)+"/sigma1_s10.0_result.npy", allow_pickle=True).item()

        output_psnr[0].append(dic_PnP_ULA["PSNR_MMSE"])
        output_psnr[1].append(dic_PSGLA_DnCNN["PSNR_MMSE"])
        output_psnr[2].append(dic_PSGLA_TV["PSNR_MMSE"])
        output_psnr[3].append(dic_DiffPIR["PSNR_MMSE"])
        output_psnr[4].append(dic_PnP["PSNR_MMSE"])
        output_ssim[0].append(dic_PnP_ULA["SIM_MMSE"])
        output_ssim[1].append(dic_PSGLA_DnCNN["SIM_MMSE"])
        output_ssim[2].append(dic_PSGLA_TV["SIM_MMSE"])
        output_ssim[3].append(dic_DiffPIR["SIM_MMSE"])
        output_ssim[4].append(dic_PnP["SIM_MMSE"])

    output_psnr = np.array(output_psnr)
    output_ssim = np.array(output_ssim)
    print("Method & PSNR & SSIM & N")
    print("PnP ULA & {:.2f} & {:.2f} & {}".format(np.mean(output_psnr[0]), np.mean(output_ssim[0]), dic_PnP_ULA['n_iter']))
    print("PSGLA DnCNN & {:.2f} & {:.2f} & {}".format(np.mean(output_psnr[1]), np.mean(output_ssim[1]), dic_PSGLA_DnCNN['n_iter']))
    print("PSGLA TV & {:.2f} & {:.2f} & {}".format(np.mean(output_psnr[2]), np.mean(output_ssim[2]), dic_PSGLA_TV['n_iter']))
    print("DiffPIR & {:.2f} & {:.2f} & {}".format(np.mean(output_psnr[3]), np.mean(output_ssim[3]), dic_DiffPIR['n_iter']))
    print("PnP & {:.2f} & {:.2f} & {}".format(np.mean(output_psnr[4]), np.mean(output_ssim[4]), dic_PnP['n_iter']))


if pars.table_number == 1:
    #generate the result of inpainting for table of result of PSGLA, PnP-ULA, DiffPIR on CBSD68 dataset.

    if torch.cuda.is_available():# LPIPS metric computation
        device = "cuda:0"
    else:
        device = "cpu"
    loss_lpips = LPIPS(net='alex', version='0.1').to(device)

    path_result = "results/result/inpainting/CBSD68/"
    n = 68
    method_name = ["RED", "RED_DnCNN"]#"DiffPIR", "PnP", "PnP DnCNN", "RED", "RED_DnCNN", "PSGLA DnCNN", "PSGLA TV"]
    m = len(method_name)
    output_psnr = [[] for _ in range(m)]
    output_ssim = [[] for _ in range(m)]
    output_lpips = [[] for _ in range(m)]
    for i in tqdm(range(n)):
        # dic_DiffPIR = np.load(path_result + "diffpir/GSDRUNet/lambd_0.05/zeta_0.999/im_"+str(i)+"/sigma1_s5.0_result.npy", allow_pickle=True).item()
        # dic_PnP = np.load(path_result + "pnp/GSDRUNet/s_5.0/delta_1e-05/lambd_0.5/n_iter_500/im_"+str(i)+"/sigma1_s5.0_result.npy", allow_pickle=True).item()
        # dic_PnP_DnCNN = np.load(path_result + "pnp/DnCNN/s_2.0/delta_1e-05/n_iter_500/im_"+str(i)+"/sigma1_s2.0_result.npy", allow_pickle=True).item()
        dic_RED = np.load(path_result + "red/GSDRUNet/s_7.0/delta_1e-05/lambd_70000.0/n_iter_500/im_"+str(i)+"/sigma1_s7.0_result.npy", allow_pickle=True).item()
        dic_RED_DnCNN = np.load(path_result + "red/DnCNN/s_2.0/delta_1e-05/lambd_150000.0/n_iter_500/im_"+str(i)+"/sigma1_s2.0_result.npy", allow_pickle=True).item()
        # dic_PnP_ULA = np.load(path_result + "pnp_ula/DnCNN/s_2.0/n_iter_100000/im_"+str(i)+"/sigma1_s2.0_result.npy", allow_pickle=True).item()
        # dic_PSGLA_DnCNN = np.load(path_result + "psgla/DnCNN/s_2.0/lambd_5.0/im_"+str(i)+"/sigma1_s2.0_result.npy", allow_pickle=True).item()
        # dic_PSGLA_TV = np.load(path_result + "psgla/TV/s_10.0/lambd_10.0/n_iter_1000/im_"+str(i)+"/sigma1_s10.0_result.npy", allow_pickle=True).item()

        dic_list = [dic_RED, dic_RED_DnCNN]#dic_DiffPIR, dic_PnP, dic_PnP_DnCNN, dic_RED, dic_PnP_ULA, dic_PSGLA_DnCNN, dic_PSGLA_TV]

        for j in range(m):
            output_psnr[j].append(dic_list[j]["PSNR_MMSE"])
            output_ssim[j].append(dic_list[j]["SIM_MMSE"])
            output_lpips[j].append(loss_lpips.forward(array2tensor(dic_list[j]["MMSE"]).to(device), array2tensor(dic_list[j]["ground_truth"]).to(device)).item())

    output_psnr = np.array(output_psnr)
    output_ssim = np.array(output_ssim)
    print("Method & PSNR & SSIM & LPIPS & N")
    for j in range(m):
        print(method_name[j] + " & {:.2f} & {:.2f} & {:.2f} & {}".format(np.mean(output_psnr[j]), np.mean(output_ssim[j]), np.mean(output_lpips[j]), dic_list[j]['n_iter']))


if pars.table_number == 2:
    #Give best parameters for Grid Search on a method.
    lambda_list = [10000.0, 30000.0, 50000.0, 70000.0, 100000.0, 150000.0, 200000.0]
    s_list = [2.0, 5.0, 7.0]

    path_result = "results/result/inpainting/set3c/red/DnCNN/"
    n = 3
    for k in range(len(s_list)):
        for l in range(len(lambda_list)):
            mean_psnr = 0
            for i in tqdm(range(n)):
                dic_PnP = np.load(path_result + "s_"+str(s_list[k])+"/delta_1e-05/lambd_"+str(lambda_list[l])+"/n_iter_500/im_"+str(i)+"/sigma1_s"+str(s_list[k])+"_result.npy", allow_pickle=True).item()
                mean_psnr += dic_PnP["PSNR_MMSE"]
            print("sigma = {}, lambda = {} : {:.2f} dB".format(s_list[k], lambda_list[l], mean_psnr/n))



if pars.fig_number == 1:
    #generate figure for inpainting qualitative comparison in the paper on the castle image.
    path_result = "results/result/inpainting/set1c/"

    name_img = "0"
    name_fig_list = ["Observation", "DiffPIR", "PnP-ULA", "PnP-PSGLA TV", "PnP-PSGLA DnCNN"]

    n = 1
    m = 6

    #size of the black rectangle
    height = 35
    width = 150
    indices = [i for i in range(m)]
    
    fig = plt.figure(figsize = (m*5.2, n*7.44))
    gs = gridspec.GridSpec(n, m, hspace = 0.2, wspace = 0)

    text_size = 30
    label_size = 25

    width = 230
    dic_DiffPIR = np.load(path_result + "diffpir/GSDRUNet/lambd_0.05/zeta_0.999/im_0/sigma1_s5.0_result.npy", allow_pickle=True).item()
    dic_PnP_ULA = np.load(path_result + "/pnp_ula/DnCNN/s_2.0/n_iter_1000000/im_0/sigma1_s2.0_result.npy", allow_pickle=True).item()
    dic_PSGLA_DnCNN = np.load(path_result + "psgla/DnCNN/s_2.0/lambd_5.0/im_0/sigma1_s2.0_result.npy", allow_pickle=True).item()
    dic_PSGLA_TV = np.load(path_result + "psgla/TV/s_10.0/lambd_10.0/n_iter_1000/im_0/sigma1_s10.0_result.npy", allow_pickle=True).item()

    gt = dic_DiffPIR["ground_truth"]
    degraded = (dic_DiffPIR["observation"], dic_DiffPIR["PSNR_y"], dic_DiffPIR["SIM_y"])
    restore_DiffPIR = (dic_DiffPIR["MMSE"], dic_DiffPIR["PSNR_MMSE"], dic_DiffPIR["SIM_MMSE"])
    restore_PnP_ULA = (dic_PnP_ULA["MMSE"], dic_PnP_ULA["PSNR_MMSE"], dic_PnP_ULA["SIM_MMSE"])
    restore_PSGLA_DnCNN = (dic_PSGLA_DnCNN["MMSE"], dic_PSGLA_DnCNN["PSNR_MMSE"], dic_PSGLA_DnCNN["SIM_MMSE"])
    restore_PSGLA_TV = (dic_PSGLA_TV["MMSE"], dic_PSGLA_TV["PSNR_MMSE"], dic_PSGLA_TV["SIM_MMSE"])

    c = 140
    c_kernel = 100
    wid, hei = 70, 70
    x_c, y_c = 140, 110
    ax = plt.subplot(gs[indices[0]])

    #add a zoom of the image
    patch_c = cv2.resize(gt[y_c:y_c+hei, x_c:x_c+wid], dsize =(c,c), interpolation=cv2.INTER_CUBIC)
    gt[-patch_c.shape[0]:,:patch_c.shape[1]] = patch_c
    rect_params_z = {'xy': (0, gt.shape[0]-patch_c.shape[0]-1), 'width': c, 'height': c, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
    ax.add_patch(plt.Rectangle(**rect_params_z))

    ax.imshow(gt.astype(np.float32))
    rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
    ax.add_patch(plt.Rectangle(**rect_params))
    text_params = {'xy': (5, 5), 'text': r"PSNR$\uparrow$SSIM$\uparrow$", 'color': 'white', 'fontsize': label_size, 'va': 'top', 'ha': 'left'}
    ax.annotate(**text_params)
    
    #add a color rectangle
    rect_params_c = {'xy': (x_c, y_c), 'width': wid, 'height': hei, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
    ax.add_patch(plt.Rectangle(**rect_params_c))

    ax.axis('off')
    ax.set_title("Ground Truth", fontsize=text_size)

    width = 170

    for j, im in enumerate([degraded, restore_DiffPIR, restore_PnP_ULA, restore_PSGLA_TV, restore_PSGLA_DnCNN]):
        ax = plt.subplot(gs[indices[1+j]])
        
        #add a zoom of the image
        patch_c = cv2.resize(im[0][y_c:y_c+hei, x_c:x_c+wid], dsize =(c,c), interpolation=cv2.INTER_CUBIC)
        im[0][-patch_c.shape[0]:,:patch_c.shape[1]] = patch_c
        rect_params_z = {'xy': (0, im[0].shape[0]-patch_c.shape[0]-1), 'width': c, 'height': c, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
        ax.add_patch(plt.Rectangle(**rect_params_z))
            
        ax.imshow(im[0].astype(np.float32))
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}".format(im[1], im[2]), 'color': 'white', 'fontsize': label_size, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)

        #add a color rectangle
        rect_params_c = {'xy': (x_c, y_c), 'width': wid, 'height': hei, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
        ax.add_patch(plt.Rectangle(**rect_params_c))

        ax.axis('off')
        ax.set_title(name_fig_list[j], fontsize=text_size)

    ax = plt.subplot(gs[indices[-1]])
    
    fig.savefig(path_figure+'/Results_restoration_inpainting_various_methods.png', dpi = 300)
    plt.show()


if pars.fig_number == 2:
    #generate figure to exclude the Prox_DRUNet as a candidat for PSGLA.
    path_result = "results/result/inpainting/set1c/psgla/Prox_DRUNet/s_50.0/lambd_5000.0/"

    name_fig_list = ["Observation", "Seed 1", "Seed 2", "Std Seed 1", "Std Seed 2"]

    n = 1
    m = 6

    #size of the black rectangle
    height = 35
    width = 150
    indices = [i for i in range(m)]
    
    fig = plt.figure(figsize = (m*5.2, n*7.44))
    gs = gridspec.GridSpec(n, m, hspace = 0.2, wspace = 0)

    text_size = 25
    label_size = 25

    width = 230
    dic_seed_1 = np.load(path_result + "seed_alg_0/im_0/sigma1_s50.0_result.npy", allow_pickle=True).item()
    dic_seed_2 = np.load(path_result + "seed_alg_1/im_0/sigma1_s50.0_result.npy", allow_pickle=True).item()
    
    gt = dic_seed_1["ground_truth"]
    degraded = (dic_seed_1["observation"], dic_seed_1["PSNR_y"], dic_seed_1["SIM_y"])
    restore_seed_1 = (dic_seed_1["MMSE"], dic_seed_1["PSNR_MMSE"], dic_seed_1["SIM_MMSE"])
    restore_seed_2 = (dic_seed_2["MMSE"], dic_seed_2["PSNR_MMSE"], dic_seed_2["SIM_MMSE"])
    std_seed_1 = dic_seed_1["std"]
    std_seed_2 = dic_seed_2["std"]

    c_kernel = 100
    wid, hei = 70, 70
    ax = plt.subplot(gs[indices[0]])

    ax.imshow(gt.astype(np.float32))
    rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
    ax.add_patch(plt.Rectangle(**rect_params))
    text_params = {'xy': (5, 5), 'text': r"PSNR$\uparrow$SSIM$\uparrow$", 'color': 'white', 'fontsize': label_size, 'va': 'top', 'ha': 'left'}
    ax.annotate(**text_params)

    ax.axis('off')
    ax.set_title("Ground Truth", fontsize=text_size)

    width = 170

    for j, im in enumerate([degraded, restore_seed_1, restore_seed_2]):
        ax = plt.subplot(gs[indices[1+j]])
        ax.imshow(im[0].astype(np.float32))
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}".format(im[1], im[2]), 'color': 'white', 'fontsize': label_size, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title(name_fig_list[j]+ "\n range = [{:.2f}, {:.2f}]".format(np.min(im[0]), np.max(im[0])), fontsize=text_size)

    for k, im in enumerate([std_seed_1, std_seed_2]):
        im = np.sum(im, axis = -1)
        ax = plt.subplot(gs[indices[4+k]])
        ax.imshow(((im - np.min(im))/(np.max(im) - np.min(im))).astype(np.float32), cmap = 'gray')
        ax.set_title(name_fig_list[3+k] + "\n range = [{:.2f}, {:.2f}]".format(np.min(im), np.max(im)), fontsize=text_size)
        ax.axis('off')
    
    fig.savefig(path_figure+'/Restoration_PSGLA_with_Prox_DRUNet.png')
    plt.show()



if pars.fig_number == 3:
    # Generate a figure to show various result on image inpainting for Appendix
    path_result = "results/result/inpainting/CBSD68"

    n = 7
    m = 5

    size_title = 40
    size_label = 25

    #size of the black rectangle
    height = 30
    width = 330
    indx_list = [0,2,3,4,27,58,66]
    fig = plt.figure(figsize = (m*7.44, n*5))
    gs = gridspec.GridSpec(n, m, hspace = 0, wspace = 0)
    for i in range(n):
        im_indx = indx_list[i]
        dic_PSGLA = np.load(path_result + "/psgla/DnCNN/s_2.0/lambd_5.0/im_"+str(im_indx)+"/sigma1_s2.0_result.npy", allow_pickle=True).item()
        dic_PnP = np.load(path_result + "/pnp/GSDRUNet/s_5.0/delta_1e-05/lambd_0.5/n_iter_500/im_"+str(im_indx)+"/sigma1_s5.0_result.npy", allow_pickle=True).item()
        dic_diffPIR = np.load(path_result + "/diffpir/GSDRUNet/lambd_0.05/zeta_0.999/im_"+str(im_indx)+"/sigma1_s5.0_result.npy", allow_pickle=True).item()

        PSGLA = (dic_PSGLA["MMSE"], dic_PSGLA["PSNR_MMSE"], dic_PSGLA["SIM_MMSE"])
        PnP = (dic_PnP["MMSE"], dic_PnP["PSNR_MMSE"], dic_PnP["SIM_MMSE"])
        diffPIR = (dic_diffPIR["MMSE"], dic_diffPIR["PSNR_MMSE"], dic_diffPIR["SIM_MMSE"])
        GT = dic_PSGLA["ground_truth"]
        Obs = (dic_PSGLA["observation"], dic_PSGLA["PSNR_y"], dic_PSGLA["SIM_y"])

        im_list = [Obs, PnP, diffPIR, PSGLA]
        name_list = ["Observation", "PnP", 'DiffPIR', 'PnP-PSGLA']

        c_patch = 120
        wid, hei = 70, 70
        if im_indx == 0:
            x_c, y_c = 360, 130
        elif im_indx== 1:
            x_c, y_c = 230, 180
        elif im_indx== 2:
            x_c, y_c = 60, 40
        elif im_indx== 3:
            x_c, y_c = 130, 240
        elif im_indx==4:
            x_c, y_c = 120, 150
        elif im_indx==5:
            x_c, y_c = 190, 140
        elif im_indx==58:
            x_c, y_c = 80, 140
        elif im_indx==66:
            x_c, y_c = 170, 100
        else:
            x_c, y_c = 240, 90

        ax = plt.subplot(gs[i, 0])
        ax.imshow(GT.astype(np.float32))
        ax.axis('off')
        if i ==0:
            ax.set_title("Ground Truth", fontsize=size_title)
            width = 200
            rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 0, 'edgecolor': 'black', 'facecolor': 'black'}
            ax.add_patch(plt.Rectangle(**rect_params))
            text_params = {'xy': (5, 5), 'text': r"PSNR$\uparrow$SSIM$\uparrow$", 'color': 'white', 'fontsize': 22, 'va': 'top', 'ha': 'left'}
            ax.annotate(**text_params)
            width = 170

        #     #add a zoom of the Ground-Truth image
        #     patch_c = cv2.resize(GT[y_c:y_c+hei, x_c:x_c+wid], dsize =(c_patch,c_patch), interpolation=cv2.INTER_CUBIC)
        #     GT[-patch_c.shape[0]:,:patch_c.shape[1]] = patch_c
        #     #add a color line around the corner area
        #     rect_params_z = {'xy': (0, GT.shape[0]-patch_c.shape[0]-1), 'width': c_patch, 'height': c_patch, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
        #     ax.add_patch(plt.Rectangle(**rect_params_z))
        #     #add a color rectangle around on the zoomed area
        #     rect_params_c = {'xy': (x_c, y_c), 'width': wid, 'height': hei, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
        #     ax.add_patch(plt.Rectangle(**rect_params_c))
        #     ax.imshow(GT.astype(np.float32))
        #     ax.axis('off')

        # if i > 0:
        #add a zoom of the Ground-Truth image
        patch_c = cv2.resize(GT[y_c:y_c+hei, x_c:x_c+wid], dsize =(c_patch,c_patch), interpolation=cv2.INTER_CUBIC)
        GT[:patch_c.shape[0],-patch_c.shape[1]:] = patch_c
        #add a color line around the corner area
        rect_params_z = {'xy': (GT.shape[1]-patch_c.shape[1]-1, 0), 'width': c_patch, 'height': c_patch, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
        ax.add_patch(plt.Rectangle(**rect_params_z))
        #add a color rectangle around on the zoomed area
        rect_params_c = {'xy': (x_c, y_c), 'width': wid, 'height': hei, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
        ax.add_patch(plt.Rectangle(**rect_params_c))
        ax.imshow(GT.astype(np.float32))
        ax.axis('off')


        for j, im in enumerate(im_list):
            ax = plt.subplot(gs[i, 1+j])            
            #add a zoom of the image
            patch_c = cv2.resize(im[0][y_c:y_c+hei, x_c:x_c+wid], dsize =(c_patch,c_patch), interpolation=cv2.INTER_CUBIC)
            im[0][:patch_c.shape[0],-patch_c.shape[1]:] = patch_c
            #add a color line around the corner area
            rect_params_z = {'xy': (im[0].shape[1]-patch_c.shape[1]-1, 0), 'width': c_patch, 'height': c_patch, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
            ax.add_patch(plt.Rectangle(**rect_params_z))
            #add a color rectangle around on the zoomed area
            rect_params_c = {'xy': (x_c, y_c), 'width': wid, 'height': hei, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
            ax.add_patch(plt.Rectangle(**rect_params_c))
            
            ax.imshow(im[0].astype(np.float32))
            rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
            ax.add_patch(plt.Rectangle(**rect_params))
            text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}".format(im[1], im[2]), 'color': 'white', 'fontsize': size_label, 'va': 'top', 'ha': 'left'}
            ax.annotate(**text_params)
            if i ==0:
                ax.set_title(name_list[j], fontsize=size_title)
            ax.axis('off')
            
    fig.savefig(path_figure+'/set_of_results_impainting_appendix.png', dpi = 300)
    plt.show()


if pars.fig_number == 4:
    #generate the first line of the figure to explained the mixing time of PnP ULA.
    path_result = "results/result/inpainting/set1c/pnp_ula/DnCNN/s_2.0/"

    name_fig_list = [r"Restored $N = 10^4$", r"Restored $N = 10^5$", r"Restored $N = 10^6$", r"Std $N = 10^4$", r"Std $N = 10^5$", r"Std $N = 10^6$"]

    n = 2
    m = len(name_fig_list) + 3
    size_space = 0.15
    vertical_space = 0.1

    #size of the black rectangle
    height = 35
    width = 150
    indices = [(0,i) for i in range(m)]
    
    fig = plt.figure(figsize = ((m-1 + size_space)*5.2, (n+vertical_space)*7.7))
    gs = gridspec.GridSpec(n, m, hspace = vertical_space, wspace = 0, width_ratios=[1, 1, 1, 1, 1, 1, 1, size_space, 1])

    text_size = 26
    label_size = 25

    dic_pnp_ula_1 = np.load(path_result + "n_iter_10000/im_0/sigma1_s2.0_result.npy", allow_pickle=True).item()
    dic_pnp_ula_2 = np.load(path_result + "n_iter_100000/im_0/sigma1_s2.0_result.npy", allow_pickle=True).item()
    dic_pnp_ula_3 = np.load(path_result + "n_iter_1000000/im_0/sigma1_s2.0_result.npy", allow_pickle=True).item()
    
    gt = dic_pnp_ula_1["ground_truth"]
    degraded = (dic_pnp_ula_1["observation"], dic_pnp_ula_1["PSNR_y"], dic_pnp_ula_1["SIM_y"])
    restore_1 = (dic_pnp_ula_1["MMSE"], dic_pnp_ula_1["PSNR_MMSE"], dic_pnp_ula_1["SIM_MMSE"])
    restore_2 = (dic_pnp_ula_2["MMSE"], dic_pnp_ula_2["PSNR_MMSE"], dic_pnp_ula_2["SIM_MMSE"])
    restore_3 = (dic_pnp_ula_3["MMSE"], dic_pnp_ula_3["PSNR_MMSE"], dic_pnp_ula_3["SIM_MMSE"])
    std_1 = dic_pnp_ula_1["std"]
    std_2 = dic_pnp_ula_2["std"]
    std_3 = dic_pnp_ula_3["std"]

    c_kernel = 100
    wid, hei = 70, 70
    ax = plt.subplot(gs[indices[0]])

    width = 230
    ax.imshow(gt.astype(np.float32))
    rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
    ax.add_patch(plt.Rectangle(**rect_params))
    text_params = {'xy': (5, 5), 'text': r"PSNR$\uparrow$SSIM$\uparrow$", 'color': 'white', 'fontsize': label_size, 'va': 'top', 'ha': 'left'}
    ax.annotate(**text_params)
    ax.axis('off')
    ax.set_title("Ground Truth", fontsize=text_size)
    width = 170

    for j, im in enumerate([restore_1, restore_2, restore_3]):
        ax = plt.subplot(gs[indices[1+j]])
        ax.imshow(im[0].astype(np.float32))
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}".format(im[1], im[2]), 'color': 'white', 'fontsize': label_size, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title(name_fig_list[j], fontsize=text_size)
    
    for j, im in enumerate([std_1, std_2, std_3]):
        ax = plt.subplot(gs[indices[4+j]])
        im = np.sum(im, axis = -1)
        ax.imshow(((im-np.min(im))/(np.max(im)-np.min(im))).astype(np.float32), cmap = 'gray')
        ax.axis('off')
        ax.set_title(name_fig_list[3+j] + "\n range = [{:.2f}, {:.2f}]".format(np.min(im), np.max(im)), fontsize=text_size)

    PSNR_mmse_list = dic_pnp_ula_3['PSNR_mmse']
    x_list = np.linspace(0,1000000, len(PSNR_mmse_list))
    ax = plt.subplot(gs[indices[-1]])
    ax.semilogx(x_list, PSNR_mmse_list)
    ax.set_yticks([15, 31])
    ax.set_yticklabels([r'$15$', r'31'], fontsize=text_size)
    ax.set_xticks([10000, 100000, 1000000])
    ax.set_xticklabels([r'$10^4$', r'$10^5$', r'$10^6$'], fontsize=text_size)
    ax.set_title("PSNR of online MMSE", fontsize=text_size)

    #generate the second line of the figure to detailed the uncertainty compute by PSGLA.
    path_result = "results/result/inpainting/set1c/psgla/DnCNN/s_2.0/lambd_5.0/"

    #size of the black rectangle
    height = 35
    width = 150
    indices_2 = [(1,i) for i in range(m)]
    
    dic_psgla_1 = np.load(path_result + "im_0/sigma1_s2.0_result.npy", allow_pickle=True).item()
    dic_psgla_2 = np.load(path_result + "n_iter_100000/im_0/sigma1_s2.0_result.npy", allow_pickle=True).item()
    dic_psgla_3 = np.load(path_result + "n_iter_1000000/im_0/sigma1_s2.0_result.npy", allow_pickle=True).item()
    
    gt = dic_psgla_1["ground_truth"]
    degraded = (dic_psgla_1["observation"], dic_psgla_1["PSNR_y"], dic_psgla_1["SIM_y"])
    restore_1 = (dic_psgla_1["MMSE"], dic_psgla_1["PSNR_MMSE"], dic_psgla_1["SIM_MMSE"])
    restore_2 = (dic_psgla_2["MMSE"], dic_psgla_2["PSNR_MMSE"], dic_psgla_2["SIM_MMSE"])
    restore_3 = (dic_psgla_3["MMSE"], dic_psgla_3["PSNR_MMSE"], dic_psgla_3["SIM_MMSE"])
    std_1 = dic_psgla_1["std"]
    std_2 = dic_psgla_2["std"]
    std_3 = dic_psgla_3["std"]
    
    c_kernel = 100
    wid, hei = 70, 70

    width = 160
    ax = plt.subplot(gs[indices_2[0]])
    ax.imshow(degraded[0].astype(np.float32))
    rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
    ax.add_patch(plt.Rectangle(**rect_params))
    text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}".format(degraded[1], degraded[2]), 'color': 'white', 'fontsize': label_size, 'va': 'top', 'ha': 'left'}
    ax.annotate(**text_params)
    ax.axis('off')
    ax.set_title("Observation", fontsize=text_size)
    width = 170

    for j, im in enumerate([restore_1, restore_2, restore_3]):
        ax = plt.subplot(gs[indices_2[1+j]])
        ax.imshow(im[0].astype(np.float32))
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}".format(im[1], im[2]), 'color': 'white', 'fontsize': label_size, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        # ax.set_title(name_fig_list[j], fontsize=text_size)

    for j, im in enumerate([std_1, std_2, std_3]):
        ax = plt.subplot(gs[indices_2[4+j]])
        im = np.sum(im, axis = -1)
        ax.imshow(((im-np.min(im))/(np.max(im)-np.min(im))).astype(np.float32), cmap = 'gray')
        ax.axis('off')
        ax.set_title("\n range = [{:.2f}, {:.2f}]".format(np.min(im), np.max(im)), fontsize=text_size)
    
    PSNR_mmse_list = dic_psgla_3['PSNR_mmse']
    x_list = np.linspace(0,1000000, len(PSNR_mmse_list))
    ax = plt.subplot(gs[indices_2[-1]])
    ax.semilogx(x_list, PSNR_mmse_list)
    ax.set_yticks([15, 31])
    ax.set_yticklabels([r'$15$', r'31'], fontsize=text_size)
    ax.set_xticks([10000, 100000, 1000000])
    ax.set_xticklabels([r'$10^4$', r'$10^5$', r'$10^6$'], fontsize=text_size)
    # ax.set_title("PSNR of online MMSE", fontsize=text_size)

    fig.text(x=0.11, y=0.7, s="PnP-ULA", va='center', ha='left', fontsize=30, rotation='vertical')
    fig.text(x=0.11, y=0.3, s="PnP-PSGLA", va='center', ha='left', fontsize=30, rotation='vertical')

    fig.savefig(path_figure+'/PnP_ULA_PSGLA_with_various_N.png', dpi=300)
    plt.show()



if pars.fig_number == 5:
    #generate figure to compute the std of the PSGLA MMSE estimator.
    path_result = "results/result/inpainting/set1c/psgla/DnCNN/s_2.0/lambd_5.0/"

    name_fig_list = ["Observation", "MMSE Estimator \n Seed = 0, $N = 10^4$", "Std MMSE Estimator, $N = 10^4$", "MMSE Estimator \n Seed = 0, $N = 10^5$", "Std MMSE Estimator, $N = 10^5$"]

    n = 1
    m = len(name_fig_list) + 1

    #size of the black rectangle
    height = 35
    width = 150
    indices = [i for i in range(m)]
    
    fig = plt.figure(figsize = (m*5.2, n*7.44))
    gs = gridspec.GridSpec(n, m, hspace = 0.2, wspace = 0)

    text_size = 23
    label_size = 25

    width = 230
    
    dic_psgla_1 = np.load(path_result + "seed_alg_0/im_0/sigma1_s2.0_result.npy", allow_pickle=True).item()
    dic_psgla_100000_1 = np.load(path_result + "n_iter_100000/seed_alg_0/im_0/sigma1_s2.0_result.npy", allow_pickle=True).item()
    
    gt = dic_psgla_1["ground_truth"]
    degraded = (dic_psgla_1["observation"], dic_psgla_1["PSNR_y"], dic_psgla_1["SIM_y"])
    restore_1 = (dic_psgla_1["MMSE"], dic_psgla_1["PSNR_MMSE"], dic_psgla_1["SIM_MMSE"])
    restore_100000_1 = (dic_psgla_100000_1["MMSE"], dic_psgla_100000_1["PSNR_MMSE"], dic_psgla_100000_1["SIM_MMSE"])
    
    wid, hei = 70, 70
    ax = plt.subplot(gs[indices[0]])

    ax.imshow(gt.astype(np.float32))
    rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
    ax.add_patch(plt.Rectangle(**rect_params))
    text_params = {'xy': (5, 5), 'text': r"PSNR$\uparrow$SSIM$\uparrow$", 'color': 'white', 'fontsize': label_size, 'va': 'top', 'ha': 'left'}
    ax.annotate(**text_params)

    ax.axis('off')
    ax.set_title("Ground Truth", fontsize=text_size)

    width = 170

    for j, im in enumerate([degraded, restore_1]):
        ax = plt.subplot(gs[indices[1+j]])
        ax.imshow(im[0].astype(np.float32))
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}".format(im[1], im[2]), 'color': 'white', 'fontsize': label_size, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title(name_fig_list[j], fontsize=text_size)

    mean = dic_psgla_1["MMSE"]
    mean_2 = dic_psgla_1["MMSE"]**2
    number_of_seed = 101
    for l in tqdm(range(1, number_of_seed)):
        dic_psgla_l = np.load(path_result + "seed_alg_"+str(l)+"/im_0/sigma1_s2.0_result.npy", allow_pickle=True).item()
        mean += dic_psgla_l["MMSE"]
        mean_2 += dic_psgla_l["MMSE"]**2
    mean = mean / number_of_seed
    mean_2 = mean_2 / number_of_seed
    var = mean_2 - mean**2
    var = var*(var>=0) + 0*(var<0)
    std = np.sqrt(var)
    std = np.sum(std, axis = -1)

    ax = plt.subplot(gs[indices[3]])
    ax.imshow(((std-np.min(std))/(np.max(std)-np.min(std))).astype(np.float32), cmap = 'gray')
    ax.axis('off')
    ax.set_title(name_fig_list[2] + "\n range = [{:.2f}, {:.2f}]".format(np.min(std), np.max(std)), fontsize=text_size)

    im = restore_100000_1
    ax = plt.subplot(gs[indices[4]])
    ax.imshow(im[0].astype(np.float32))
    rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
    ax.add_patch(plt.Rectangle(**rect_params))
    text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}".format(im[1], im[2]), 'color': 'white', 'fontsize': label_size, 'va': 'top', 'ha': 'left'}
    ax.annotate(**text_params)
    ax.axis('off')
    ax.set_title(name_fig_list[3], fontsize=text_size)
    
    mean = dic_psgla_100000_1["MMSE"]
    mean_2 = dic_psgla_100000_1["MMSE"]**2
    number_of_seed = 10
    for l in tqdm(range(1, number_of_seed)):
        dic_psgla_l = np.load(path_result + "n_iter_100000/seed_alg_"+str(l)+"/im_0/sigma1_s2.0_result.npy", allow_pickle=True).item()
        mean += dic_psgla_l["MMSE"]
        mean_2 += dic_psgla_l["MMSE"]**2
    mean = mean / number_of_seed
    mean_2 = mean_2 / number_of_seed
    var = mean_2 - mean**2
    var = var*(var>=0) + 0*(var<0)
    std = np.sqrt(var)
    std = np.sum(std, axis = -1)

    ax = plt.subplot(gs[indices[-1]])
    ax.imshow(((std-np.min(std))/(np.max(std)-np.min(std))).astype(np.float32), cmap = 'gray')
    ax.axis('off')
    ax.set_title(name_fig_list[-1] + "\n range = [{:.2f}, {:.2f}]".format(np.min(std), np.max(std)), fontsize=text_size)

    
    fig.savefig(path_figure+'/Std_PSGLA_estimator.png')
    plt.show()