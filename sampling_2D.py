from utils_2D import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--name', type=str, default='symetric_gaussians', choices=['symetric_gaussians', 'disymmetric_gaussians', 'cross'])
parser.add_argument('--N', type = int, help = 'Number of iterations of algorithms PnP ULA and SnoPnP ULA')
parser.add_argument('--metric_each_step', type = bool, default=False, help = 'If True compute the Wasserstein distance at each step')
pars = parser.parse_args()

np.random.seed(0) #for reproductivity

path_result = "results"
os.makedirs(path_result, exist_ok = True)
path_result = "results/result_GMM"
os.makedirs(path_result, exist_ok = True)

###
# Algorithms
###

def PnP_ULA(N, x_0, y, delta, A, sigma, MMSE_denoiser, epsilon, alpha, Sample_posterior = [], compute_metric_each_step = False):
    """
    Run PnP-ULA with N iterations, x_0 as initialization, y as an observation, A and sigma constant of the inverse problem,
    delta as a step size and MMSE_denoiser as a denoiser of level epsilon. alpha is the regularization parameter.
    """
    n = y.shape[0] #dimension of the observation, here n = 2
    X = [x_0] #list of the sample
    Wass_dist_list = []
    
    def score_data_fidelity(y,x,A,sigma):
        return A.T@(y-A@x)/sigma**2
    
    for i in tqdm(range(N-1)):
        x_k = X[-1]
        z_k1 = np.random.randn(n)
        x_k1 = x_k + delta * score_data_fidelity(y,x_k,A,sigma) + alpha*delta*(1/epsilon)*(MMSE_denoiser(x_k, epsilon)-x_k) + np.sqrt(2*delta)*z_k1
        X.append(x_k1)
        if compute_metric_each_step and i % 100 == 0:
            Wass_dist_list.append(Wasserstein_distance(np.array(X), Sample_posterior[:len(X),:]))

    X = np.array(X)
    if compute_metric_each_step:
        return X, Wass_dist_list
    else:
        return X


def SnoPnP_ULA(N, x_0, y, delta, A, sigma, MMSE_denoiser, alpha, Sample_posterior = [], compute_metric_each_step = False):
    """
    Run SnoPnP-ULA with N iterations, x_0 as initialization, y as an observation, A and sigma constant of the inverse problem,
    delta as a step size and MMSE_denoiser as the exact MMSE denoiser.
    """
    n = y.shape[0] #dimension of the observation, here n = 2
    X = [x_0] #list of the sample
    Wass_dist_list = []
    
    def score_data_fidelity(y,x,A,sigma):
        return A.T@(y-A@x)/sigma**2
    
    for i in tqdm(range(N-1)):
        x_k = X[-1]
        z_k1 = np.random.randn(n)
        x_k1 = MMSE_denoiser(x_k + (delta /alpha) * score_data_fidelity(y,x_k,A,sigma) + np.sqrt(2*delta)*z_k1, delta)
        X.append(x_k1)
        if compute_metric_each_step and i % 100 == 0:
            Wass_dist_list.append(Wasserstein_distance(np.array(X), Sample_posterior[:len(X),:]))
    
    X = np.array(X)
    if compute_metric_each_step:
        return X, Wass_dist_list
    else:
        return X

if pars.N == None:
    N_list = [100, 1000, 10000]
else:
    N_list = [pars.N]

for N in N_list:
    #first test - constant definition
    name = pars.name #'symetric_gaussians', 'disymmetric_gaussians', 'cross'
    mu_list, sigma_list, pi_list = gaussian_mixt_example(name)
    A = np.eye(2)  #np.array([[2,0],[0,1]])
    In = np.eye(2)
    sigma = 1 #the noise level represent sigma^2

    epsilon_pnp_ula = 0.5
    MMSE_denoiser = Theorical_MMSE(mu_list, sigma_list, pi_list)
    delta_pnp_ula = 0.1
    alpha_pnp_ula = 1.5
    Y = [np.array([0,0]), np.array([0,-2]), np.array([-6,6])] #observation
    
    Sample_posterior = []
    Sample_posterior_2 = []
    for y in Y:
        Sample_posterior.append(sample_posterior(A, y, sigma, N, mu_list, sigma_list, pi_list))
        Sample_posterior_2.append(sample_posterior(A, y, sigma, N, mu_list, sigma_list, pi_list))

    Sample_ULA = []
    Wass_dist_ULA_list = []
    for i in range(3):
        if not(pars.metric_each_step):
            Sample_ULA.append(PnP_ULA(N, Y[i], Y[i], delta_pnp_ula, A, sigma, MMSE_denoiser, epsilon_pnp_ula, alpha_pnp_ula))
        else:
            Samples_ULA_i, Wass_dist_list_i = PnP_ULA(N, Y[i], Y[i], delta_pnp_ula, A, sigma, MMSE_denoiser, epsilon_pnp_ula, alpha_pnp_ula, Sample_posterior = Sample_posterior[i], compute_metric_each_step = True)
            Sample_ULA.append(Samples_ULA_i)
            Wass_dist_ULA_list.append(Wass_dist_list_i)
            # plot of the evolution of the Wasserstein dist
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(Wass_dist_list_i)
            fig.savefig(path_result+"/Wasserstein_dist_each_step_PnP_ULA_"+name+"_N"+str(N)+"_y_"+str(i)+".png")


    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        ax2 = ax[i]
        sample_ULA = Sample_ULA[i]
        sample_ULA = np.random.permutation(sample_ULA)[:1000] #take a sample of 1000 points
        y = Y[i]
        mu_cond_list, sigma_cond_list, p_list = constantes_conditionnal_prob(A, y, sigma, mu_list, sigma_list, pi_list)
        alpha_list = Alpha(p_list)
        draw_gaussian_mixture(ax2, mu_cond_list, sigma_cond_list, alpha_list, rbox = 5, color = 'k',  label = 'x_data|y', linewidth = 2)
        ax2.scatter(sample_ULA[:, 0], sample_ULA[:, 1], alpha=0.6, c='g', label = 'x_data|y sample') #sample of PnP-ULA
        draw_gaussian_mixture(ax2, mu_list, sigma_list, pi_list, rbox = 9, color = 'b',  label = 'x_data', linewidth = 2)
        ax2.scatter(y[0], y[1], c='r', label = 'y')
        ax2.grid(False)
        ax2.legend()
    fig.savefig(path_result+"/Sample_PnP_ULA_"+name+"_N"+str(N)+".png")

    delta_snopnp_ula = 0.3
    alpha_snopnp_ula = 2 / 3

    Sample_SnoPnP_ULA = []
    Wass_dist_PSGLA_list = []
    for i in range(3):
        if not(pars.metric_each_step):
            Sample_SnoPnP_ULA.append(SnoPnP_ULA(N, Y[i], Y[i], delta_snopnp_ula, A, sigma, MMSE_denoiser, alpha_snopnp_ula))
        else:
            Samples_PSGLA_i, Wass_dist_list_i = SnoPnP_ULA(N, Y[i], Y[i], delta_snopnp_ula, A, sigma, MMSE_denoiser, alpha_snopnp_ula, Sample_posterior = Sample_posterior[i], compute_metric_each_step = True)
            Sample_SnoPnP_ULA.append(Samples_PSGLA_i)
            Wass_dist_PSGLA_list.append(Wass_dist_list_i)
            # plot of the evolution of the Wasserstein dist
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(Wass_dist_list_i)
            fig.savefig(path_result+"/Wasserstein_dist_each_step_PSGLA_"+name+"_N"+str(N)+"_y_"+str(i)+".png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        ax2 = ax[i]
        sample_ULA = Sample_SnoPnP_ULA[i]
        sample_ULA = np.random.permutation(sample_ULA)[:1000] #take a sample of 1000 points
        y = Y[i]
        mu_cond_list, sigma_cond_list, p_list = constantes_conditionnal_prob(A, y, sigma, mu_list, sigma_list, pi_list)
        alpha_list = Alpha(p_list)
        draw_gaussian_mixture(ax2, mu_cond_list, sigma_cond_list, alpha_list, rbox = 5, color = 'k',  label = 'x_data|y', linewidth = 2)
        ax2.scatter(sample_ULA[:, 0], sample_ULA[:, 1], alpha=0.6, c='g', label = 'x_data|y sample') #sample of PnP-ULA
        draw_gaussian_mixture(ax2, mu_list, sigma_list, pi_list, rbox = 9, color = 'b',  label = 'x_data', linewidth = 2)
        ax2.scatter(y[0], y[1], c='r', label = 'y')
        ax2.grid(False)
        ax2.legend()
    fig.savefig(path_result+"/Sample_SnoPnP_ULA_"+name+"_N"+str(N)+".png")

    # Compute of distances
    Sliced_Wass_PnP_ULA, Sliced_Wass_SnoPnP_ULA, Sliced_Wass_ref, Wass_PnP_ULA, Wass_SnoPnP_ULA, Wass_ref, MMSE_PnP_ULA, MMSE_SnoPnP_ULA = [], [], [], [], [], [], [], []
    for i in range(len(Y)):
        print('Observation '+str(i))
        #Sliced Wasserstein distance
        Sliced_dist_pnp_ula = ot.sliced.sliced_wasserstein_distance(Sample_posterior[i], Sample_ULA[i], n_projections=50, p=2, seed=None)
        Sliced_dist_snopnp_ula = ot.sliced.sliced_wasserstein_distance(Sample_posterior[i], Sample_SnoPnP_ULA[i], n_projections=50, p=2, seed=None)
        Sliced_dist_two_posterior = ot.sliced.sliced_wasserstein_distance(Sample_posterior[i], Sample_posterior_2[i], n_projections=50, p=2, seed=None)
        print("Sliced Wasserstein for PnP ULA = {:.2f} and SnoPnP ULA = {:.2f} and reference dist = {:.2f}".format(Sliced_dist_pnp_ula, Sliced_dist_snopnp_ula,Sliced_dist_two_posterior))
        Sliced_Wass_PnP_ULA.append(Sliced_dist_pnp_ula)
        Sliced_Wass_SnoPnP_ULA.append(Sliced_dist_snopnp_ula)
        Sliced_Wass_ref.append(Sliced_dist_two_posterior)

        # if N >= 1000:
        # Wasserstein distance
        Wass_dist_pnp_ula = Wasserstein_distance(Sample_posterior[i], Sample_ULA[i])
        Wass_dist_snopnp_ula = Wasserstein_distance(Sample_posterior[i], Sample_SnoPnP_ULA[i])
        Wass_dist_two_posterior = Wasserstein_distance(Sample_posterior[i], Sample_posterior_2[i])
        print("Wasserstein dist for PnP ULA = {:.2f} and SnoPnP ULA = {:.2f} and reference dist = {:.2f}".format(Wass_dist_pnp_ula, Wass_dist_snopnp_ula,Wass_dist_two_posterior))
        Wass_PnP_ULA.append(Wass_dist_pnp_ula)
        Wass_SnoPnP_ULA.append(Wass_dist_snopnp_ula)
        Wass_ref.append(Wass_dist_two_posterior)

        # Kernel approximation of the point cloud to compute distance
        X_0, X_1 = np.mgrid[-8:8:100j, -8:8:100j]
        positions = np.vstack([X_0.ravel(), X_1.ravel()])
        values = np.vstack([Sample_ULA[i][:,0], Sample_ULA[i][:,1]])
        kernel = stats.gaussian_kde(values)
        Z_PnP_ULA = np.reshape(kernel(positions).T, X_0.shape)
        Z_PnP_ULA = Z_PnP_ULA / np.sum(Z_PnP_ULA)
        # # Plot of the empirical density
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.imshow(Z)
        # fig.savefig(path_result+"/density_pnp_ula_Y"+str(i)+"_N_"+str(N)+".png")

        values = np.vstack([Sample_SnoPnP_ULA[i][:,0], Sample_SnoPnP_ULA[i][:,1]])
        kernel = stats.gaussian_kde(values)
        Z_SnoPnP_ULA = np.reshape(kernel(positions).T, X_0.shape)
        Z_SnoPnP_ULA = Z_SnoPnP_ULA / np.sum(Z_SnoPnP_ULA)
        # # Plot of the empirical density
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.imshow(Z)
        # fig.savefig(path_result+"/density_snopnp_ula_Y"+str(i)+"_N_"+str(N)+".png")

        mu_cond_list, sigma_cond_list, p_list = constantes_conditionnal_prob(A, Y[i], sigma, mu_list, sigma_list, pi_list)
        posterior_density = gaussian_mixture_density(positions, mu_cond_list, sigma_cond_list, p_list)
        Z_post = np.reshape(posterior_density.T, X_0.shape)
        Z_post = Z_post / np.sum(Z_post)
        # # Plot of the posterior density
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.imshow(Z_post)
        # fig.savefig(path_result+"/posteror_density_Y"+str(i)+"_N_"+str(N)+".png")
        
        mmse_pnp_ula = np.sum((Z_PnP_ULA - Z_post)**2)
        mmse_snopnp_ula = np.sum((Z_SnoPnP_ULA - Z_post)**2)
        print("MMSE dist for PnP ULA = {} and SnoPnP ULA = {}".format(mmse_pnp_ula, mmse_snopnp_ula))
        MMSE_PnP_ULA.append(mmse_pnp_ula)
        MMSE_SnoPnP_ULA.append(mmse_snopnp_ula)


    #save the result of the experiment
    dict = {
            'A' : A,
            'mu_list' : mu_list,
            'sigma_list' : sigma_list,
            'pi_list' : pi_list,
            'sigma' : sigma,
            'delta_pnp_ula' : delta_pnp_ula,
            'delta_snopnp_ula' : delta_snopnp_ula,
            'alpha_pnp_ula' : alpha_pnp_ula,
            'alpha_snopnp_ula' : alpha_snopnp_ula,
            'epsilon_pnp_ula' : epsilon_pnp_ula,
            'Y' : Y,
            'Sample_PnP_ULA' : Sample_ULA,
            'Sample_SnoPnP_ULA' : Sample_SnoPnP_ULA,
            'Sliced_Wass_PnP_ULA' : Sliced_Wass_PnP_ULA, 
            'Sliced_Wass_SnoPnP_ULA' : Sliced_Wass_SnoPnP_ULA, 
            'Sliced_Wass_ref' : Sliced_Wass_ref, 
            'Wass_PnP_ULA' : Wass_PnP_ULA, 
            'Wass_SnoPnP_ULA' : Wass_SnoPnP_ULA, 
            'Wass_ref' : Wass_ref, 
            'MMSE_PnP_ULA' : MMSE_PnP_ULA, 
            'MMSE_SnoPnP_ULA' : MMSE_SnoPnP_ULA,
        }
    if pars.metric_each_step:
        dict['Wass_dist_ULA_list'] = Wass_dist_ULA_list
        dict['Wass_dist_PSGLA_list'] = Wass_dist_PSGLA_list

    np.save(path_result+"/Sample_PnP_SnoPnP_ULA_"+ name +"_N"+str(N)+'_result.npy', dict)