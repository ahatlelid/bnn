from importlib.resources import path
from re import I
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from utils.model import Net_mask
from utils.data import Data
from utils.loss_experiment import Loss
from utils.lr_scheduler import lr_scheduler
from utils.weights import set_weights
from scripts.parameters import get_parameters
#
import os
import shutil

from torch.distributions.multivariate_normal import MultivariateNormal

if __name__ == "__main__":
    """Method that trains the neural net."""

    tensor_D = torch.tensor(
        [[1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0,  1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0,  0, 1, -1, 0, 0, 0, 0, 0],
        [0, 0,  0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0,  0, 0, 0, 1, -1, 0, 0, 0],
        [0, 0,  0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0,  0, 0, 0, 0, 0, 1, -1, 0],
        [0, 0,  0, 0, 0, 0, 0, 0, 1, -1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1],], dtype=torch.float
    )
    tensor_Q_m  = torch.mm(torch.t(tensor_D), tensor_D)
    n_param = tensor_D.size(dim=0)
    tau2 = 1000 # 1/tau2 is the noise added to the diagonal
    tensor_Q_m_modified = tensor_Q_m + torch.eye(n_param)*(1./tau2)
    sigma2_eps = 0.01  # 1/sigma2_eps is the factor before the likelihood
    tensor_mu_m = torch.zeros(n_param)

    tensor_Sigma_m = torch.inverse(tensor_Q_m_modified)
    tensor_Sigma_eps = torch.eye(n_param)*sigma2_eps
    tensor_mu_eps = tensor_mu_m 

    path_to_output_folder = '../saved_models/experiment/'


    l2_lambda = 1  # should be 1./(n_data*tau**2) for real scaling
    
    # clean folder if it exists
    if os.path.exists(path_to_output_folder) and os.path.isdir(path_to_output_folder):
        shutil.rmtree(path_to_output_folder)
    os.makedirs(path_to_output_folder)


    data = Data(tensor_mu_m, tensor_Sigma_m, tensor_mu_eps, tensor_Sigma_eps)

    run = 0
    run_path = path_to_output_folder + '/' + str(run)
    os.makedirs(run_path)
    model = Net_mask() 
    model_weight_name = run_path + "/model_weights.pth"
    torch.save(model.state_dict(), model_weight_name)

    initial_lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
    loss = Loss(tensor_Q_m_modified, sigma2_eps, l2_lambda)

    model_weight_name = run_path + "/model_weights.pth"

    lr_counter = 0
    # if not using the cutom scheduler, this is a list of all the lrs to be used
    lr_list = [0.001, 0.0009, 0.0008, 0.00075, 0.0007, 0.00065, 0.0006, 0.00059, 0.00058, 0.00057, 0.00056, 0.00055, 0.00054, 0.00053, 0.00052, 0.00051, 0.0005, 0.00045, 0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.00015, 0.0001, 0.000095, 0.00009, 0.000085, 0.00008, 0.000075, 0.00007, 0.000065, 0.00006, 0.000055, 0.00005, 0.000045, 0.00004, 0.000035, 0.00003, 0.000025, 0.00002, 0.000015, 0.00001]

    # this controls the type of training, only one can be True
    always_same_mask = True
    same_mask_per_batch = False
    different_masks_per_batch = False

    total_runs = 1000  # very high number, either lower it, or break run when error has converged 
    n_batches = 1000 # run n_batches before printing error. The error is averages across n_batches. Number of observation for each print is n_batches*batch_size
    batch_size = 1000
    for j in range(total_runs):
        losses = torch.zeros(n_batches, 4)
        for i in range(n_batches):
            tensor_batch = torch.zeros(batch_size, 2*n_param)
            tensor_d_sample =  MultivariateNormal(loc=torch.zeros(n_param), covariance_matrix=(torch.eye(n_param)*30)).sample(sample_shape=(batch_size,))

            if always_same_mask:
                mask = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 0, 1, 0]])
                tensor_masks = torch.cat(batch_size*[mask])
            if same_mask_per_batch:
                tensor_n_masked = torch.randint(n_param, (1,))
                tensor_masks = torch.rand(1, n_param).argsort(dim=1)
                tensor_masks = (tensor_masks < tensor_n_masked)*1
                tensor_masks = torch.cat(batch_size*[tensor_masks])
            if different_masks_per_batch:
                tensor_n_masked = torch.randint(n_param, (batch_size,))
                tensor_masks = torch.rand(batch_size, n_param).argsort(dim=1)
                tensor_masks = (tensor_masks < tensor_n_masked.unsqueeze(1))*1

            tensor_batch[:,:n_param] = tensor_d_sample*tensor_masks
            tensor_batch[:,n_param:] = tensor_masks

            optimizer.zero_grad() 
            tensor_output = model(tensor_batch)
            tensor_losses_sum, tensor_losses_likelihood, tensor_losses_prior, tensor_losses_regularization = loss.loss(tensor_batch, tensor_output, model)
            losses[i][0] = tensor_losses_sum
            losses[i][1] = tensor_losses_likelihood
            losses[i][2] = tensor_losses_prior
            losses[i][3] = tensor_losses_regularization
            tensor_losses_sum.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()
        print(f"Run: {run:3} | Epoch: {j:3} | lr: {lr_list[lr_counter]:10.7f} | total loss: {torch.mean(losses[:,0]):10.5f} | likelihood: {torch.mean(losses[:,1]):10.5f} | prior: {torch.mean(losses[:,2]):10.5f} | regularization: {torch.mean(losses[:,3]):12.10f}")

        if j % 10 == 0 and j != 0:  # change learning rate after 10*n_batches*batch_size observations
            # set new learning rate
            for param in optimizer.param_groups:
                param['lr'] = lr_list[lr_counter] 
            lr_counter += 1
            print('new lr')
        if j % 10 == 0: 
            # plot figure
            plt.figure(j)
            obs = data.get_tensor_data(1, [2.5, 1.5, -3, -3], [1, 3, 5, 8])
            out_raw = model(obs)
            out = out_raw[0].detach().numpy()
            post, _ = data.get_tensor_posterior(obs[0])
            plt.plot(range(10), post, 'b')
            plt.plot(range(10), out, 'r')
            plt.plot([1, 3, 5, 8], [2.5, 1.5, -3, -3], 'go')
            plt.legend(['posterior', 'estimate', 'points'])
            plt.title(f"Neural net estimate vs analytical (tau2={tau2})")
            plt.savefig('../saved_models/experiment/plot.pdf')
        torch.save(model.state_dict(), model_weight_name)