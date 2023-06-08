from importlib.resources import path
from re import I
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

import sys
sys.path.append("..")
from utils.model3 import Net_mask
#from utils.data import Data
from utils.loss_experiment import Loss
#from utils.lr_scheduler import lr_scheduler
#from utils.weights import set_weights
#from scripts.parameters import get_parameters
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
    #n_param = tensor_D.size(dim=0)
    #vu = 1000 # 1/tau2 is the noise added to the diagonal
    #inversion_noise = 1./vu
    #tensor_Q_m_modified = tensor_Q_m + torch.ones(n_param, n_param)*inversion_noise
    #sigma2_eps = 0.01  # 1/sigma2_eps is the factor before the likelihood
    #tensor_mu_m = torch.zeros(n_param)

    #tensor_Sigma_m = torch.inverse(tensor_Q_m_modified)
    #tensor_Sigma_eps = torch.eye(n_param)*sigma2_eps
    #tensor_mu_eps = tensor_mu_m 

    #data = Data(tensor_mu_m, tensor_Sigma_m, tensor_mu_eps, tensor_Sigma_eps)


    #number_of_data = 10000
    tau2 = 0.3**2
    sigma = 0.1
    sigma2 = sigma**2 #=0.01
    #l2_lambda =  1./(number_of_data*lambda2) 
    

    lr_initial = 0.1
    lr_start = 0.001
    lr_minimum = 0.00001 
    lr_steps = 300
    lr_list = np.array([0.1, 0.01])
    lr_list = np.append(lr_list, np.linspace(lr_start, lr_minimum, lr_steps))
    lr_counter = 0


    #total_runs = 100000  # very high number, either lower it, or break run when error has converged 
    total_runs = len(lr_list)*100
    n_batches = 1000 # run n_batches before printing error. The error is averages across n_batches. Number of observation for each print is n_batches*batch_size
    batch_size = 1000

    #tensor_batch = torch.load('../data/100_000/data_n_100_000_var_1.pt')



    # clean folder if it exists
    path_to_output_folder = '../saved_models/models_infinite_data/MAP_inf_prior_10/'
    #if os.path.exists(path_to_output_folder) and os.path.isdir(path_to_output_folder):
    #    shutil.rmtree(path_to_output_folder)
    os.makedirs(path_to_output_folder)


    model = Net_mask() 
    #model_weight_name = run_path + "/model_weights.pth"
    #torch.save(model.state_dict(), model_weight_name)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_initial)

    batch_size = 1000

    tau2 = 0.3**2 #=0.09
    sigma2 = 0.1**2 #=0.01

    data_fit_coefficient = 1./sigma2
    data_reg_coefficient = 1
    #prior_coefficient = 1./tau2*0
    prior_coefficient = 1./tau2/10
    loss = Loss(tensor_Q_m, data_fit_coefficient, data_reg_coefficient, prior_coefficient)


    n_param = 10
    n_data = 1000

    n_batch_runs = 1000

    size = 100
    #total_runs = len(lr_list)*100
    #total_runs = len(lr_list)*size
    total_runs = 100_000 
    for j in range(total_runs):
        #losses = torch.zeros(100, 4)
        losses = torch.zeros(1000, 4)
        #idx = torch.randperm(tensor_batch.size(0))
        #tensor_batch = tensor_batch[idx,:]
        for i in range(n_batch_runs):
            tensor_data = torch.zeros(n_data, 2*n_param)
            tensor_d_sample =  MultivariateNormal(loc=torch.zeros(n_param), covariance_matrix=torch.eye(n_param)).sample(sample_shape=(n_data,))
            tensor_n_masked = torch.randint(n_param, (n_data,))
            tensor_masks = torch.rand(n_data, n_param).argsort(dim=1)
            tensor_masks = (tensor_masks < tensor_n_masked.unsqueeze(1))*1
            tensor_data[:,:n_param] = tensor_d_sample*tensor_masks
            tensor_data[:,n_param:] = tensor_masks
            tensor_batch = tensor_data

            optimizer.zero_grad() 
            tensor_output = model(tensor_batch)
            total_loss, tensor_data_fit_loss, tensor_data_regularization_loss, tensor_prior_loss = loss.loss(tensor_batch, tensor_output, model)
            losses[i][0] = total_loss 
            losses[i][1] = tensor_data_fit_loss
            losses[i][2] = tensor_data_regularization_loss
            losses[i][3] = tensor_prior_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()

        # print info after n_batches*batch_size
        #if j % 10 == 0:
        #print(f"Run: {run:3} | Epoch: {j:3} | lr: {lr_list[lr_counter]:10.7f} | total loss: {torch.mean(losses[:,0]):10.5f} | likelihood: {torch.mean(losses[:,1]):10.5f} | prior: {torch.mean(losses[:,2]):10.5f} | regularization: {torch.mean(losses[:,3]):12.10f}")
        #print(f"n 1000 batches: {j} | lr: {lr_list[lr_counter]:10.7f} | total loss: {torch.mean(losses[:,0]):10.5f} | data loss: {torch.mean(losses[:,1]):10.5f} | data reg loss: {torch.mean(losses[:,2]):10.5f} | prior loss: {torch.mean(losses[:,3]):12.10f}")
        print(f"n: {j:3} | lr: {lr_list[lr_counter]:10.7f} | {torch.mean(losses[:,0]):10.3f} {torch.mean(losses[:,1])*data_fit_coefficient:10.3f} {torch.mean(losses[:,2])*data_reg_coefficient:10.3f} {torch.mean(losses[:,3])*prior_coefficient:12.3f} | {torch.mean(losses[:,1])+torch.mean(losses[:,2])+torch.mean(losses[:,3]):10.3f} {torch.mean(losses[:,1]):10.3f} {torch.mean(losses[:,2]):10.3f} {torch.mean(losses[:,3]):10.3f}")

        #if j % 100 == 0 and j != 0:  # change learning rate after 10*n_batches*batch_size observations
        if j % 10 == 0 and j != 0:  # change learning rate after 10*n_batches*batch_size observations
            # set new learning rate
            #print(lr_counter)
            #print(len(lr_list))
            #if lr_counter < len(lr_list):
            #    for param in optimizer.param_groups:
            #        param['lr'] = lr_list[lr_counter] 
            #    if lr_counter < len(lr_list) - 1:
            #        lr_counter += 1
            #        print('new lr')
            for param in optimizer.param_groups:
                param['lr'] = lr_list[lr_counter] 
            lr_counter += 1
            print('new lr')
        torch.save(model.state_dict(), f'{path_to_output_folder}model_weights.pth')