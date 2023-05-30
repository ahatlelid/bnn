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
    n_param = tensor_D.size(dim=0)
    tau2 = 1000 # 1/tau2 is the noise added to the diagonal
    tensor_Q_m_modified = tensor_Q_m + torch.eye(n_param)*(1./tau2)
    sigma2_eps = 0.01  # 1/sigma2_eps is the factor before the likelihood
    tensor_mu_m = torch.zeros(n_param)

    tensor_Sigma_m = torch.inverse(tensor_Q_m_modified)
    tensor_Sigma_eps = torch.eye(n_param)*sigma2_eps
    tensor_mu_eps = tensor_mu_m 

    #data = Data(tensor_mu_m, tensor_Sigma_m, tensor_mu_eps, tensor_Sigma_eps)


    number_of_data = 10000
    lambda_ = 0.5
    lambda2 = lambda_**2
    l2_lambda =  1./(number_of_data*lambda2) 
    

    initial_lr = 0.1
    # if not using the cutom scheduler, this is a list of all the lrs to be used
    lr_list = np.array([0.1, 0.01, 0.001])
    lr_list = np.append(lr_list, np.linspace(0.001, 0.0001, 10))
    #lr_list = np.array([0.1, 0.01, 0.001, 0.00075, 0.0005, 0.00025, 0.0001])


    #total_runs = 100000  # very high number, either lower it, or break run when error has converged 
    total_runs = len(lr_list)*100
    n_batches = 1000 # run n_batches before printing error. The error is averages across n_batches. Number of observation for each print is n_batches*batch_size
    batch_size = 1000

    tensor_batch = torch.load('../data/1.0e+05/data/data.pt')

    data_fit_coefficient = 1./0.01
    data_reg_coefficient = 1
    prior_coefficient = 1./0.09/100 # 

    #run_types = ['map', 'pos', 'neg']
    #run_types = ['neg']
    run_types = ['pos']
    #run_types = ['map']
    #run_type = 'pos'
    #run_type = 'neg'

    for run_type in run_types:

        # clean folder if it exists
        path_to_output_folder = '../saved_models/rml_100/' + run_type + '/'
        if os.path.exists(path_to_output_folder) and os.path.isdir(path_to_output_folder):
            shutil.rmtree(path_to_output_folder)
        os.makedirs(path_to_output_folder)


        runs = 100
        for run in range(runs):
            run_path = path_to_output_folder + '/' + str(run)
            os.makedirs(run_path)
            torch.manual_seed(0)
            model = Net_mask() 
            model_weight_name = run_path + "/model_weights.pth"
            torch.save(model.state_dict(), model_weight_name)

            optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
            #loss = Loss(tensor_Q_m_modified, sigma2_eps, l2_lambda)
            loss = Loss(tensor_Q_m, data_fit_coefficient, data_reg_coefficient, prior_coefficient)
            lr_counter = 0

            if run_type == 'pos':
                data_noise = torch.load(f'../data/1.0e+05/rml_noise/{run}/noise_data.pt')
                data_reg_noise = torch.load(f'../data/1.0e+05/rml_noise/{run}/noise_data_regularization.pt')
                tensor_batch[:,:n_param] += data_noise*tensor_batch[:,n_param:]
                loss.add_gaussian_noise(sign='positive', num=run)
            if run_type == 'neg':
                data_noise = torch.load(f'../data/rml/{run}/noise_data_{run}.pt')
                data_noise *= -1
                tensor_batch[:,:n_param] += data_noise*tensor_batch[:,n_param:]
                loss.add_gaussian_noise(sign='negative', num=run)


            
            size = 100
            #total_runs = len(lr_list)*100
            total_runs = len(lr_list)*size
            for j in range(total_runs):
                #losses = torch.zeros(100, 4)
                losses = torch.zeros(size, 4)
                idx = torch.randperm(tensor_batch.size(0))
                tensor_batch = tensor_batch[idx,:]
                data_reg_noise = data_reg_noise[idx,:]
                for i in range(100):
                #for i in range(10):
                    tensor_batch_ = tensor_batch[batch_size*i:batch_size*(i+1)]
                    data_reg_noise_ = data_reg_noise[batch_size*i:batch_size*(i+1)] 

                    optimizer.zero_grad() 
                    tensor_output = model(tensor_batch_)
                    tensor_losses_sum, tensor_losses_likelihood, tensor_losses_prior, tensor_losses_regularization = loss.loss(tensor_batch_, tensor_output, model, data_reg_noise_)
                    losses[i][0] = tensor_losses_sum
                    losses[i][1] = tensor_losses_likelihood
                    losses[i][2] = tensor_losses_prior
                    losses[i][3] = tensor_losses_regularization
                    tensor_losses_sum.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0, error_if_nonfinite=False)
                    optimizer.step()

                # print info after n_batches*batch_size
                if j % 10 == 0:
                    #print(f"Run: {run:3} | Epoch: {j:3} | lr: {lr_list[lr_counter]:10.7f} | total loss: {torch.mean(losses[:,0]):10.5f} | likelihood: {torch.mean(losses[:,1]):10.5f} | prior: {torch.mean(losses[:,2]):10.5f} | regularization: {torch.mean(losses[:,3]):12.10f}")
                    print(f"Run: {run:3} | Epoch: {j:3} | lr: {lr_list[lr_counter]:10.7f} | {torch.mean(losses[:,0]):10.3f} {torch.mean(losses[:,1])*data_fit_coefficient:10.3f} {torch.mean(losses[:,2])*data_reg_coefficient:10.3f} {torch.mean(losses[:,3])*prior_coefficient:12.3f} | {torch.mean(losses[:,1])+torch.mean(losses[:,2])+torch.mean(losses[:,3]):10.3f} {torch.mean(losses[:,1]):10.3f} {torch.mean(losses[:,2]):10.3f} {torch.mean(losses[:,3]):10.3f}")

                #if j % 100 == 0 and j != 0:  # change learning rate after 10*n_batches*batch_size observations
                if j % size == 0 and j != 0:  # change learning rate after 10*n_batches*batch_size observations
                    # set new learning rate
                    for param in optimizer.param_groups:
                        param['lr'] = lr_list[lr_counter] 
                    lr_counter += 1
                    print('new lr')
                #if j % 100 == 0: 
                if j % size == 0: 
                    # plot figure
                    #plt.figure(random.randint(0,1e10))
                    #obs = data.get_tensor_data(1, [2.5, 1.5, -3, -3], [1, 3, 5, 8])
                    #out_raw = model(obs)
                    #out = out_raw[0].detach().numpy()
                    #post, _ = data.get_tensor_posterior(obs[0])
                    #plt.plot(range(10), post, 'b')
                    #plt.plot(range(10), out, 'r')
                    #plt.plot([1, 3, 5, 8], [2.5, 1.5, -3, -3], 'go')
                    #plt.legend(['posterior', 'estimate', 'points'])
                    #plt.title(f"Neural net estimate vs analytical (tau2={tau2})")
                    #plt.savefig(run_path + '/plot.pdf')
                    continue
                torch.save(model.state_dict(), model_weight_name)
            print('============= new run =============')