from importlib.resources import path
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from utils.model import Net_mask
from utils.data import Data
from utils.loss import Loss
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
    tau2 = 1000 # 1/tau2 is the noise added
    tensor_Q_m_modified = tensor_Q_m + torch.eye(n_param)*(1./tau2)
    sigma2_eps = 0.01
    tensor_mu_m = torch.zeros(n_param)

    tensor_Sigma_m = torch.inverse(tensor_Q_m_modified)
    tensor_Sigma_eps = torch.eye(n_param)*sigma2_eps
    tensor_mu_eps = tensor_mu_m 

    # data options
    generate_data = True
    generate_data_noise = False
    generate_parameter_noise = False
    generate_initial_weight_values = False

    # path to data folders
    path_to_data_folder = '../data/run1/data_n_100000_var_1.pt'
    path_to_data_noise_folder = '../data/..'
    path_to_parameter_noise_folder = '../data/..'
    path_to_weight_initial_values = '../data/..'

    # output folder
    path_to_output_folder = '../saved_models/experiment/'

    # runs
    number_of_runs = 10

    # data options
    number_of_data = 100_000
    batch_size = 1000
    n_batches = 100 # n_batches*batch_size=number_of_data
    n_epochs = 100

    # scaling factors
    prior_variance = 0.5**2
    data_fit_scale_factor = 0.01
    regularization_scale_factor = 1
    prior_scale_factor = 1./(number_of_data*prior_variance)

    # randomized maximum likelihood
    sign_of_noise = 'positive'

    # initial weight
    initial_weight = 'default'

    # learning rate options
    initial_lr = 0.1
    lowest_lr = 1e-5
    reduction_factor = 0.2
    lr_automatic_ajust = False # True if lr should drop depenidng on loss, False if it should drop depending on epoch
    epochs_with_same_lr = 50
    required_reduction = 0.001
    span = 20
    delay = 20 
    display_info = True

    n_data = 1000
    #n_data = int(n_data*4)
    #batch_size = 10_000
    #n_batches = round(n_data/batch_size)
    #n_bathes = 100
    n_epochs = 1000000
    tau = 0.5
    l2_lambda = 1./(n_data*tau**2)
    
    runs=1
    #sign = 'negative' # positive, negative
    #init_type = 'default' # map, positive, negative, default
    #outer_folder = 'rml_negative'

    # parameters to be set if using lr scheduler automatic
    #span = 20
    #delay = 20 
    #required_reduction = 0.001
    #lowest_lr = 1e-5 

    factor = 0.2
    #display_info = True

    #dir_path = '..]/'
    #dir_path = '../saved_models/experiment' + str(outer_folder)
    #dir_path = '../saved_models/experiment'
    if os.path.exists(path_to_output_folder) and os.path.isdir(path_to_output_folder):
        shutil.rmtree(path_to_output_folder)
    os.makedirs(path_to_output_folder)

    #tensor_data = data_generator.get_tensor_data(n_data)

    for run in range(runs):

        data = Data(tensor_mu_m, tensor_Sigma_m, tensor_mu_eps, tensor_Sigma_eps)

        run_path = path_to_output_folder + '/' + str(run)
        os.makedirs(run_path)
        model = Net_mask() # TODO see if can move this outside of loop
        model_weight_name = run_path + "/model_weights.pth"
        torch.save(model.state_dict(), model_weight_name)

        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
        lrs = lr_scheduler(optimizer, lr=initial_lr, span=span, delay=delay, required_reduction=required_reduction, lowest_lr=lowest_lr, factor=factor)
        #loss = Loss(tensor_Q_m_modified, sigma2_eps, l2_lambda)
        loss = Loss(tensor_Q_m_modified, sigma2_eps, l2_lambda)

        tensor_losses = torch.zeros((n_epochs, n_batches, 3)) # Store total loss, data loss, prior loss and regularization loss
        model_weight_name = run_path + "/model_weights.pth"

        #tensor_data = torch.load('../data/new/test_data_n100_000_var0_1.pt')
        #if run == 0: # run=0 is MLE estimate
        #    tensor_data_with_noise = tensor_data.detach().clone()
        #else: 
        #    model = set_weights(model, init_type=init_type, variance=0.5**2, path='../saved_models/map/0/model_weights.pth')
        #    tensor_data_with_noise = tensor_data.detach().clone()
        #    tensor_data_with_noise = data_generator.add_noise(tensor_data_with_noise, sigma2_data_noise=sigma2_eps, sign=sign)
        #    loss.add_gaussian_noise(noise_variance=tau**2, model=model, sign=sign)

        lr_counter = 0
        lr_list = [0.1, 0.05, 0.001, 0.0009, 0.0008, 0.00075, 0.0007, 0.00065, 0.0006, 0.00059, 0.00058, 0.00057, 0.00056, 0.00055, 0.00054, 0.00053, 0.00052, 0.00051, 0.0005, 0.00045, 0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.00015, 0.0001, 0.000095, 0.00009, 0.000085, 0.00008, 0.000075, 0.00007, 0.000065, 0.00006, 0.000055, 0.00005, 0.000045, 0.00004, 0.000035, 0.00003, 0.000025, 0.00002, 0.000015, 0.00001]
        for i in range(n_epochs):

            tensor_data = torch.zeros(number_of_data, 2*n_param)
            #number_of_data = 1000
            tensor_data = torch.zeros(number_of_data, 2*n_param)
            #tensor_d_sample =  MultivariateNormal(loc=torch.zeros(n_param), covariance_matrix=(torch.eye(n_param)*100)).sample(sample_shape=(number_of_data,))
            tensor_d_sample =  MultivariateNormal(loc=torch.zeros(n_param), covariance_matrix=(torch.eye(n_param)*30)).sample(sample_shape=(number_of_data,))
            #u_min = -10
            #u_max = 10
            #tensor_d_sample = (u_min - u_max) * torch.rand(number_of_data, n_param) + u_max 
            #print(tensor_d_sample)
            #print(tensor_d_sample.shape)
            #exit()
            tensor_n_masked = torch.randint(n_param, (number_of_data,))
            tensor_masks = torch.rand(number_of_data, n_param).argsort(dim=1)
            tensor_masks = (tensor_masks < tensor_n_masked.unsqueeze(1))*1
            tensor_data[:,:n_param] = tensor_d_sample*tensor_masks
            tensor_data[:,n_param:] = tensor_masks
            #tensor_batch = tensor_data


            #epochs_with_same_lr = 10000
            if i % epochs_with_same_lr == 0 and i != 0: 
                for param in optimizer.param_groups:
                    param['lr'] = lr_list[lr_counter] 
                lr_counter += 1
                print('new lr')
            #lrs.step(tensor_losses, i, run_path+'plot.pdf')  # lr schduler with atomatic scheduler
            for j in range(n_batches):
                #tensor_batch = tensor_data_with_noise[batch_size*j:batch_size*(j+1)]
                tensor_batch = tensor_data[batch_size*j:batch_size*(j+1)]
                optimizer.zero_grad() 
                tensor_output = model(tensor_batch)
                tensor_losses_sum, tensor_losses[i][j] = loss.loss(tensor_batch, tensor_output, model)
                #tensor_losses_sum, tensor_losses = loss.loss(tensor_batch, tensor_output, model)
                tensor_losses_sum.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0, error_if_nonfinite=False)
                optimizer.step()
            if display_info:
            #if i % 500 == 0 and i != 0: 
                print(f"Run: {run:3} | Epoch: {i:3} | lr: {lr_list[lr_counter]:10.7f} | total loss: {torch.mean(torch.sum(tensor_losses[i],1)):10.5f} | likelihood: {torch.mean(tensor_losses[i,:,0]):10.5f} | prior: {torch.mean(tensor_losses[i,:,1]):10.5f} | regularization {torch.mean(tensor_losses[i,:,2]):20.15f}")
                #print(f"Run: {run:3} | Epoch: {i:3} | lr: {lr_list[lr_counter]:10.7f} | total loss: {tensor_losses_sum:10.5f} | likelihood: {tensor_losses[0]:10.5f} | prior: {tensor_losses[1]:10.5f}")#" | regularization {torch.mean(tensor_losses[i,:,2]):20.15f}")
            torch.save(model.state_dict(), model_weight_name)

            # plotting and saving figure
            if i % 50 == 0: 
                plt.figure(i)
                #obs = data.get_tensor_data(1, [2.5, 1.5, -2, -2], [1, 3,5, 8])
                obs = data.get_tensor_data(1, [2.5, 1.5, -3, -3], [1, 3,5, 8])
                out_raw = model(obs)
                out = out_raw[0].detach().numpy()
                post, _ = data.get_tensor_posterior(obs[0])
                plt.plot(range(10), post, 'b')
                plt.plot(range(10), out, 'r')
                #plt.plot([1, 3, 5, 8], [2.5, 1.5, -2, -2], 'go')
                plt.plot([1, 3, 5, 8], [2.5, 1.5, -3, -3], 'go')
                plt.legend(['posterior', 'estimate', 'points'])
                plt.savefig('../saved_models/experiment/plot.pdf')

        torch.save(model.state_dict(), model_weight_name)
        print(f"Run: {run:3} | Epoch: {i:3} | lr: {lr_list[lr_counter]:7.5f} | total loss: {torch.mean(torch.sum(tensor_losses[i],1)):5.2f} | likelihood: {torch.mean(tensor_losses[i,:,0]):5.2f} | prior: {torch.mean(tensor_losses[i,:,1]):5.2f} | regularization {torch.mean(tensor_losses[i,:,2]):12.10f}")