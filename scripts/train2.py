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
    # Setting all values.
    tensor_mu_m, tensor_Sigma_m, tensor_mu_eps, tensor_Sigma_eps, tensor_Q_m, sigma2_eps, tau2 = get_parameters()
    data_generator = Data(tensor_mu_m, tensor_Sigma_m, tensor_mu_eps, tensor_Sigma_eps)

    n_data = 100000
    #n_data = int(n_data*4)
    batch_size = 1000
    n_batches = round(n_data/batch_size)
    n_epochs = 100
    lr = 0.1
    tau = 0.5
    l2_lambda = 1./(n_data*tau**2)
    
    runs=10
    noisy = False
    init_weights = False
    sign = 'positive' # positive, negative
    init_type = 'negative' # map, positive, negative, default
    outer_folder = 'newnew'
    #save = 3
    

    # parameters to be set if using lr scheduler automatic
    span = 20
    delay = 20 
    required_reduction = 0.001
    lowest_lr = 1e-5 

    epochs_with_same_lr = 20
    factor = 0.2
    display_info = True

    dir_path = '../saved_models/' + str(outer_folder)
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    #tensor_data = data_generator.get_tensor_data(n_data)
    tensor_data = torch.load('../data/run1/data_n_100000_var_1.pt')

    for run in range(runs):
        run_path = dir_path + '/' + str(run)
        os.makedirs(run_path)
        model = Net_mask() # TODO see if can move this outside of loop
        #model.load_state_dict(torch.load('../data/run1/model_weights.pth'))
        model_weight_name = run_path + "/model_weights.pth"
        torch.save(model.state_dict(), model_weight_name)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        lrs = lr_scheduler(optimizer, lr=lr, span=span, delay=delay, required_reduction=required_reduction, lowest_lr=lowest_lr, factor=factor)
        loss = Loss(tensor_Q_m, sigma2_eps, l2_lambda)

        tensor_losses = torch.zeros((n_epochs, n_batches, 3)) # Store total loss, data loss, prior loss and regularization loss
        model_weight_name = run_path + "/model_weights.pth"

        #if noisy:
        #    noise = torch.load('../data/run1/data_noise_n_100000_var_0_1.pt')
        #    mask = tensor_data[:,10:]
        #    Gd = tensor_data[:,:10]
        #    noise = noise*mask
        #    if sign == 'negative':
        #        noise *= -1
        #    tensor_data[:,:10] += noise
        #    #tensor_data_with_noise = tensor_data
        #tensor_data_with_noise = tensor_data

        #if run == 0: # run=0 is MLE estimate
        #    tensor_data_with_noise = tensor_data.detach().clone()
        #else: 
        #model = set_weights(model, init_type=init_type, variance=0.5**2, path='../saved_models/map/0/model_weights.pth')
        tensor_data_with_noise = tensor_data.detach().clone()
        tensor_data_with_noise = data_generator.add_noise(tensor_data_with_noise, sigma2_data_noise=sigma2_eps, sign=sign)
        loss.add_gaussian_noise(noise_variance=tau**2, model=model, sign=sign)

        if noisy:
            loss.add_gaussian_noise(noise_variance=tau**2, model=model, sign=sign)

        if init_weights:
            model = set_weights(model, init_type=init_type, variance=0.5**2, path='../saved_models/map/0/model_weights.pth')

        for i in range(n_epochs):
            # shuffle indexes
            idx = torch.randperm(tensor_data.size(0))
            tensor_data = tensor_data[idx,:]
            if i % epochs_with_same_lr == 0 and i != 0: 
                lrs.update()
                print('new lr')
            #lrs.step(tensor_losses, i, run_path+'plot.pdf')  # lr schduler with atomatic scheduler
            for j in range(n_batches):
                tensor_batch = tensor_data_with_noise[batch_size*j:batch_size*(j+1)]
                optimizer.zero_grad() 
                tensor_output = model(tensor_batch)
                tensor_losses_sum, tensor_losses[i][j] = loss.loss(tensor_batch, tensor_output, model)
                tensor_losses_sum.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0, error_if_nonfinite=False)
                optimizer.step()
            if display_info:
                print(f"Run: {run:3} | Epoch: {i:3} | lr: {lrs.lr:7.5f} | total loss: {torch.mean(torch.sum(tensor_losses[i],1)):5.2f} | likelihood: {torch.mean(tensor_losses[i,:,0]):5.2f} | prior: {torch.mean(tensor_losses[i,:,1]):5.2f} | regularization {torch.mean(tensor_losses[i,:,2]):12.10f}")
            torch.save(model.state_dict(), model_weight_name)
        torch.save(model.state_dict(), model_weight_name)
        print(f"Run: {run:3} | Epoch: {i:3} | lr: {lrs.lr:7.5f} | total loss: {torch.mean(torch.sum(tensor_losses[i],1)):5.2f} | likelihood: {torch.mean(tensor_losses[i,:,0]):5.2f} | prior: {torch.mean(tensor_losses[i,:,1]):5.2f} | regularization {torch.mean(tensor_losses[i,:,2]):12.10f}")