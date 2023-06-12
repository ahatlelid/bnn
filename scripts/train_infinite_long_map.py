from importlib.resources import path
from re import I
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

import sys
sys.path.append("..")
from utils.model import Net_mask
from utils.loss import Loss

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

    lr_initial = 0.1
    lr_start = 0.001
    lr_minimum = 0.00001 
    lr_steps = 300
    lr_list = np.array([0.1, 0.01])
    lr_list = np.append(lr_list, np.linspace(lr_start, lr_minimum, lr_steps))
    lr_counter = 0


    total_runs = len(lr_list)*100

    path_to_output_folder = '../saved_models/testing/inf/map/'


    model = Net_mask() 
    model.load_state_dict(torch.load(f'../data/@/weight_init/map_inf.pth'))

    optimizer = torch.optim.SGD(model.parameters(), lr=lr_initial)

    batch_size = 1000

    tau2 = 0.09
    sigma2 = 0.01 

    data_fit_coefficient = 1./sigma2
    data_reg_coefficient = 1
    prior_coefficient = 1./tau2/100

    tensor_coefficients = torch.Tensor([data_fit_coefficient, data_reg_coefficient, prior_coefficient])
    torch.save(tensor_coefficients, f'{path_to_output_folder}coefficients.pt')

    loss = Loss(tensor_Q_m, data_fit_coefficient, data_reg_coefficient, prior_coefficient)


    n_param = 10
    n_data = 1000

    n_batch_runs = 1000

    size = 100
    total_runs = len(lr_list)*10
    training_losses = torch.zeros(len(lr_list)*10, 5)
    batch_counter = 0
    for j in range(total_runs):
        losses = torch.zeros(n_batch_runs, 4)
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

        print(f"MAP-inf | Batches (1000): {batch_counter:4}/{len(lr_list)*10} | lr: {lr_list[lr_counter]:10.7f} | {torch.mean(losses[:,0]):10.3f} {torch.mean(losses[:,1])*data_fit_coefficient:10.3f} {torch.mean(losses[:,2])*data_reg_coefficient:10.3f} {torch.mean(losses[:,3])*prior_coefficient:10.3f} | {torch.mean(losses[:,1])+torch.mean(losses[:,2])+torch.mean(losses[:,3]):10.3f} {torch.mean(losses[:,1]):10.3f} {torch.mean(losses[:,2]):10.3f} {torch.mean(losses[:,3]):10.3f}")
        training_losses[batch_counter,0] = batch_counter 
        training_losses[batch_counter,1:] = torch.mean(losses, dim=0)
        torch.save(training_losses, f'{path_to_output_folder}training_losses.pt')
        batch_counter += 1

        if j % 10 == 0 and j != 0:  # change learning rate after 10*n_batches*batch_size observations
            # set new learning rate
            for param in optimizer.param_groups:
                param['lr'] = lr_list[lr_counter] 
            lr_counter += 1
            print('new lr')
        torch.save(model.state_dict(), f'{path_to_output_folder}model_weights.pth')
    torch.save(torch.Tensor([0]), f'{path_to_output_folder}finished.pt')