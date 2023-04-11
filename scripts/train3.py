from re import I
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

    tau = 0.5
    l2_lambda = 0
    
    

    #span = 20
    #delay = 20 
    #required_reduction = 0.001
    #lowest_lr = 1e-5 

    #epochs_with_same_lr = 20
    #factor = 0.2
    #display_info = True


    #tensor_data = data_generator.get_tensor_data(n_data)


    n_param = 10
    batch_size = 1000

    model_weight_name = '../saved_models/aa/model_weights.pth'

    model = Net_mask() # TODO see if can move this outside of loop
    torch.save(model.state_dict(), model_weight_name)
    lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #lrs = lr_scheduler(optimizer, lr=lr, span=span, delay=delay, required_reduction=required_reduction, lowest_lr=lowest_lr, factor=factor)
    loss = Loss(tensor_Q_m, sigma2_eps, l2_lambda)

    counter = 0
    while True:
        tensor_losses = torch.zeros(1000, 3) # Store total loss, data loss, prior loss and regularization loss
        for i in range(1000):
            tensor_data = torch.zeros(batch_size, 2*n_param)
            tensor_d_sample =  MultivariateNormal(loc=torch.zeros(n_param), covariance_matrix=(torch.eye(10)*1)).sample(sample_shape=(batch_size,))
            tensor_n_masked = torch.randint(n_param, (batch_size,))
            tensor_masks = torch.rand(batch_size, n_param).argsort(dim=1)
            tensor_masks = (tensor_masks < tensor_n_masked.unsqueeze(1))*1
            tensor_data[:,:n_param] = tensor_d_sample*tensor_masks
            tensor_data[:,n_param:] = tensor_masks

            tensor_output = model(tensor_data)
            tensor_losses_sum, tensor_losses[i] = loss.loss(tensor_data, tensor_output, model)
            tensor_losses_sum.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0, error_if_nonfinite=False)
            optimizer.step()
        torch.save(model.state_dict(), model_weight_name)
        print(f"N baches (1000): {counter:3} | lr: {lr:7.5f} | total loss: {torch.mean(torch.sum(tensor_losses, axis=1)):5.2f} | likelihood: {torch.mean(tensor_losses[0,:]):5.2f} | prior: {torch.mean(tensor_losses[:, 1]):5.2f} | regularization {torch.mean(tensor_losses[:,2]):12.10f}")
        counter += 1
