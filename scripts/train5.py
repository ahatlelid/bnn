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
    #sigma2_eps = 0.1

    n_data = 100000
    batch_size = 1000
    n_batches = round(n_data/batch_size)
    n_epochs = 1000
    lr = 0.1
    tau = 0.5
    l2_lambda = 1./(n_data*tau**2)
    
    min_delay = 10
    runs=1
    epochs_with_same_lr = 20
    sign = 'positive' # positive, negative
    init_type = 'default' # map, positive, negative, default
    outer_folder = 'map_inf2'

    #lr = 0.1
    span = 100
    delay = 100 
    required_reduction = 0.001
    lowest_lr = 1e-5 
    factor = 0.5
    display_info = True

    # 
    dir_path = '../saved_models/' + str(outer_folder)
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    tensor_data = data_generator.get_tensor_data(n_data)

    for run in range(runs):
        run_path = dir_path + '/' + str(run)
        os.makedirs(run_path)
        model = Net_mask() # TODO see if can move this outside of loop
        model_weight_name = run_path + "/model_weights.pth"
        #model.load_state_dict(torch.load("../saved_models/map/0/model_weights.pth"))
        torch.save(model.state_dict(), model_weight_name)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        lrs = lr_scheduler(optimizer, lr=lr, span=span, delay=delay, required_reduction=required_reduction, lowest_lr=lowest_lr, factor=factor)
        loss = Loss(tensor_Q_m, sigma2_eps, l2_lambda)

        n_batches = 1000
        tensor_losses = torch.zeros((n_epochs, n_batches, 3)) # Store total loss, data loss, prior loss and regularization loss
        model_weight_name = run_path + "/model_weights.pth"

        #model = set_weights(model, init_type=init_type, variance=0.5**2, path='../saved_models/map/0/model_weights.pth')

        if run == 0:
            tensor_data_with_noise = tensor_data.detach().clone()
        else: # run=0 is MLE estimate
            tensor_data_with_noise = tensor_data.detach().clone()
            tensor_data_with_noise = data_generator.add_noise(tensor_data_with_noise, sigma2_data_noise=sigma2_eps, sign=sign)
            loss.add_gaussian_noise(noise_variance=tau**2, model=model, sign=sign)

        for i in range(n_epochs):
            #if i % epochs_with_same_lr == 0 and i != 0: lrs.update()
            lrs.step(tensor_losses, i, run_path+'plot.pdf')
            for j in range(n_batches):
                #tensor_batch = tensor_data_with_noise[batch_size*j:batch_size*(j+1)]
                #tensor_batch =  MultivariateNormal(loc=torch.zeros(10), covariance_matrix=(torch.eye(10)*1)).sample(sample_shape=(batch_size,))
                tensor_batch = data_generator.get_tensor_data(batch_size)
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