from importlib.resources import path
import torch
import numpy as np
import sys
sys.path.append("..")
from utils.model import Net_mask
from utils.loss import Loss
import os
import shutil

if __name__ == "__main__":
    """
    Main method for training 100 RML runs.
    Set size to 4 or 5 to train with 
    10^4 or 10^5 data respecievly.    
    """

    size = 4 # Train with 10**size data  
    N = int(10**size)

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

    data_file_path = f'../data/@/1.0e+0{size}/data/data.pt'
    tensor_data = torch.load(data_file_path)

    noise_path = f'../data/@/1.0e+0{size}/rml_noise/'
    save_path = f'../saved_models/testing2/rml/long/e{size}/'

    sigma2 = 0.01
    tau2 = 0.09
    batch_size = 1000
    data_fit_coefficient = 1./sigma2
    data_reg_coefficient = 1
    prior_coefficient = (1./tau2)*(batch_size/N) 
    tensor_coefficients = torch.Tensor([data_fit_coefficient, data_reg_coefficient, prior_coefficient])

    # create lr list
    lr_list = np.array([0.1, 0.01])
    lr_start = 0.001
    lr_minimum = 0.00001 
    lr_steps = 300
    lr_list = np.append(lr_list, np.linspace(lr_start, lr_minimum, lr_steps))
    lr_counter = 0

    # Loop to train 100 RML models
    for run in range(100):
        save_path_run = f'{save_path}{run}/'
        if os.path.exists(save_path_run) and os.path.isdir(save_path_run):
            shutil.rmtree(save_path_run)
        os.makedirs(save_path_run)

        model = Net_mask() 
        model.load_state_dict(torch.load(f'../data/@/weight_init/rml_e{size}_{run}.pth'))
        model_weight_name = f'{save_path_run}model_weights.pth'
        torch.save(model.state_dict(), model_weight_name)
        torch.save(tensor_coefficients, f'{save_path_run}coefficients.pt')

        optimizer = torch.optim.SGD(model.parameters(), lr=lr_list[0])
        loss = Loss(tensor_Q_m, data_fit_coefficient, data_reg_coefficient, prior_coefficient)

        # Load and add noise the data and parameters
        data_noise = torch.load(f'{noise_path}{run}/noise_data.pt')
        data_reg_noise = torch.load(f'{noise_path}{run}/noise_data_regularization.pt')
        tensor_data[:,:n_param] -= data_noise*tensor_data[:,n_param:]
        loss.add_gaussian_noise(filename=f'{noise_path}{run}/noise_parameter.pt')
        
        total_runs = len(lr_list)*int(10**7/N)
        training_losses = torch.zeros(len(lr_list)*10, 6)
        lr_counter = 0
        batch_counter = 0
        for j in range(total_runs):
            train_size_loops = int(N/1000)
            losses = torch.zeros(train_size_loops, 4)
            idx = torch.randperm(tensor_data.size(0)) 
            tensor_data = tensor_data[idx,:] # shuffle batch
            data_reg_noise = data_reg_noise[idx,:]
            for i in range(train_size_loops):
                tensor_batch_ = tensor_data[batch_size*i:batch_size*(i+1)]
                data_reg_noise_ = data_reg_noise[batch_size*i:batch_size*(i+1)] 

                optimizer.zero_grad() 
                tensor_output = model(tensor_batch_)
                tensor_losses_sum, tensor_losses_likelihood, tensor_losses_prior, tensor_losses_regularization = loss.loss(tensor_batch_, tensor_output, model, data_reg_noise=data_reg_noise_)
                losses[i][0] = tensor_losses_sum
                losses[i][1] = tensor_losses_likelihood
                losses[i][2] = tensor_losses_prior
                losses[i][3] = tensor_losses_regularization
                tensor_losses_sum.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0, error_if_nonfinite=False)
                optimizer.step()

            # Print info after 10^3 batches
            if j % int(10**6/N) == 0:
                print(f"e{size} Run: {run:3} | Batches (1000): {batch_counter:4}/{len(lr_list)*10} | lr: {lr_list[lr_counter]:10.7f} | {torch.mean(losses[:,0]):10.3f} {torch.mean(losses[:,1])*data_fit_coefficient:10.3f} {torch.mean(losses[:,2])*data_reg_coefficient:10.3f} {torch.mean(losses[:,3])*prior_coefficient:10.3f} | {torch.mean(losses[:,1])+torch.mean(losses[:,2])+torch.mean(losses[:,3]):10.3f} {torch.mean(losses[:,1]):10.3f} {torch.mean(losses[:,2]):10.3f} {torch.mean(losses[:,3]):10.3f}")
                training_losses[batch_counter,0] = batch_counter 
                training_losses[batch_counter,1] = lr_counter
                training_losses[batch_counter,2:] = torch.mean(losses, dim=0)
                torch.save(training_losses, f'{save_path_run}/training_losses.pt')
                batch_counter += 1

            # Change learning rate and store weights after 10^4 batches
            if j % int(10**7/N) == 0 and j != 0:  
                torch.save(model.state_dict(), f'{save_path_run}model_checkpoint_{lr_counter}.pth')
                # Set new learning rate
                for param in optimizer.param_groups:
                    param['lr'] = lr_list[lr_counter] 
                lr_counter += 1
                print('new lr')
            torch.save(model.state_dict(), model_weight_name)
        torch.save(torch.Tensor([0]), f'{save_path_run}finished_{run}.pt')
        print('============= new run =============')