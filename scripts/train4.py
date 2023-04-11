import torch
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from utils.model import Net_mask
#from utils.loss4 import Loss
from utils.lr_scheduler import lr_scheduler
#
import os
import shutil

from torch.distributions.multivariate_normal import MultivariateNormal

if __name__ == "__main__":
    # Setting all values.

    batch_size = 1000
    lr = 0.1
    lr_reduction_factor = 0.2
    

    model = Net_mask()
    model = model.double()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    from utils.loss4 import Loss
    loss = Loss()

    run = 0
    outer_folder = 'map_testing'
    dir_path = '../saved_models/' + str(outer_folder)
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    run_path = dir_path + '/' + str(run)
    os.makedirs(run_path)
    model_weight_name = run_path + "/model_weights.pth"
    torch.save(model.state_dict(), model_weight_name)

    epochs_with_same_lr = 30
    counter = 0
    while True:
        loss_tens = torch.zeros(batch_size, 6)
        if counter % epochs_with_same_lr == 0 and counter != 0:
            lr = lr*lr_reduction_factor
            for param in optimizer.param_groups:
                param['lr'] = lr
        for i in range(batch_size):
            tensor_data = torch.zeros(1, 20)
            tensor_data = tensor_data.type(torch.DoubleTensor) #new
            tensor_d_sample =  MultivariateNormal(loc=torch.zeros(10), covariance_matrix=(torch.eye(10)*1)).sample(sample_shape=(1,))
            tensor_n_masked = torch.randint(10, (1,))
            tensor_masks = torch.rand(1, 10).argsort(dim=1)
            tensor_masks = (tensor_masks < tensor_n_masked.unsqueeze(1))*1
            tensor_data[:,:10] = tensor_d_sample*tensor_masks
            tensor_data[:,10:] = tensor_masks

            optimizer.zero_grad() 
            tensor_output = model(tensor_data)
            tensor_loss, likelihood_loss, prior_loss, lik, reg, pri = loss.loss(tensor_data, tensor_output, model, counter, i)

            #tensor_loss.backward(retain_graph=True, create_graph=True)
            tensor_loss.backward()
            loss_tens[i,0] = tensor_loss
            loss_tens[i,1] = likelihood_loss
            loss_tens[i,2] = prior_loss
            loss_tens[i,3] = lik
            loss_tens[i,4] = reg
            loss_tens[i,5] = pri


            #if i < 20:
            #for p in model.parameters():
            #    for j in p:
            #        if len(j.shape) > 0:
            #            for k in j:
            #                if abs(k) > 10: list(model.parameters())[p][j][k] = torch.tensor(10.)
            #                #print(k)
            #        else:
            #            if abs(j) > 10: list(model.parameters())[p][j] = torch.tensor(10.)
                        #print(j)

            #with torch.no_grad():
            #    for i in range(len(list(model.parameters()))):
            #        #print(list(model.parameters())[i].shape)
            #        if len(list(model.parameters())[i].shape) > 1:
            #            for j in range(list(model.parameters())[i].shape[0]):
            #                for k in range(len(list(model.parameters())[i][j])):
            #                    #print(list(model.parameters())[i][j][k])
            #                    if abs(list(model.parameters())[i][j][k]) > 10:
            #                        list(model.parameters())[i][j][k] = 10
            #        else:
            #            #print(list(model.parameters())[i].shape)
            #            for j in range(list(model.parameters())[i].shape[0]):
            #                #print(list(model.parameters())[i][j])
            #                if abs(list(model.parameters())[i][j]) > 10:
            #                    list(model.parameters())[i][j] = 10
            
            #else:
            #    exit()
            #threshold = 10
            #for p in model.parameters():
            #        p.data.clamp_(-1., 1.)
                    

            #print('\n\n\n\n\n\n')
            #if i >10: exit()
            #   for p in model.parameters():
            #        print(p)
            #else:
            #    exit()
            #else:
            #    exit()
                #if p.grad.norm() > threshold:
                    #print(p.grad.norm())
                #    torch.nn.utils.clip_grad_norm_(p, threshold)
                    #print(p.grad.norm())
            #exit()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=1, error_if_nonfinite=False) # always zero where there is no points
            optimizer.step()
            #torch.nn.utils.clip_grad_value_(model.parameters(), 1)



        torch.save(model.state_dict(), model_weight_name)
        # | numpy reg: {torch.mean(reg[:,4]):10.5f} | pri: {torch.mean(loss_tens[:,5]):10.5f}")
        print(f"Batch: {counter:3} lr: {lr:7.5f} | total: {torch.mean(loss_tens[:,0]):5.2f} | lik: {torch.mean(loss_tens[:,1]):7.3f} | reg: {torch.mean(loss_tens[:,2]):7.3f} | n_lik: {torch.mean(loss_tens[:,3]):7.3f} | n_reg: {torch.mean(loss_tens[:,4]):7.3f} | pri: {torch.mean(loss_tens[:,5]):7.5f}.")
        counter += 1