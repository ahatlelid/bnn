import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

class Loss():
    def __init__(self, tensor_Q_m, sigma2_eps, penalty_coefficient):
        self.n_param = tensor_Q_m.shape[0] 
        self.tensor_Q_m = tensor_Q_m
        self.sigma2_eps = sigma2_eps
        self.penalty_coefficient = penalty_coefficient
        self.tensor_parameter_noise = None
        self.counter = 0


    def add_gaussian_noise(self, noise_variance=None, model=None, sign='positive', num=0):
        self.tensor_parameter_noise = torch.load(f'../data/rml/{num}/noise_param_{num}.pt')
        if sign == 'negative':
            self.tensor_parameter_noise *= -1


    def loss(self, tensor_input, tensor_output, model):
        original = True # can have different masks in batch
        with_G_matrix = False  # all masks must be equal within a batch
        with_masked_select = False # all masks must b eequal within a batch

        tensor_Psi = tensor_output
        tensor_Gd = tensor_input[:,:self.n_param]
        tensor_mask = tensor_input[:,self.n_param:]
        tensor_GPsi = tensor_Psi*tensor_mask

        if original:
            tensor_squared_error = torch.square(tensor_GPsi - tensor_Gd)
            tensor_squared_error_sum = torch.sum(tensor_squared_error, [0, 1])
            tensor_likelihood_loss = tensor_squared_error_sum/self.sigma2_eps

        # This block creates a G matrix. All masks must be equal in the batch
        if with_G_matrix:
            mask = tensor_mask[0].squeeze()
            ones = (mask == 1.).sum()
            G = torch.zeros(ones, self.n_param)
            idx = (mask == 1).nonzero()[:,0]
            for i, v in enumerate(idx):
                G[i,v] = 1
            G_transposed = torch.transpose(G, 1, 0)
            GPsi = torch.matmul(tensor_output, G_transposed)
            Gd = torch.matmul(tensor_input[:,:self.n_param], G_transposed)
            squared_diff = (GPsi - Gd)**2
            sum_squared_diff = torch.sum(squared_diff, dim=1)
            batch_sum_squared_diff = torch.sum(sum_squared_diff)
            tensor_likelihood_loss = batch_sum_squared_diff/self.sigma2_eps


        if with_masked_select: # selects observation with built in functionality. All masks must be equal in a batch
            mask = tensor_mask > 0.5
            d = torch.masked_select(tensor_Gd, mask)
            Psi = torch.masked_select(tensor_Psi, mask) 
            tensor_likelihood_loss = torch.sum(torch.square(d - Psi))/self.sigma2_eps


        # prior loss
        tensor_PsiQ_m = torch.matmul(tensor_Psi, self.tensor_Q_m)
        tensor_PsiQ_m = torch.unsqueeze(tensor_PsiQ_m, 1)
        tensor_Psi = torch.unsqueeze(tensor_Psi, 2)
        tensor_PsiQ_mPsi = torch.bmm(tensor_PsiQ_m, tensor_Psi).squeeze(2)
        tensor_prior_loss = torch.sum(tensor_PsiQ_mPsi)
        tensor_prior_loss *= 1

        # L2 regularization loss
        tensor_parameters = torch.cat([param.view(-1) for param in model.parameters()])
        if self.tensor_parameter_noise != None:
            tensor_parameters = tensor_parameters + self.tensor_parameter_noise
        tensor_regularization_loss = torch.sum(tensor_parameters**2)
        tensor_regularization_loss_scaled = self.penalty_coefficient * tensor_regularization_loss
        #tensor_regularization_loss_scaled *= 0

        batch_size = tensor_input.size()[0]

        tensor_losses_sum = (tensor_likelihood_loss + tensor_prior_loss)/batch_size + tensor_regularization_loss_scaled

        return tensor_losses_sum, tensor_likelihood_loss/batch_size, tensor_prior_loss/batch_size, tensor_regularization_loss_scaled/batch_size