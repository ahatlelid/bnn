import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

class Loss():
    def __init__(self, tensor_Q_m, data_fit_coefficient, data_reg_coefficient, prior_coefficient):
        self.n_param = tensor_Q_m.shape[0] 
        self.tensor_Q_m = tensor_Q_m
        self.data_fit_coefficient = data_fit_coefficient
        self.data_reg_coefficient = data_reg_coefficient
        self.prior_coefficient = prior_coefficient
        self.tensor_parameter_noise = None
        self.counter = 0


    def add_gaussian_noise(self, filename):
        self.tensor_parameter_noise = torch.load(filename)
        print('parameter_noise', self.tensor_parameter_noise)


    #def add_gaussian_noise(self, noise_variance=None, model=None, sign='positive', num=0):
    #    self.tensor_parameter_noise = torch.load(f'../data/rml/10_000/{num}/noise_param_{num}.pt')


    def loss(self, tensor_input, tensor_output, model, data_reg_noise=None):

        tensor_Psi = tensor_output
        tensor_Gd = tensor_input[:,:self.n_param]
        tensor_mask = tensor_input[:,self.n_param:]
        tensor_GPsi = tensor_Psi*tensor_mask

        # data residual loss
        tensor_squared_error = torch.square(tensor_GPsi - tensor_Gd)
        tensor_squared_error_sum = torch.sum(tensor_squared_error, [0, 1])
        tensor_data_fit_loss = tensor_squared_error_sum

        # data regularization loss
        if data_reg_noise is not None:
            tensor_Psi += data_reg_noise
        tensor_PsiQ_m = torch.matmul(tensor_Psi, self.tensor_Q_m)
        tensor_PsiQ_m = torch.unsqueeze(tensor_PsiQ_m, 1)
        tensor_Psi = torch.unsqueeze(tensor_Psi, 2)
        tensor_PsiQ_mPsi = torch.bmm(tensor_PsiQ_m, tensor_Psi).squeeze(2)
        tensor_data_regularization_loss = torch.sum(tensor_PsiQ_mPsi)

        # prior loss 
        tensor_parameters = torch.cat([param.view(-1) for param in model.parameters()])
        if self.tensor_parameter_noise != None:
            tensor_parameters = tensor_parameters + self.tensor_parameter_noise
        tensor_prior_loss = torch.sum(tensor_parameters**2)

        total_loss = \
            tensor_data_fit_loss*self.data_fit_coefficient + \
            tensor_data_regularization_loss*self.data_reg_coefficient + \
            tensor_prior_loss*self.prior_coefficient

        return total_loss, tensor_data_fit_loss, tensor_data_regularization_loss, tensor_prior_loss