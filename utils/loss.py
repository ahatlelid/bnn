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


    def add_gaussian_noise(self, noise_variance, model, sign='positive'):
        #n_theta = torch.cat([param.view(-1) for param in model.parameters()]).size()[0]
        #tensor_mu_theta = torch.zeros(n_theta)
        #tensor_Sigma_theta = torch.eye(n_theta)*noise_variance
        #self.tensor_parameter_noise = MultivariateNormal(loc=tensor_mu_theta, covariance_matrix=tensor_Sigma_theta).sample(sample_shape=(1,))[0]
        self.tensor_parameter_noise = torch.load('../data/run1/theta_noise.pt')
        if sign == 'negative':
            self.tensor_parameter_noise *= -1


    def loss(self, tensor_input, tensor_output, model):
        #tensor_input = tensor_input[0]
        #tensor_output = tensor_output[0]
        tensor_Psi = tensor_output
        tensor_Gd = tensor_input[:,:self.n_param]
        tensor_mask = tensor_input[:,self.n_param:]
        #tensor_GPsi = tensor_Psi*tensor_mask

        # new
        tensor_GPsi = tensor_Psi*tensor_mask
        tensor_squared_error = torch.square(tensor_Psi - tensor_Gd)
        tensor_squared_error = tensor_squared_error*tensor_mask
        tensor_squared_error_sum = torch.sum(tensor_squared_error, [0, 1])
        tensor_likelihood_loss = tensor_squared_error_sum/self.sigma2_eps
        tensor_likelihood_loss *= 1#00#1_000_000 

        #lik_loss = 0
        #print(tensor_mask.shape[0])
        #print(tensor_mask.shape[1])
        #exit()
        #print(tensor_input.shape[0])
        #print(tensor_output.shape[0])
        #for i in range(tensor_mask.shape[0]):
        #    for j in range(tensor_mask.shape[1]):
        #        #print(tensor_mask[i][j])
        #        if tensor_mask[i][j] == 1:
        #            lik_loss += (tensor_Psi[i][j] - tensor_Gd[i][j])**2
        #lik_loss /= self.sigma2_eps
        #tensor_likelihood_loss = lik_loss
        #print(lik_loss)

        #print
        #n1 = tensor_Psi.detach().numpy()
        #n2 = tensor_Gd.detach().numpy()
        #n3 = tensor_mask.detach().numpy()
        #n4 = n1*n3
        #n5 = n2 - n4
        #n6 = n5**2
        #n7 = np.sum(n6, axis=1)
        #n8 = np.sum(n7)
        #n9 = n8/self.sigma2_eps
        #print(n9)


        # likelihood loss
        #tensor_squared_error = torch.square(tensor_GPsi - tensor_Gd)
        #tensor_squared_error_sum = torch.sum(tensor_squared_error, [0, 1])
        #tensor_likelihood_loss = tensor_squared_error_sum/self.sigma2_eps
        #tensor_likelihood_loss *= 1#00#1_000_000 
        #lik_loss = 0
        #print(tensor_likelihood_loss)

        # new

        # prior loss
        tensor_PsiQ_m = torch.matmul(tensor_Psi, self.tensor_Q_m)
        tensor_PsiQ_m = torch.unsqueeze(tensor_PsiQ_m, 1)
        tensor_Psi = torch.unsqueeze(tensor_Psi, 2)
        tensor_PsiQ_mPsi = torch.bmm(tensor_PsiQ_m, tensor_Psi).squeeze(2)
        tensor_prior_loss = torch.sum(tensor_PsiQ_mPsi)
        tensor_prior_loss *= 1#00#100#4_000_000
        #tensor_prior_loss = 0

        # L2 regularization loss
        tensor_parameters = torch.cat([param.view(-1) for param in model.parameters()])
        if self.tensor_parameter_noise != None:
            tensor_parameters = tensor_parameters + self.tensor_parameter_noise
        tensor_regularization_loss = torch.sum(tensor_parameters**2)
        tensor_regularization_loss_scaled = self.penalty_coefficient * tensor_regularization_loss
        tensor_regularization_loss_scaled = 0

        batch_size = tensor_input.size()[0]

        tensor_losses = torch.zeros(3)
        tensor_losses[0] = tensor_likelihood_loss / batch_size
        tensor_losses[1] = tensor_prior_loss / batch_size
        tensor_losses[2] = tensor_regularization_loss_scaled / batch_size
        tensor_losses_sum = (tensor_likelihood_loss + tensor_prior_loss)/batch_size + tensor_regularization_loss_scaled

        return tensor_losses_sum, tensor_losses