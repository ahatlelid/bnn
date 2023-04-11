import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class Loss():
    def __init__(self, tensor_Q_m, sigma2_eps, penalty_coefficient):
        self.n_param = tensor_Q_m.shape[0] 
        self.tensor_Q_m = tensor_Q_m
        self.sigma2_eps = sigma2_eps
        self.penalty_coefficient = penalty_coefficient
        self.tensor_parameter_noise = None
        self.counter = 0
        #print(self.tensor_Q_m)
        #exit


    def add_gaussian_noise(self, noise_variance, model, sign='positive'):
        n_theta = torch.cat([param.view(-1) for param in model.parameters()]).size()[0]
        tensor_mu_theta = torch.zeros(n_theta)
        tensor_Sigma_theta = torch.eye(n_theta)*noise_variance
        self.tensor_parameter_noise = MultivariateNormal(loc=tensor_mu_theta, covariance_matrix=tensor_Sigma_theta).sample(sample_shape=(1,))[0]
        if sign == 'negative':
            self.tensor_parameter_noise *= -1


    def loss(self, tensor_input, tensor_output):
        print(tensor_input)
        print(tensor_output)
        tensor_Psi = tensor_output
        tensor_Gd = tensor_input[:,:self.n_param]
        tensor_mask = tensor_input[:,self.n_param:]
        tensor_GPsi = tensor_Psi*tensor_mask

        # likelihood loss
        print(tensor_GPsi - tensor_Gd)
        tensor_squared_error = torch.square(tensor_GPsi - tensor_Gd)
        print(tensor_squared_error)
        tensor_squared_error_sum = torch.sum(tensor_squared_error, [0, 1])
        print(tensor_squared_error_sum)
        tensor_likelihood_loss = tensor_squared_error_sum/self.sigma2_eps

        # prior loss
        tensor_PsiQ_m = torch.matmul(tensor_Psi, self.tensor_Q_m)
        tensor_PsiQ_m = torch.unsqueeze(tensor_PsiQ_m, 1)
        tensor_Psi = torch.unsqueeze(tensor_Psi, 2)
        tensor_PsiQ_mPsi = torch.bmm(tensor_PsiQ_m, tensor_Psi).squeeze(2)
        tensor_prior_loss = torch.sum(tensor_PsiQ_mPsi)

        batch_size = tensor_input.size()[0]

        tensor_losses = torch.zeros(2)
        tensor_losses[0] = tensor_likelihood_loss / batch_size
        tensor_losses[1] = tensor_prior_loss / batch_size
        tensor_losses_sum = (tensor_likelihood_loss + tensor_prior_loss)/batch_size

        print('hey')
        print(batch_size)
        print(tensor_likelihood_loss)
        print(self.sigma2_eps)
        return tensor_losses_sum, tensor_losses