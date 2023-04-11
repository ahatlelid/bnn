

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import sys
sys.path.append("..")


class Data():
    #def __init__(self, tensor_mu_m, tensor_Sigma_m, tensor_mu_eps, tensor_Sigma_eps):
    #    self.tensor_mu_m = tensor_mu_m
    #    self.tensor_Sigma_m = tensor_Sigma_m
    #    self.tensor_mu_eps = tensor_mu_eps
    #    self.tensor_Sigma_eps = tensor_Sigma_eps
    #    self.tensor_mu_d = tensor_mu_m
    #    self.tensor_Sigma_d = self.tensor_Sigma_m + self.tensor_Sigma_eps
    #    self.n_param = tensor_mu_m.size(dim=0)

        #self.tensor_D = tensor_D
        #self.sigma2_eps = sigma2_eps
        #self.inversion_noise = inversion_noise
    def __init__(self):
        pass


    def generate_data(self, n, n_param, sigma2_eps):
        tensor_data = torch.zeros(n, 2*n_param)
        tensor_d_sample =  MultivariateNormal(loc=torch.zeros(n_param), covariance_matrix=(torch.eye(n_param)*sigma2_eps)).sample(sample_shape=(n,))
        tensor_n_masked = torch.randint(n_param, (n,))
        tensor_masks = torch.rand(n, n_param).argsort(dim=1)
        tensor_masks = (tensor_masks < tensor_n_masked.unsqueeze(1))*1
        tensor_data[:,:n_param] = tensor_d_sample*tensor_masks
        tensor_data[:,n_param:] = tensor_masks
        return tensor_data






if __name__ == "__main__":
    data = Data()
    tens_data = data.generate_data(100_000, 10, 0.1)
    torch.save(tens_data, '../data/new/test_data_n100_000_var0_1.pt')