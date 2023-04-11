
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt

class Data():
    def __init__(self, tensor_mu_m, tensor_Sigma_m, tensor_mu_eps, tensor_Sigma_eps):
        self.tensor_mu_m = tensor_mu_m
        self.tensor_Sigma_m = tensor_Sigma_m
        self.tensor_mu_eps = tensor_mu_eps
        self.tensor_Sigma_eps = tensor_Sigma_eps
        self.tensor_mu_d = tensor_mu_m
        self.tensor_Sigma_d = self.tensor_Sigma_m + self.tensor_Sigma_eps
        self.n_param = tensor_mu_m.size(dim=0)
        self.history = torch.zeros(1000, 20)
        self.counter = 0


    def get_tensor_data(self, n, values=None, positions=None, num=None):
        """Creating a tensor of data points and random masks"""
        if values is not None and positions is not None:
            tensor_data = torch.zeros(2*self.n_param)
            for v, p in zip(values, positions):
                tensor_data[p] = v
                tensor_data[p+10] = 1
            tensor_data = tensor_data.unsqueeze(0)
            return tensor_data
        else:
            tensor_data = torch.zeros(n, 2*self.n_param)
            tensor_d_sample =  MultivariateNormal(loc=self.tensor_mu_d, covariance_matrix=(torch.eye(10)*1)).sample(sample_shape=(n,))
            tensor_n_masked = torch.randint(self.n_param, (n,))
            if num is not None: 
                tensor_n_masked = torch.tensor([num])
            tensor_masks = torch.rand(n, self.n_param).argsort(dim=1)
            tensor_masks = (tensor_masks < tensor_n_masked.unsqueeze(1))*1
            tensor_data[:,:self.n_param] = tensor_d_sample*tensor_masks
            tensor_data[:,self.n_param:] = tensor_masks
            #print(tensor_data)
            if n == 1:
                self.history[self.counter] = tensor_data
                self.counter += 1
            return tensor_data

    def add_noise(self, tensor_data, sigma2_data_noise, sign='positive'):
        noise =  MultivariateNormal(loc=self.tensor_mu_eps, covariance_matrix=(torch.eye(self.n_param)*sigma2_data_noise)).sample(sample_shape=(tensor_data.size()[0],))
        if sign == 'negative':
            noise *= -1
        mask = tensor_data[:,self.n_param:]
        Gd = tensor_data[:,:self.n_param]
        noise = noise*mask
        tensor_data[:,:self.n_param] += noise
        return tensor_data


    def get_tensor_posterior(self, input_tensor):
        d = input_tensor[:self.n_param]
        mask = input_tensor[self.n_param:]
        location = torch.where(mask > 0.5)
        d = d[location]
        n_data = len(d)
        G = torch.zeros(n_data, self.n_param)
        for idx, elem in enumerate(location[0]):
            G[idx, elem] = 1
        Sigma_eps = torch.matmul(G, torch.matmul(self.tensor_Sigma_eps, torch.t(G)))
        mu_eps = torch.matmul(G, self.tensor_mu_eps)
    
        #matrix calculations
        Sigma_mm = self.tensor_Sigma_m
        Sigma_dm = torch.matmul(G,Sigma_mm)
        Sigma_md = torch.matmul(Sigma_mm, torch.t(G))
        Sigma_dd = torch.matmul(G, torch.matmul(Sigma_mm, torch.t(G))) + Sigma_eps
        Sigma_dd_inv = torch.inverse(Sigma_dd)
        mu_m_d = self.tensor_mu_m + torch.matmul(torch.matmul(Sigma_md, Sigma_dd_inv),(d-mu_eps))
        Sigma_m_d = Sigma_mm - torch.matmul(torch.matmul(Sigma_md, Sigma_dd_inv), Sigma_dm)
        return mu_m_d, Sigma_m_d