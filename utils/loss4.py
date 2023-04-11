from re import I
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch.nn as nn

#class loss(nn.module):
class Loss():
    def __init__(self):
        #super(Loss, self).__init__()
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
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1],], dtype=torch.float64
        )
        self.tensor_Q_m = torch.mm(torch.t(tensor_D), tensor_D) 
        self.sigma2 = 0.01
        D = np.array(
            [[1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0,  1, -1, 0, 0, 0, 0, 0, 0],
            [0, 0,  0, 1, -1, 0, 0, 0, 0, 0],
            [0, 0,  0, 0, 1, -1, 0, 0, 0, 0],
            [0, 0,  0, 0, 0, 1, -1, 0, 0, 0],
            [0, 0,  0, 0, 0, 0, 1, -1, 0, 0],
            [0, 0,  0, 0, 0, 0, 0, 1, -1, 0],
            [0, 0,  0, 0, 0, 0, 0, 0, 1, -1],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
        )
        self.Q_m = np.transpose(D)@D
        tau2 = 1000  # 1 is big
        self.tensor_Q_m_modified = self.tensor_Q_m + torch.eye(10)*(1./tau2) 
        self.tensor_Sigma_m = torch.inverse(self.tensor_Q_m_modified)

    #def forward(self, tensor_input, tensor_output, model, c, i):
    def loss(self, tensor_input, tensor_output, model, c, i):

        inp = tensor_input[0].detach().numpy()
        out = tensor_output[0].detach().numpy()

        d = inp[:10]
        mask = inp[10:]
        GPsi = mask*out
        Gd = mask*d
        lik = np.sum((GPsi - Gd)**2)/self.sigma2
        reg = np.transpose(out)@self.Q_m@out
        reg = np.transpose(out)@out
        #reg = np.sum(np.gradient(out)**2)

        
        tensor_input = tensor_input[0]
        #print(tensor_input)
        tensor_output = tensor_output[0]
        #print(tensor_input)
        #print(tensor_output)

        tensor_Psi = tensor_output
        tensor_Gd = tensor_input[:10]
        tensor_mask = tensor_input[10:]
        tensor_GPsi = tensor_Psi*tensor_mask

        tensor_squared_error = torch.square(tensor_GPsi - tensor_Gd)
        tensor_squared_error_sum = torch.sum(tensor_squared_error)
        tensor_likelihood_loss = tensor_squared_error_sum/self.sigma2
        #print('lik ', tensor_likelihood_loss)

        #tensor_PsiQ_m = torch.matmul(torch.t(tensor_Psi), self.tensor_Q_m)
        #tensor_PsiQ_mPsi = torch.matmul(tensor_PsiQ_m, tensor_Psi)
        #tensor_PsiQ_mPsi = torch.matmul(torch.t(tensor_Psi), tensor_Psi)
        #tensor_PsiQ_mPsi = 0
        a = torch.gradient(tensor_output)
        b = torch.square(a[0])
        tensor_PsiQ_mPsi = torch.sum(b)
        tensor_PsiQ_mPsi *= 5
        #print(tensor_PsiQ_mPsi)


        #tensor_parameters = torch.cat([param.view(-1) for param in model.parameters()])
        #tensor_regularization_loss = torch.sum(tensor_parameters**2)
        #tensor_regularization_loss *= 1
        tensor_regularization_loss = 0
        

        #return tensor_likelihood_loss + tensor_PsiQ_mPsi, tensor_likelihood_loss, tensor_PsiQ_mPsi, torch.tensor(lik), torch.tensor(reg)
        #print(tensor_likelihood_loss + tensor_PsiQ_mPsi)
        return tensor_likelihood_loss + tensor_PsiQ_mPsi + tensor_regularization_loss, tensor_likelihood_loss, tensor_PsiQ_mPsi, lik, reg, tensor_regularization_loss


