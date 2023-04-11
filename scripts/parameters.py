import torch

def get_parameters():
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
    tensor_Q_m = torch.mm(torch.t(tensor_D), tensor_D) 
    n_param = tensor_D.size(dim=0)
    tau2 = 1000  # 1 is big
    tensor_Q_m_modified = tensor_Q_m + torch.eye(n_param)*(1./tau2) 
    tensor_Sigma_m = torch.inverse(tensor_Q_m_modified)
    sigma2_eps = 0.01
    tensor_Sigma_eps = torch.eye(n_param)*sigma2_eps
    tensor_mu_m = torch.zeros(n_param)
    tensor_mu_eps = tensor_mu_m 
    return tensor_mu_m, tensor_Sigma_m, tensor_mu_eps, tensor_Sigma_eps, tensor_Q_m, sigma2_eps, tau2


def get_parameters_1():
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
    tensor_Q_m = torch.mm(torch.t(tensor_D), tensor_D) 
    n_param = tensor_D.size(dim=0)
    tau2 = 1  # 1 is big
    tensor_Q_m_modified = tensor_Q_m + torch.eye(n_param)*(1./tau2) 
    tensor_Sigma_m = torch.inverse(tensor_Q_m_modified)
    sigma2_eps = 0.01
    tensor_Sigma_eps = torch.eye(n_param)*sigma2_eps
    tensor_mu_m = torch.zeros(n_param)
    tensor_mu_eps = tensor_mu_m 
    return tensor_mu_m, tensor_Sigma_m, tensor_mu_eps, tensor_Sigma_eps, tensor_Q_m, sigma2_eps, tau2


def get_parameters_2():
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
    tensor_Q_m = torch.mm(torch.t(tensor_D), tensor_D) 
    n_param = tensor_D.size(dim=0)
    tau2 = 1  # 1 is big
    tensor_Q_m_modified = tensor_Q_m + torch.eye(n_param)*(1./tau2) 
    tensor_Sigma_m = torch.inverse(tensor_Q_m_modified)
    sigma2_eps = 0.01
    tensor_Sigma_eps = torch.eye(n_param)*sigma2_eps
    tensor_mu_m = torch.zeros(n_param)
    tensor_mu_eps = tensor_mu_m 
    return tensor_mu_m, tensor_Sigma_m, tensor_mu_eps, tensor_Sigma_eps, tensor_Q_m, sigma2_eps, tau2


def get_parameters_3():
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
    tensor_Q_m = torch.mm(torch.t(tensor_D), tensor_D) 
    n_param = tensor_D.size(dim=0)
    tau2 = 0.01  # 1 is big
    tensor_Q_m_modified = tensor_Q_m + torch.eye(n_param)*(1./tau2) 
    tensor_Sigma_m = torch.inverse(tensor_Q_m_modified)
    sigma2_eps = 0.01
    tensor_Sigma_eps = torch.eye(n_param)*sigma2_eps
    tensor_mu_m = torch.zeros(n_param)
    tensor_mu_eps = tensor_mu_m 
    return tensor_mu_m, tensor_Sigma_m, tensor_mu_eps, tensor_Sigma_eps, tensor_Q_m, sigma2_eps, tau2


if __name__=="__main__":
    get_parameters() 