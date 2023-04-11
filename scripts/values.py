import torch

d = {} # d is a dictonary holding all values needed for training
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
tau2 = 1000 # 1/tau2 is the noise added
tensor_Q_m_modified = tensor_Q_m + torch.eye(n_param)*(1./tau2)
sigma2_eps = 0.01
tensor_mu_m = torch.zeros(n_param)

d['tensor_Q_m'] = tensor_Q_m 
d['n_param'] = n_param
d['tau2'] = 1000  
d['tensor_Q_m_modified'] = tensor_Q_m_modified
d['tensor_Sigma_m'] = torch.inverse(tensor_Q_m_modified)
d['sigma2_eps'] = sigma2_eps
d['tensor_Sigma_eps'] = torch.eye(n_param)*sigma2_eps
d['tensor_mu_m'] = tensor_mu_m
d['tensor_mu_eps'] = tensor_mu_m 

# data options
d['generate_data'] = True
d['generate_data_noise'] = False
d['generate_parameter_noise'] = False
d['generate_initial_weight_values'] = False

# path to data folders
d['path_to_data_folder'] = '../data/run1/data_n_100000_var_1.pt'
d['path_to_data_noise_folder'] = '../data/..'
d['path_to_parameter_noise_folder'] = '../data/..'
d['path_to_weight_initial_values'] = '../data/..'

# output folder
d['path_to_output_folder'] = '../saved_models/experiment/'

# runs
d['number_of_runs'] = 10

# data options
number_of_data = 100000
d['number_of_data'] = number_of_data
d['batch_size'] = 1000
d['n_batches'] = 100 # n_batches*batch_size=number_of_data
d['n_epochs'] = 100

# scaling factors
prior_variance = 0.5**2
d['prior_variance'] = prior_variance
d['data_fit_scale_factor'] = 0.01
d['regularization_scale_factor'] = 1
d['prior_scale_factor'] = 1./(number_of_data*prior_variance)

# randomized maximum likelihood
d['sign_of_noise'] = 'positive'

# initial weight
d['initial_weight'] = 'default'

# learning rate options
d['initial_lr'] = 0.1
d['lowest_lr'] = 1e-5
d['reduction_factor'] = 0.2
d['lr_automatic_ajust'] = False # True if lr should drop depenidng on loss, False if it should drop depending on epoch
d['epochs_with_same_lr'] = 50
d['required_reduction'] = 0.001
d['span'] = 20
d['delay'] = 20 
d['display_info'] = True


torch.save(d, 'stored_values/values.pt')