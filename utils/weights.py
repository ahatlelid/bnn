import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import sys
sys.path.append("..")


def set_weights(model, init_type, variance, path):
    if init_type == 'default':
        return model
    elif init_type == 'map':
        model.load_state_dict(torch.load(path))
        return model
    else: 
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #noise =  MultivariateNormal(loc=torch.zeros(n), covariance_matrix=(torch.eye(n)*variance)).sample(sample_shape=(tensor_data.size()[0],))
        noise =  MultivariateNormal(loc=torch.zeros(n), covariance_matrix=(torch.eye(n)*variance)).sample(sample_shape=(1,))[0]
        if init_type == 'negative':
            print('negative')
            noise *= -1
        counter = 0
        with torch.no_grad():
            for i in range(len(list(model.parameters()))):
                #print(list(model.parameters())[i].shape)
                if len(list(model.parameters())[i].shape) > 1:
                    for j in range(list(model.parameters())[i].shape[0]):
                        for k in range(len(list(model.parameters())[i][j])):
                            #print(list(model.parameters())[i][j][k])
                            list(model.parameters())[i][j][k] = noise[counter]
                            counter += 1
                else:
                    #print(list(model.parameters())[i].shape)
                    for j in range(list(model.parameters())[i].shape[0]):
                        #print(list(model.parameters())[i][j])
                        list(model.parameters())[i][j] = noise[counter]
                        counter += 1
        return model