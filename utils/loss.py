import torch

class Loss():
    """Class to calculte sum of loss per batch."""

    def __init__(self, 
            tensor_Q_m,
            data_fit_coefficient,
            data_reg_coefficient,
            prior_coefficient
        ):
        self.n_param = tensor_Q_m.shape[0] 
        self.tensor_Q_m = tensor_Q_m
        self.data_fit_coefficient = data_fit_coefficient
        self.data_reg_coefficient = data_reg_coefficient
        self.prior_coefficient = prior_coefficient
        self.tensor_parameter_noise = None

    def add_gaussian_noise(self, filename):
        """Set parameter noise for RML as instance variable."""
        self.tensor_parameter_noise = torch.load(filename)

    def loss(self, tensor_input, tensor_output, model, data_reg_noise=None):
        """
        Calcluates the loss per batch.
        For RML, give regularization noise as argument. 
        """

        # Extract predictions for the observed data
        tensor_Psi = tensor_output
        tensor_Gd = tensor_input[:,:self.n_param]
        tensor_mask = tensor_input[:,self.n_param:]
        tensor_GPsi = tensor_Psi*tensor_mask

        # Sum of data fit loss for batch
        tensor_squared_error = torch.square(tensor_GPsi - tensor_Gd)
        tensor_squared_error_sum = torch.sum(tensor_squared_error, [0, 1])
        tensor_data_fit_loss = tensor_squared_error_sum

        # Sum of data regularization loss for batch
        if data_reg_noise is not None:
            tensor_Psi -= data_reg_noise
        tensor_PsiQ_m = torch.matmul(tensor_Psi, self.tensor_Q_m)
        tensor_PsiQ_m = torch.unsqueeze(tensor_PsiQ_m, 1)
        tensor_Psi = torch.unsqueeze(tensor_Psi, 2)
        tensor_PsiQ_mPsi = torch.bmm(tensor_PsiQ_m, tensor_Psi).squeeze(2)
        tensor_data_regularization_loss = torch.sum(tensor_PsiQ_mPsi)

        # Prior loss 
        tensor_parameters = torch.cat([param.view(-1) for param in model.parameters()])
        if self.tensor_parameter_noise != None:
            tensor_parameters = tensor_parameters - self.tensor_parameter_noise
        tensor_prior_loss = torch.sum(tensor_parameters**2)

        # Adding scaled losses together
        total_loss = \
            tensor_data_fit_loss*self.data_fit_coefficient + \
            tensor_data_regularization_loss*self.data_reg_coefficient + \
            tensor_prior_loss*self.prior_coefficient

        return total_loss, tensor_data_fit_loss, \
            tensor_data_regularization_loss, tensor_prior_loss