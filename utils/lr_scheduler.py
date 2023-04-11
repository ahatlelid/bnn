import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

class lr_scheduler():
    def __init__(self, optimizer, lr, span, delay, required_reduction, lowest_lr, factor):
        self.optimizer = optimizer
        self.lr = lr
        self.span = span
        self.delay = delay
        self.required_reduction = required_reduction
        self.factor = factor
        self.delay_counter = 0
        self.lowest_lr = lowest_lr


    def step(self, tensor_losses, i, name): 
        tensor_total_batch_losses = torch.sum(tensor_losses, 2)
        tensor_mean_epoch_losses = torch.mean(tensor_total_batch_losses, 1)
        if self.delay_counter >= self.delay:
            if i > self.span:
                #tensor_mean_epoch_losses = torch.mean(tensor_losses, 1)
                tensor_mean_prev = torch.mean(tensor_mean_epoch_losses[i-(self.span+1):i])
                tensor_mean_current = torch.mean(tensor_mean_epoch_losses[i-self.span:i])

                if torch.abs(tensor_mean_prev - tensor_mean_current) < self.required_reduction*tensor_mean_prev:
                    if self.lr < self.lowest_lr:
                        return
                    self.update()
                    self.delay_counter = 0
                    plt.figure(0)
                    plt.plot(range(i), torch.mean(torch.sum(tensor_losses,2),1)[0:i].detach())
                    plt.yscale('log',base=10)
                    plt.title('Batch loss')
                    plt.xlabel('# batch')
                    plt.ylabel('Loss per batch / batch size')
                    plt.savefig(name)
        self.delay_counter += 1
    
    def update(self):
        self.lr = self.lr*self.factor
        for param in self.optimizer.param_groups:
            param['lr'] = self.lr 

