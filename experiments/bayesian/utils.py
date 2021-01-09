# import torch
# import torch.nn.functional as F
# from torch import optim
# from torchcontrib.optim import SWA

# import numpy as np

# class swaLR(optim.lr_scheduler.LambdaLR):
#     def __init__(self, optimizer, steps_per_epoch, num_epochs, swa_dec_pct, swa_start_pct, swa_freq_pct, swa_lr_factor, last_epoch=-1, verbose=False):
#         if not isinstance(optimizer, SWA):
#             raise TypeError("Expecting a SWA optimizer")
        
#         lr_lambda = swa_learning_rate(num_epochs, swa_dec_pct, swa_start_pct, swa_lr_factor)
#         self.swa_start = int(swa_start_pct * steps_per_epoch * num_epochs)
#         self.swa_freq = int(swa_freq_pct * steps_per_epoch * num_epochs)
#         print("swa_start is {}, swa_freq is {}".format(self.swa_start, self.swa_freq))
#         self.steps_per_epoch = steps_per_epoch
#         self.num_epochs = num_epochs
#         super().__init__(optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch, verbose=verbose)
         
#     def step(self, epoch=None):
#         super().step(epoch)            
#         if (self._step_count + 1) > self.swa_start and (self._step_count + 1 - self.swa_start) % self.swa_freq == 0:
#             self.model.collect_model(self.model.base_model)
#             self.optimizer.update_swa()
        
def swa_learning_rate(steps_per_epoch, num_epochs, swa_dec_pct=.5, swa_start_pct=.75, swa_freq_pct=.05, swa_lr_factor=.5):
    def learning_rate_scheduler(epoch):
        t = epoch / num_epochs
        if t <= swa_dec_pct:
            factor = 1.0
        elif t <= swa_start_pct:
            factor = 1.0 - (1.0 - swa_lr_factor) * (t - swa_dec_pct) / (swa_start_pct - swa_dec_pct)
        else:
            factor = swa_lr_factor
        return factor
    return learning_rate_scheduler