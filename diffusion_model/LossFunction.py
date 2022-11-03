"""
This contains the class implementation of the loss function, and allows the user to specify a 
loss function to use.
"""

import torch.nn.functional as F

class LossFunction:
    def __init__(self, loss_type='l1'):
        self.loss_type = loss_type
    def get_loss(self, noise, noise_pred):
        # L1 Loss
        if self.loss_type == 'l1':
            return F.l1_loss(noise, noise_pred)
        # Raise error if loss type is not one of the supported ones
        else:
            raise NotImplementedError()
            