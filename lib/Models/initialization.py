import torch.nn as nn
from torch.nn import init


class WeightInit:
    """
    Class for weight-initialization. Would have been nice to just inherit
    but PyTorch does not have a class for weight initialization. However methods
    for weight initialization are imported and used from the following script:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py

    Parameters:
        init_scheme (str): Weight-initialization scheme

    Attributes:
        init_methods_dict (dict): Dictionary of available layer initialization methods.
            Contains: "xavier-uniform", "xavier-normal", "kaiming-uniform", "kaiming-normal"
        init_method (function): Weight-initialization function according to init_scheme. If
            initialization function doesn't exist, "kaiming-normal" is used as default.
    """

    def __init__(self, init_scheme):
        self.init_scheme = init_scheme

        # dictionary containing valid weight initialization techniques
        self.init_methods_dict = {'xavier-normal': init.xavier_normal_,
                                  'xavier-uniform': init.xavier_uniform_,
                                  'kaiming-normal': init.kaiming_normal_,
                                  'kaiming-uniform': init.kaiming_uniform_}

        if self.init_scheme in self.init_methods_dict:
            self.init_method = self.init_methods_dict[self.init_scheme]
        else:
            print("weight-init scheme doesn't exist - using kaiming_normal instead")
            self.init_method = self.init_methods_dict['kaiming-normal']

    def init_model(self, model):
        """
        Loops over all convolutional, fully-connexted and batch-normalization
        layers and calls the layer_init function for each module (layer) in
        the model to initialize weights and biases for the whole model.

        Parameters:
            model (torch.nn.Module): Model architecture
        """

        for m in model.modules():
            # weight and batch-norm initialization
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self.layer_init(m)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def layer_init(self, m):
        """
        Initializes a single module (layer).

        Parameters:
            m (torch.nn.Module): Module (layer) of the model
        """

        self.init_method(m.weight.data)

        if not isinstance(m.bias, type(None)):
            m.bias.data.fill_(0)
