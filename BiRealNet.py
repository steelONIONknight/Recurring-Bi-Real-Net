import torch
import numpy as np
from collections import OrderedDict
from Binary_Module import BinaryConv2d


class BiRealNet(torch.nn.Module):
    def __init__(self,n_class=10,input_size=32):
        super(BiRealNet,self).__init__()
        
