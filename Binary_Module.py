import torch
import time
import numpy as np
import math
def approximate_sign(input):
    if input<-1:
        return -1
    elif input>=-1 and input<0:
        return 2*input+input**2
    elif input>=0 and input<1:
        return 2*input-input**2
    else:
        return 1
class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(self,input):
        self.save_for_backward(input)
        output=approximate_sign(input)
        return output
    @staticmethod
    def backward(self,grad_output):
        input,=self.saved_tensors
        grad_output[input.ge(1)] = 0
        grad_output[input.le(-1)] = 0
        return grad_output  
class BinaryConv2d(torch.nn.Conv2d):
    def __init__(self,*kargs,**kwargs):
        super(BinaryConv2d,self).__init__(*kargs,**kwargs)
    
    def forward(self,input):
        if input.size(1)!=3:
            input=Binarize.apply(input)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize.apply(self.weight.org)
        out = torch.nn.functional.conv2d(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
        return out

