import torch
import torch.nn as nn


class DentReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p):
        ctx.save_for_backward(input)
        ctx.p = p
        output = input.clone()
        mask1 = p <= input
        mask2 = input <= 0
        output[mask1 & mask2] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        p = ctx.p
        grad_input = grad_output.clone()
        mask1 = p <= input
        mask2 = input <= 0
        grad_input[mask1 & mask2] = 0
        return grad_input, None


class DentReLU(nn.Module):
    def __init__(self, p: float = -0.2):
        super(DentReLU, self).__init__()
        self.p = p

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return DentReLUFunction.apply(input, self.p)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "p={}".format(self.p)
