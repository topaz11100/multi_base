"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""
## 给出layer的函数
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama=1.):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

## 这一块是给出
## 给出适合 snn 的残差链接方式
class BiRFKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""
    def __init__(self, d_model, N=2, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        ## 给出了原本的结果
        log_A_real = torch.log(0.5 * torch.ones(H, N))
        A_imag = math.pi * repeat(torch.arange(N), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """
        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
        return K
    

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


    
## 如果把自适应调节的方式添加到网络中，是不是可以实现较好的效果
class BiRFModel(nn.Module):
    def __init__(self, d_model, d_state=4, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.kernel = BiRFKernel(self.h, N=self.n, **kernel_args)
        self.dt = self.kernel.log_dt
        self.beta = 1.
        self.act1 = ZIF.apply
        self.act2 = ZIF.apply
        # self.D = nn.Parameter(torch.randn(self.h))
        # self.conv1 = CausalConvSoftmaxWeight(in_channels=d_model, kernel_size=4)
        # self.act = MaskedSlidingPSN()

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        
        # print(u.shape)
        # u = torch.from_numpy(u).to(dtype=torch.float32)
        L = u.size(-1)
        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # y = y + self.D.unsqueeze(dim=-1) * u

        s = self.act1(y.real - 1.)
        
        return s, y.real
    
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

class LIFModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.act = MaskedSlidingPSN()
        self.act = ZIF.apply
    def forward(self, u):
        return self.act(u - 0.5)
    
class SDTCM(nn.Module):
    def __init__(self, d_model, d_state=4, dropout=0.0, **kernel_args):
        super().__init__()
        self.h = d_model
        self.neuron1 = BiRFModel(d_model=d_model, d_state=d_state)
        self.neuron2 = BiRFModel(d_model=d_model, d_state=d_state)
        self.dp1 = nn.Dropout(dropout)
        
        self.neuron3 = LIFModel(d_model=d_model)
        self.lin1 = nn.Linear(2 * self.h, self.h)

        self.dp2 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(self.h, self.h)

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask

        s, _ = self.neuron1(u)
        rev_s, __ = self.neuron2(u.flip(dims=[-1])).flip(dims=[-1])
        s = torch.concat([s, rev_s], dim=1)

        y = self.lin1(self.dp1(s).transpose(-1, -2)).transpose(-1, -2)
        
        x = y + u

        s = self.neuron3(x)
        y = self.lin2(self.dp2(s).transpose(-1, -2)).transpose(-1, -2) + x
        
        return y, _ # Return a dummy state to satisfy this repo's interface, but this can be modified
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

if __name__=='__main__':
    model = SDTCM(d_model=1).cuda()
    x = torch.randn(16, 1, 32).cuda()
    print(model(x).shape)