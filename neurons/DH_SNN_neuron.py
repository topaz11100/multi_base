import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

surrograte_type = 'MG'
print('gradient type: ', surrograte_type)


gamma = 0.5
lens = 0.5
R_m = 1


beta_value = 1.8
b_j0_value = 0.01



def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

# define approximate firing function

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        if surrograte_type == 'G':
            temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        #multi gaussian
        elif surrograte_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif surrograte_type =='linear':
            temp = F.relu(1-input.abs())
        elif surrograte_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        elif surrograte_type == 'rect':
            temp = input.abs() < 0.5
        return grad_input * temp.float()*gamma
    

    
act_fun_adp = ActFun_adp.apply    



def mem_update_pra(inputs, mem, spike, v_th, tau_m, dt=1,device=None):
    """
    neural model with soft reset
    """   
    alpha = torch.sigmoid(tau_m)
    mem = mem * alpha  + (1 - alpha) * R_m * inputs-v_th*spike
    inputs_ = mem - v_th

    spike = act_fun_adp(inputs_)  
    return mem, spike


def mem_update_pra_noreset(inputs, mem, spike, v_th, tau_m, dt=1,device=None):
    """
    neural model without reset
    Args:
        input(float): soma input.
        mem(float): soma membrane potential
        spike(int): spike or not spike
        vth(float): threshold
        tau_m(float): time factors of soma
    """   
    alpha = torch.sigmoid(tau_m)
    #without reset
    mem = mem * alpha  + (1 - alpha) * R_m * inputs#-v_th*spike
    inputs_ = mem - v_th

    spike = act_fun_adp(inputs_)  
    return mem, spike
def mem_update_pra_hardreset(inputs, mem, spike, v_th, tau_m, dt=1,device=None):
    """
    neural model with hard reset
    Args:
        input(float): soma input.
        mem(float): soma membrane potential
        spike(int): spike or not spike
        vth(float): threshold
        tau_m(float): time factors of soma
    """   
    alpha = torch.sigmoid(tau_m)
    #hard reset
    mem = mem * alpha*(1-spike)  + (1 - alpha) * R_m * inputs#-v_th*spike
    inputs_ = mem - v_th

    spike = act_fun_adp(inputs_)  
    return mem, spike


def output_Neuron_pra(inputs, mem, tau_m, dt=1,device=None):
    """
    The read out neuron is leaky integrator without spike
    Args:
        input(float): soma input.
        mem(float): soma membrane potential
        tau_m(float): time factors of soma
    """
    alpha = torch.sigmoid(tau_m)
    if device is not None and alpha.device != torch.device(device):
        alpha = alpha.to(device)
    mem = mem *alpha +  (1-alpha)*inputs
    return mem

## readout layer
class readout_integrator_test(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,device='cpu',bias=True,dt = 1):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
        """
        super(readout_integrator_test, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dt = dt
        self.dense = nn.Linear(input_dim,output_dim,bias=bias)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        
        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)

    
    def _infer_device(self, device=None, ref=None, input_tensor=None):
        if device is not None:
            return torch.device(device)
        if input_tensor is not None:
            return input_tensor.device
        if ref is not None:
            return ref.device
        return torch.device(self.device)

    def set_neuron_state(self,batch_size, device=None):
        dev = self._infer_device(device=device, ref=self.dense.weight)
        self.device = dev
        self.mem = torch.rand(batch_size,self.output_dim, device=dev)

    def forward(self,input_spike):
        dev = self._infer_device(input_tensor=input_spike, ref=self.dense.weight)
        #synaptic inputs
        d_input = self.dense(input_spike.float())
        # neuron model without spiking
        assert self.mem.device == dev, "state device mismatch; reset_state/set_neuron_state not called or wrong device"
        self.mem = output_Neuron_pra(d_input,self.mem,self.tau_m,self.dt,device=dev)
        return self.mem


#DH-SFNN layer
class spike_dense_test_denri_wotanh_R(nn.Module):
    def __init__(self,input_dim,output_dim,tau_minitializer = 'uniform',low_m = 0,high_m = 4,
                 tau_ninitializer = 'uniform',low_n = 0,high_n = 4,vth = 0.5,dt = 1,branch = 4,device='cpu',bias=True,test_sparsity = False,sparsity=0.5,mask_share=1):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            tau_ninitializer(str): the method of initialization of tau_n
            low_n(float): the low limit of the init values of tau_n
            high_n(float): the upper limit of the init values of tau_n
            vth(float): threshold
            branch(int): the number of dendritic branches
            test_sparsity(bool): if testing the sparsity of connection pattern 
            sparsity(float): the sparsity ratio
            mask_share(int): the number of neuron share the same connection pattern 
        """
        super(spike_dense_test_denri_wotanh_R, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt
        if test_sparsity:
            self.sparsity = sparsity 
        else:
            self.sparsity = 1/branch
        
        #group size for hardware implementation
        self.mask_share = mask_share
        self.pad = ((input_dim)//branch*branch+branch-(input_dim)) % branch
        self.dense = nn.Linear(input_dim+self.pad,output_dim*branch)
        #sparsity
        self.overlap = 1/branch
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim,branch))
        self.test_sparsity = test_sparsity

        #the number of dendritic branch
        self.branch = branch

        self.create_mask()
        
        # timing factor of membrane potential
        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)
            
            
        # timing factor of dendritic branches
        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n,low_n,high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n,low_n)

    

    def _infer_device(self, device=None, ref=None, input_tensor=None):
        if device is not None:
            return torch.device(device)
        if input_tensor is not None:
            return input_tensor.device
        if ref is not None:
            return ref.device
        return torch.device(self.device)

    #init
    def set_neuron_state(self,batch_size, device=None):
        dev = self._infer_device(device=device, ref=self.dense.weight)
        self.device = dev
        #mambrane potential
        self.mem = Variable(torch.rand(batch_size,self.output_dim, device=dev))
        self.spike = Variable(torch.rand(batch_size,self.output_dim, device=dev))
        # dendritic currents
        if self.branch == 1:
            self.d_input = Variable(torch.rand(batch_size,self.output_dim,self.branch, device=dev))
        else:
            self.d_input = Variable(torch.zeros(batch_size,self.output_dim,self.branch, device=dev))
        #threshold
        self.v_th = Variable(torch.ones(batch_size,self.output_dim, device=dev)*self.vth)

    #create connection pattern
    def create_mask(self):

        input_size = self.input_dim+self.pad
        mask = torch.zeros(self.output_dim*self.branch,input_size)
        for i in range(self.output_dim//self.mask_share):
            seq = torch.randperm(input_size)
            # j as the branch index
            for j in range(self.branch):
                if self.test_sparsity:
                    if j*input_size // self.branch+int(input_size * self.sparsity)>input_size:
                        for k in range(self.mask_share):
                            mask[(i*self.mask_share+k)*self.branch+j,seq[j*input_size // self.branch:-1]] = 1
                            mask[(i*self.mask_share+k)*self.branch+j,seq[:j*input_size // self.branch+int(input_size * self.sparsity)-input_size]] = 1
                    else:
                        for k in range(self.mask_share):
                            mask[(i*self.mask_share+k)*self.branch+j,seq[j*input_size // self.branch:j*input_size // self.branch+int(input_size * self.sparsity)]] = 1
                else:
                    for k in range(self.mask_share):
                        mask[(i*self.mask_share+k)*self.branch+j,seq[j*input_size // self.branch:(j+1)*input_size // self.branch]] = 1
        if hasattr(self, "mask"):
            self.mask.copy_(mask.to(self.mask.device))
        else:
            self.register_buffer("mask", mask)
    def apply_mask(self):
        with torch.no_grad():
            mask_dev = self.mask
            if mask_dev.device != self.dense.weight.device:
                mask_dev = mask_dev.to(self.dense.weight.device)
            self.dense.weight.mul_(mask_dev)
    def forward(self,input_spike):
        # timing factor of dendritic branches
        beta = torch.sigmoid(self.tau_n)
        dev = self._infer_device(input_tensor=input_spike, ref=self.dense.weight)
        padding = input_spike.new_zeros(input_spike.size(0),self.pad)
        k_input = torch.cat((input_spike.float(),padding),1)
        assert self.mem.device == dev, "state device mismatch; reset_state/set_neuron_state not called or wrong device"
        #update dendritic currents
        self.d_input = beta*self.d_input+(1-beta)*self.dense(k_input).reshape(-1,self.output_dim,self.branch)
        #summation of dendritic currents
        l_input = (self.d_input).sum(dim=2,keepdim=False)

        #update membrane potential and generate spikes
        self.mem,self.spike = mem_update_pra(l_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=dev)

        return self.mem,self.spike
    
    
#Vanilla SFNN layer
class spike_dense_test_origin(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,vth = 0.5,dt = 4,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(spike_dense_test_origin, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim,output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)

    def set_neuron_state(self,batch_size):

        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)

        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)
    def forward(self,input_spike):
        k_input = input_spike.float()

        d_input = self.dense(k_input)
        self.mem,self.spike = mem_update_pra(d_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)    
        return self.mem,self.spike
    
#Vanilla SFNN layer without reset 
class spike_dense_test_origin_noreset(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,vth = 0.5,dt = 4,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(spike_dense_test_origin_noreset, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim,output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)

    def set_neuron_state(self,batch_size):
        # self.mem = (torch.rand(batch_size,self.output_dim)*self.b_j0).to(self.device)
        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)

        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)
    def forward(self,input_spike):
        k_input = input_spike.float()
        d_input = self.dense(k_input)
        
        # neural model without reset
        self.mem,self.spike = mem_update_pra_noreset(d_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike

#Vanilla SFNN layer with hard reset 
class spike_dense_test_origin_hardreset(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,vth = 0.5,dt = 4,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(spike_dense_test_origin_hardreset, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim,output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)

    def set_neuron_state(self,batch_size):
        # self.mem = (torch.rand(batch_size,self.output_dim)*self.b_j0).to(self.device)
        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)

        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)
    def forward(self,input_spike):
        k_input = input_spike.float()
        d_input = self.dense(k_input)
        # neural model with hard reset
        self.mem,self.spike = mem_update_pra_hardreset(d_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike


# DH-SFNN for multitimescale_xor task
class spike_dense_test_denri_wotanh_R_xor(nn.Module):
    def __init__(self,input_dim,output_dim,tau_minitializer = 'uniform',low_m = 0,high_m = 4,
                 tau_ninitializer = 'uniform',low_n = 0,high_n = 4,low_n1 = 2,high_n1 = 6,low_n2 = -4,high_n2 = 0,vth = 0.5,dt = 4,branch = 4,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            tau_ninitializer(str): the method of initialization of tau_n
            low_n(float): the low limit of the init values of tau_n
            high_n(float): the upper limit of the init values of tau_n
            low_n1(float): the low limit of the init values of tau_n in branch 1 for the beneficial initializaiton
            high_n1(float): the upper limit of the init values of tau_n in branch 1 for the beneficial initializaiton
            low_n2(float): the low limit of the init values of tau_n in branch 2 for the beneficial initializaiton
            high_n2(float): the upper limit of the init values of tau_n in branch 2 for the beneficial initializaiton
            vth(float): threshold
            branch(int): the number of dendritic branches
        """
        super(spike_dense_test_denri_wotanh_R_xor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.is_adaptive = is_adaptive
        self.device = device
        self.vth = vth
        self.dt = dt
        mask_rate = 1/branch

        self.pad = ((input_dim)//branch*branch+branch-(input_dim)) % branch
        self.dense = nn.Linear(input_dim+self.pad,output_dim*branch,bias=bias)

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim,branch))

        self.branch = branch

        self.create_mask()
        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)

        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n,low_n,high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n,low_n)
        # init different branch with different scale
        elif tau_ninitializer  == 'seperate':
            nn.init.uniform_(self.tau_n[:,0],low_n1,high_n1)
            nn.init.uniform_(self.tau_n[:,1],low_n2,high_n2)

    def _infer_device(self, device=None, ref=None, input_tensor=None):
        if device is not None:
            return torch.device(device)
        if input_tensor is not None:
            return input_tensor.device
        if ref is not None:
            return ref.device
        return torch.device(self.device)

    def set_neuron_state(self,batch_size, device=None):

        dev = self._infer_device(device=device, ref=self.dense.weight)
        self.device = dev
        self.mem = Variable(torch.rand(batch_size,self.output_dim, device=dev))
        self.spike = Variable(torch.rand(batch_size,self.output_dim, device=dev))
        self.d_input = Variable(torch.zeros(batch_size,self.output_dim,self.branch, device=dev))

        self.v_th = Variable(torch.ones(batch_size,self.output_dim, device=dev)*self.vth)

    def create_mask(self):
        input_size = self.input_dim+self.pad
        mask = torch.zeros(self.output_dim*self.branch,input_size)
        for i in range(self.output_dim):
            for j in range(self.branch):
                mask[i*self.branch+j,j*input_size // self.branch:(j+1)*input_size // self.branch] = 1
        if hasattr(self, "mask"):
            self.mask.copy_(mask.to(self.mask.device))
        else:
            self.register_buffer("mask", mask)
    def apply_mask(self):
        with torch.no_grad():
            mask_dev = self.mask
            if mask_dev.device != self.dense.weight.device:
                mask_dev = mask_dev.to(self.dense.weight.device)
            self.dense.weight.mul_(mask_dev)
    def forward(self,input_spike):

        beta = torch.sigmoid(self.tau_n)
        dev = self._infer_device(input_tensor=input_spike, ref=self.dense.weight)
        padding = input_spike.new_zeros(input_spike.size(0),self.pad)
        k_input = torch.cat((input_spike.float(),padding),1)
        self.d_input = beta*self.d_input+(1-beta)*self.dense(k_input).reshape(-1,self.output_dim,self.branch)

        l_input = (self.d_input).sum(dim=2,keepdim=False)
        assert self.mem.device == dev, "state device mismatch; reset_state/set_neuron_state not called or wrong device"
        self.mem,self.spike = mem_update_pra(l_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=dev)

        return self.mem,self.spike
    
#Vanilla SRNN layer
class spike_rnn_test_origin(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,vth = 1,dt = 1,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(spike_rnn_test_origin, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim+output_dim,output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)


    def set_neuron_state(self,batch_size):
        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)
    def forward(self,input_spike):

        #concat the the forward inputs with the recurrent inputs 
        k_input = torch.cat((input_spike.float(),self.spike),1)
        #synaptic inputs
        d_input = self.dense(k_input)
        self.mem,self.spike = mem_update_pra(d_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike

#vanilla SRNN with the noreset LIF neuron
class spike_rnn_test_origin_noreset(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tau_minitializer = 'uniform',low_m = 0,high_m = 4,vth = 0.5,dt = 1,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            vth(float): threshold
        """
        super(spike_rnn_test_origin_noreset, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        #self.is_adaptive = is_adaptive
        self.device = device
        self.vth = vth
        self.dt = dt

        self.dense = nn.Linear(input_dim+output_dim,output_dim)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)


    def set_neuron_state(self,batch_size):
        self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)
    def forward(self,input_spike):

        #concat the the forward inputs with the recurrent inputs 
        k_input = torch.cat((input_spike.float(),self.spike),1)
        #synaptic inputs
        d_input = self.dense(k_input)
        self.mem,self.spike = mem_update_pra_noreset(d_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        
        return self.mem,self.spike





#DH-SRNN layer
class spike_rnn_test_denri_wotanh_R(nn.Module):
    def __init__(self,input_dim,output_dim,tau_minitializer = 'uniform',low_m = 0,high_m = 4,
                 tau_ninitializer = 'uniform',low_n = 0,high_n = 4,vth = 0.5,dt = 4,branch = 4,device='cpu',bias=True):
        """
        Args:
            input_dim(int): input dimension.
            output_dim(int): the number of readout neurons
            tau_minitializer(str): the method of initialization of tau_m
            low_m(float): the low limit of the init values of tau_m
            high_m(float): the upper limit of the init values of tau_m
            tau_ninitializer(str): the method of initialization of tau_n
            low_n(float): the low limit of the init values of tau_n
            high_n(float): the upper limit of the init values of tau_n
            vth(float): threshold
            branch(int): the number of dendritic branches
        """
        super(spike_rnn_test_denri_wotanh_R, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt
        mask_rate = 1/branch
        self.pad = ((input_dim+output_dim)//branch*branch+branch-(input_dim+output_dim)) % branch
        self.dense = nn.Linear(input_dim+output_dim+self.pad,output_dim*branch)

        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim,branch))
        #the number of dendritic branch
        self.branch = branch

        self.create_mask()
        
        # timing factor of membrane potential
        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m,low_m,high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m,low_m)
        # timing factor of dendritic branches       
        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n,low_n,high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n,low_n)

    #init
    def _infer_device(self, device=None, ref=None, input_tensor=None):
        if device is not None:
            return torch.device(device)
        if input_tensor is not None:
            return input_tensor.device
        if ref is not None:
            return ref.device
        return torch.device(self.device)

    def set_neuron_state(self,batch_size, device=None):

        dev = self._infer_device(device=device, ref=self.dense.weight)
        self.device = dev
        self.mem = Variable(torch.rand(batch_size,self.output_dim, device=dev))
        self.spike = Variable(torch.rand(batch_size,self.output_dim, device=dev))
        self.d_input = Variable(torch.zeros(batch_size,self.output_dim,self.branch, device=dev))

        self.v_th = Variable(torch.ones(batch_size,self.output_dim, device=dev)*self.vth)

    #create connection pattern
    def create_mask(self):
        input_size = self.input_dim+self.output_dim+self.pad
        mask = torch.zeros(self.output_dim*self.branch,input_size)
        for i in range(self.output_dim):
            seq = torch.randperm(input_size)
            for j in range(self.branch):
                mask[i*self.branch+j,seq[j*input_size // self.branch:(j+1)*input_size // self.branch]] = 1
        if hasattr(self, "mask"):
            self.mask.copy_(mask.to(self.mask.device))
        else:
            self.register_buffer("mask", mask)
    def apply_mask(self):
        with torch.no_grad():
            mask_dev = self.mask
            if mask_dev.device != self.dense.weight.device:
                mask_dev = mask_dev.to(self.dense.weight.device)
            self.dense.weight.mul_(mask_dev)
    def forward(self,input_spike):
        # timing factor of dendritic branches
        beta = torch.sigmoid(self.tau_n)
        dev = self._infer_device(input_tensor=input_spike, ref=self.dense.weight)
        padding = input_spike.new_zeros(input_spike.size(0),self.pad)
        k_input = torch.cat((input_spike.float(),self.spike,padding),1)
        assert self.mem.device == dev and self.spike.device == dev, "state device mismatch; reset_state/set_neuron_state not called or wrong device"
        #update dendritic currents
        self.d_input = beta*self.d_input+(1-beta)*self.dense(k_input).reshape(-1,self.output_dim,self.branch)
        #summation of dendritic currents
        l_input = (self.d_input).sum(dim=2,keepdim=False)

        #update membrane potential and generate spikes
        self.mem,self.spike = mem_update_pra(l_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=dev)

        return self.mem,self.spike
