from abc import abstractmethod
from typing import Callable
import torch
from SeqSNN.network.snn import TSLIF_base
from torch import nn


class BaseNode(TSLIF_base.MemoryModule):
    def __init__(self,
                 v_threshold: float = 1.,
                 v_reset: float = 0.,
                 surrogate_function: Callable = None,
                 detach_reset: bool = False,
                 step_mode='s', backend='torch',
                 store_v_seq: bool = True):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('v_s', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold

        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq


        self.alpha_s = torch.nn.Parameter(torch.randn([1, 128], dtype=torch.float))
        self.alpha_l = torch.nn.Parameter(torch.randn([1, 128], dtype=torch.float))

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset

        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v


    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold, 2.0)

    def sl_neuronal_fire(self):
        s_s = self.surrogate_function(self.v - self.v_threshold, 2.0)
        s_l = self.surrogate_function(self.v_s - self.v_threshold,  2.0)
        return s_s, s_l

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        # spike = self.neuronal_fire()
        s_s, s_l = self.sl_neuronal_fire()
        spike = self.alpha_s * s_s + self.alpha_l * s_l
        # self.neuronal_reset(spike)
        self.neuronal_reset(s_s, s_l)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):

        #### time series ###
        T = x_seq.shape[-1]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[:, t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        # if self.store_v_seq:
        #     self.v_seq = torch.stack(v_seq)
        outputs = torch.stack(y_seq, dim=0).permute(1, 0)

        return outputs

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class TSLIFNode(BaseNode):
    def __init__(self,
                 v_threshold=1.0,
                 v_reset=0.,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 hard_reset=False,
                 step_mode='s',
                 k=2,
                 decay_factor: torch.Tensor = torch.tensor([0.8, 0.2, 0.3, 0.7], dtype=torch.float),
                 gamma: float = 0.5):
        super(TSLIFNode, self).__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.k = k
        for i in range(1, self.k + 1):
            self.register_memory('v' + str(i), 0.)

        self.names = self._memories
        self.hard_reset = hard_reset
        self.gamma = gamma
        self.decay_factor = torch.nn.Parameter(decay_factor)
        self.kk = torch.nn.Parameter(torch.tensor([0.8], dtype=torch.float))
        self.yy = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float))

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        # self.names['v1'] = self.names['v1'] - torch.sigmoid(self.decay_factor[0][0]) * self.names['v2'] + x
        # self.names['v2'] = self.names['v2'] + torch.sigmoid(self.decay_factor[0][1]) * self.names['v1']


        self.names['v1'] = self.decay_factor[0] * self.names['v1'] + self.decay_factor[1] * x - self.yy * self.names['v2']
        self.names['v2'] = self.decay_factor[2] * self.names['v2'] + self.decay_factor[3] * x - self.kk * self.names['v1']

        # self.names['v1'] =  self.names['v1'] + (1 - torch.sigmoid(self.decay_factor[0])) * x
        # self.names['v2'] =  self.names['v2'] + (1 - torch.sigmoid(self.decay_factor[1])) * x - self.names['v1']

        self.v = self.names['v2']
        self.v_s = self.names['v1']

    def neuronal_reset(self, spike_s, spike_l):


        if not self.hard_reset:
            # soft reset
            # self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.gamma)
            self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_l , self.gamma)
            self.names['v2'] = self.jit_soft_reset(self.names['v2'], spike_s, self.v_threshold)
        else:
            # hard reset
            for i in range(2, self.k + 1):
                self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d,  self.v_reset)

    def forward(self, x: torch.Tensor):
        # self.v = 0.
        # self.v1 = 0.
        # self.v2 = 0.
        return super().single_step_forward(x)
    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
               f"hard_reset={self.hard_reset}, " \
               f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"








class BaseNode1(TSLIF_base.MemoryModule):
    def __init__(self,
                 v_threshold: float = 1.,
                 v_reset: float = 0.,
                 surrogate_function: Callable = None,
                 detach_reset: bool = False,
                 step_mode='s', backend='torch',
                 store_v_seq: bool = True):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
            self.register_memory('v_s', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold

        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq
        # dimension should change to fit the dataset.
        self.alpha_s = torch.nn.Parameter(torch.randn([1, 168], dtype=torch.float))
        self.alpha_l = torch.nn.Parameter(torch.randn([1, 168], dtype=torch.float))


        # self.v_threshold_s = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float))
        # self.v_threshold_l = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float))
    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset

        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v


    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        # print('22', self.surrogate_function)
        return self.surrogate_function(self.v - self.v_threshold, 2.0)

    def sl_neuronal_fire(self):
        s_s = self.surrogate_function(self.v - self.v_threshold, 2.0)
        s_l = self.surrogate_function(self.v_s - self.v_threshold,  2.0)
        return s_s, s_l

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        # spike = self.neuronal_fire()
        s_s, s_l = self.sl_neuronal_fire()
        # test = s_s.sum() - s_l.sum()
        spike = self.alpha_s * s_s + self.alpha_l * s_l
       # self.neuronal_reset(spike)
        self.neuronal_reset(s_s, s_l)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):

        #### time series ###
        T = x_seq.shape[-1]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[:, t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
            # print('v_seq', self.v_seq.shape)

        #
        # if self.store_v_seq:
        #     self.v_seq = torch.stack(v_seq)
        outputs = torch.stack(y_seq, dim=0).permute(1, 0)
        # print(outputs.shape)
        # dsa
        return outputs
        # return torch.stack(y_seq, dim=0)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class TSLIFNode2(BaseNode1):
    def __init__(self,
                 v_threshold=1.0,
                 v_reset=0.,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 hard_reset=False,
                 step_mode='s',
                 k=2,
                 decay_factor: torch.Tensor = torch.tensor([0.8, 0.2, 0.3, 0.7], dtype=torch.float),
                 gamma: float = 0.5):
        super(TSLIFNode2, self).__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.k = k
        for i in range(1, self.k + 1):
            self.register_memory('v' + str(i), 0.)

        self.names = self._memories
        self.hard_reset = hard_reset
        self.gamma = gamma
        self.decay_factor = torch.nn.Parameter(decay_factor)
        self.kk = torch.nn.Parameter(torch.tensor([0.8], dtype=torch.float))
        self.yy = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float))

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        # self.names['v1'] = self.names['v1'] - torch.sigmoid(self.decay_factor[0][0]) * self.names['v2'] + x
        # self.names['v2'] = self.names['v2'] + torch.sigmoid(self.decay_factor[0][1]) * self.names['v1']


        self.names['v1'] = self.decay_factor[0] * self.names['v1'] + self.decay_factor[1] * x - self.yy * self.names['v2']
        self.names['v2'] = self.decay_factor[2] * self.names['v2'] + self.decay_factor[3] * x - self.kk * self.names['v1']

        # self.names['v1'] =  self.names['v1'] + (1 - torch.sigmoid(self.decay_factor[0])) * x
        # self.names['v2'] =  self.names['v2'] + (1 - torch.sigmoid(self.decay_factor[1])) * x - self.names['v1']

        self.v = self.names['v2']
        self.v_s = self.names['v1']

    def neuronal_reset(self, spike_s, spike_l):


        if not self.hard_reset:
            # soft reset
            # self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.gamma)
            self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_l , self.gamma)
            self.names['v2'] = self.jit_soft_reset(self.names['v2'], spike_s, self.v_threshold)
        else:
            # hard reset
            for i in range(2, self.k + 1):
                self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d,  self.v_reset)

    def forward(self, x: torch.Tensor):
        # self.v = 0.
        # self.v1 = 0.
        # self.v2 = 0.
        # test= super().single_step_forward(x)
        # s = test.sum()
        return super().single_step_forward(x)
    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
               f"hard_reset={self.hard_reset}, " \
               f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"


