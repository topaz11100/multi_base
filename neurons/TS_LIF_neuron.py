import torch
import torch.nn as nn
import copy
from abc import abstractmethod
from typing import Callable


class MemoryModule(nn.Module):
    def __init__(self):
        """
        * :ref:`API in English <MemoryModule.__init__-en>`

        .. _MemoryModule.__init__-cn:

        ``MemoryModule`` 是SpikingJelly中所有有状态（记忆）模块的基类。

        * :ref:`中文API <MemoryModule.__init__-cn>`

        .. _MemoryModule.__init__-en:

        ``MemoryModule`` is the base class of all stateful modules in SpikingJelly.

        """
        super().__init__()
        self._memories = {}
        self._memories_rv = {}

    def register_memory(self, name: str, value):
        """
        * :ref:`API in English <MemoryModule.register_memory-en>`

        .. _MemoryModule.register_memory-cn:

        :param name: 变量的名字
        :type name: str
        :param value: 变量的值
        :type value: any

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量的重置值会被设置为 ``value``。每次调用 ``self.reset()``
        函数后， ``self.name`` 都会被重置为 ``value``。

        * :ref:`中文API <MemoryModule.register_memory-cn>`

        .. _MemoryModule.register_memory-en:

        :param name: variable's name
        :type name: str
        :param value: variable's value
        :type value: any

        Register the variable to memory dict, which saves stateful variables (e.g., the membrane potential of a
        spiking neuron). The reset value of this variable will be ``value``. ``self.name`` will be set to ``value`` after
        each calling of ``self.reset()``.

        """
        assert not hasattr(self, name), f'{name} has been set as a member variable!'
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        """
        * :ref:`API in English <MemoryModule.reset-en>`

        .. _MemoryModule.reset-cn:

        重置所有有状态变量为默认值。

        * :ref:`中文API <MemoryModule.reset-cn>`

        .. _MemoryModule.reset-en:

        Reset all stateful variables to their default values.
        """
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if '_memories' in self.__dict__:
            memories = self.__dict__['_memories']
            if name in memories:
                return memories[name]

        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        _memories = self.__dict__.get('_memories')
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        memories = list(self._memories.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + memories

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def memories(self):
        """
        * :ref:`API in English <MemoryModule.memories-en>`

        .. _MemoryModule.memories-cn:

        :return: 返回一个所有状态变量的迭代器
        :rtype: Iterator

        * :ref:`中文API <MemoryModule.memories-cn>`

        .. _MemoryModule.memories-en:

        :return: an iterator over all stateful variables
        :rtype: Iterator
        """
        for name, value in self._memories.items():
            yield value

    def named_memories(self):
        """
        * :ref:`API in English <MemoryModule.named_memories-en>`

        .. _MemoryModule.named_memories-cn:

        :return: 返回一个所有状态变量及其名称的迭代器
        :rtype: Iterator

        * :ref:`中文API <MemoryModule.named_memories-cn>`

        .. _MemoryModule.named_memories-en:

        :return: an iterator over all stateful variables and their names
        :rtype: Iterator
        """

        for name, value in self._memories.items():
            yield name, value

    def detach(self):
        """
        * :ref:`API in English <MemoryModule.detach-en>`

        .. _MemoryModule.detach-cn:

        从计算图中分离所有有状态变量。

        .. tip::

            可以使用这个函数实现TBPTT(Truncated Back Propagation Through Time)。


        * :ref:`中文API <MemoryModule.detach-cn>`

        .. _MemoryModule.detach-en:

        Detach all stateful variables.

        .. admonition:: Tip
            :class: tip

            We can use this function to implement TBPTT(Truncated Back Propagation Through Time).

        """

        for key in self._memories.keys():
            if isinstance(self._memories[key], torch.Tensor):
                self._memories[key].detach_()

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)
        # do not apply on default values
        # for key, value in self._memories_rv.items():
        #     if isinstance(value, torch.Tensor):
        #         self._memories_rv[key] = fn(value)
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica


class StepModule:
    def supported_step_mode(self):
        """
        * :ref:`API in English <StepModule.supported_step_mode-en>`
        .. _StepModule.supported_step_mode-cn:
        :return: 包含支持的后端的tuple
        :rtype: tuple[str]
        返回此模块支持的步进模式。
        * :ref:`中文 API <StepModule.supported_step_mode-cn>`
        .. _StepModule.supported_step_mode-en:
        :return: a tuple that contains the supported backends
        :rtype: tuple[str]
        """
        return ('s', 'm')

    @property
    def step_mode(self):
        """
        * :ref:`API in English <StepModule.step_mode-en>`
        .. _StepModule.step_mode-cn:
        :return: 模块当前使用的步进模式
        :rtype: str
        * :ref:`中文 API <StepModule.step_mode-cn>`
        .. _StepModule.step_mode-en:
        :return: the current step mode of this module
        :rtype: str
        """
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        """
        * :ref:`API in English <StepModule.step_mode-setter-en>`
        .. _StepModule.step_mode-setter-cn:
        :param value: 步进模式
        :type value: str
        将本模块的步进模式设置为 ``value``
        * :ref:`中文 API <StepModule.step_mode-setter-cn>`
        .. _StepModule.step_mode-setter-en:
        :param value: the step mode
        :type value: str
        Set the step mode of this module to be ``value``
        """
        if value not in self.supported_step_mode():
            raise ValueError(f'step_mode can only be {self.supported_step_mode()}, but got "{value}"!')
        self._step_mode = value


class BaseNode(MemoryModule):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_function: Callable = None,
        detach_reset: bool = False,
        step_mode='s',
        backend='torch',
        store_v_seq: bool = True,
        hidden_dim: int | None = None,
        neuron_shape: tuple[int, ...] | None = None,
    ):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        resolved_hidden = hidden_dim
        if resolved_hidden is None and neuron_shape is not None:
            if len(neuron_shape) != 1:
                raise ValueError(f"neuron_shape must be 1D for TS-LIF, got {neuron_shape}.")
            resolved_hidden = int(neuron_shape[0])
        if resolved_hidden is None:
            raise ValueError("hidden_dim or neuron_shape must be provided for TS-LIF neurons")
        if resolved_hidden <= 0:
            raise ValueError(f"hidden_dim must be positive, got {resolved_hidden}")

        self.hidden_dim = resolved_hidden
        self.neuron_shape = (resolved_hidden,)

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

        self.alpha_s = torch.nn.Parameter(torch.randn([1, self.hidden_dim], dtype=torch.float))
        self.alpha_l = torch.nn.Parameter(torch.randn([1, self.hidden_dim], dtype=torch.float))

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
    def __init__(
        self,
        v_threshold=1.0,
        v_reset=0.0,
        surrogate_function: Callable = None,
        detach_reset=False,
        hard_reset=False,
        step_mode='s',
        k=2,
        decay_factor: torch.Tensor = torch.tensor([0.8, 0.2, 0.3, 0.7], dtype=torch.float),
        gamma: float = 0.5,
        hidden_dim: int | None = None,
        neuron_shape: tuple[int, ...] | None = None,
    ):
        super(TSLIFNode, self).__init__(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            hidden_dim=hidden_dim,
            neuron_shape=neuron_shape,
        )
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
                self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_s,  self.v_reset)

    def forward(self, x: torch.Tensor):
        # self.v = 0.
        # self.v1 = 0.
        # self.v2 = 0.
        return super().single_step_forward(x)
    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
               f"hard_reset={self.hard_reset}, " \
               f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"








class BaseNode1(MemoryModule):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_function: Callable = None,
        detach_reset: bool = False,
        step_mode='s',
        backend='torch',
        store_v_seq: bool = True,
        hidden_dim: int | None = None,
        neuron_shape: tuple[int, ...] | None = None,
    ):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        resolved_hidden = hidden_dim
        if resolved_hidden is None and neuron_shape is not None:
            if len(neuron_shape) != 1:
                raise ValueError(f"neuron_shape must be 1D for TS-LIF, got {neuron_shape}.")
            resolved_hidden = int(neuron_shape[0])
        if resolved_hidden is None:
            raise ValueError("hidden_dim or neuron_shape must be provided for TS-LIF neurons")
        if resolved_hidden <= 0:
            raise ValueError(f"hidden_dim must be positive, got {resolved_hidden}")

        self.hidden_dim = resolved_hidden
        self.neuron_shape = (resolved_hidden,)

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
        self.alpha_s = torch.nn.Parameter(torch.randn([1, self.hidden_dim], dtype=torch.float))
        self.alpha_l = torch.nn.Parameter(torch.randn([1, self.hidden_dim], dtype=torch.float))


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
    def __init__(
        self,
        v_threshold=1.0,
        v_reset=0.0,
        surrogate_function: Callable = None,
        detach_reset=False,
        hard_reset=False,
        step_mode='s',
        k=2,
        decay_factor: torch.Tensor = torch.tensor([0.8, 0.2, 0.3, 0.7], dtype=torch.float),
        gamma: float = 0.5,
        hidden_dim: int | None = None,
        neuron_shape: tuple[int, ...] | None = None,
    ):
        super(TSLIFNode2, self).__init__(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            hidden_dim=hidden_dim,
            neuron_shape=neuron_shape,
        )
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
                self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_s,  self.v_reset)

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
