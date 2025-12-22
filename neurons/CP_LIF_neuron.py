import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) jitcompile: torch.compile decorator (fallback 포함)
# -----------------------------
def jitcompile(fn=None, **compile_kwargs):
    """
    @jitcompile 데코레이터로 사용 가능.
    PyTorch 2.x torch.compile이 있으면 컴파일 적용.
    환경/설정상 실패하면 원본 함수를 그대로 반환(fallback).
    """
    if fn is None:
        return lambda f: jitcompile(f, **compile_kwargs)

    if hasattr(torch, "compile"):
        try:
            return torch.compile(fn, **compile_kwargs)
        except Exception:
            return fn
    return fn


# -----------------------------
# 2) Surrogate spike (torch-only, compile-friendly)
#    forward: hard step
#    backward: sigmoid gradient (STE style)
# -----------------------------
@jitcompile
def surrogate_spike(x: torch.Tensor, beta: float) -> torch.Tensor:
    """
    x: membrane - threshold
    beta: surrogate slope (크면 더 날카롭고, 작으면 더 부드러움)

    forward는 hard step, backward는 sigmoid derivative가 흐르도록:
      s = H(x) + (sigmoid(beta x) - sigmoid(beta x).detach())
    """
    hard = (x > 0).to(dtype=x.dtype)
    soft = torch.sigmoid(beta * x)
    return hard + (soft - soft.detach())


# -----------------------------
# 3) 핵심 LIF step (pure function)
# -----------------------------
@jitcompile
def lif_step(
    x: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    v_th: float,
    r: torch.Tensor,
    detach_reset: bool,
    surrogate_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x: input current (B, *neuron_shape)
    v: membrane (B, *neuron_shape)
    alpha: exp(-dt/tau) broadcastable to v (typically (1, *neuron_shape))
    v_th: threshold (scalar)
    r: soft reset amount broadcastable to v (typically (1, *neuron_shape))
    detach_reset: reset에서 spike gradient 끊을지 여부
    """
    # stable discretization: v <- alpha*v + (1-alpha)*x
    v = alpha * v + (1.0 - alpha) * x

    s = surrogate_spike(v - v_th, surrogate_beta)

    s_reset = s.detach() if detach_reset else s
    v = v - s_reset * r
    return s, v


# -----------------------------
# 4) Learnable per-neuron LIF Module (stateful)
# -----------------------------
class CP_LIF(nn.Module):
    """
    - 입력 x의 shape: (B, *neuron_shape) 또는 (T, B, *neuron_shape)
    - 뉴런별 학습 파라미터:
        tau_raw (-> tau = softplus(tau_raw) + tau_min)  shape = neuron_shape
        r_raw   (-> r   = softplus(r_raw)   + r_min)    shape = neuron_shape
    - state:
        self.v  shape = (B, *neuron_shape)
    """

    def __init__(
        self,
        neuron_shape: tuple[int, ...],   # 예: (N,) 또는 (C,H,W) 등
        dt: float = 1.0,
        v_threshold: float = 1.0,
        tau_init: float = 2.0,
        soft_reset_init: float = 1.0,
        tau_min: float = 1e-3,
        soft_reset_min: float = 1e-6,
        detach_reset: bool = False,
        surrogate_beta: float = 10.0,
        compile_backend: str | None = None,   # 필요하면 "inductor" 등 지정
        compile_mode: str | None = None,      # 필요하면 "default" / "max-autotune" 등
    ):
        super().__init__()
        if len(neuron_shape) == 0:
            raise ValueError("neuron_shape must be non-empty, e.g. (N,) or (C,H,W).")

        self.neuron_shape = tuple(int(d) for d in neuron_shape)
        self.dt = float(dt)
        self.v_threshold = float(v_threshold)
        self.tau_min = float(tau_min)
        self.soft_reset_min = float(soft_reset_min)
        self.detach_reset = bool(detach_reset)
        self.surrogate_beta = float(surrogate_beta)

        # learnable raw parameters (per-neuron)
        # tau = softplus(tau_raw) + tau_min
        # r   = softplus(r_raw)   + r_min
        tau0 = max(float(tau_init) - self.tau_min, 1e-6)
        r0 = max(float(soft_reset_init) - self.soft_reset_min, 1e-6)

        # inverse-softplus 초기화: raw = log(exp(y)-1)
        tau_raw0 = torch.log(torch.expm1(torch.tensor(tau0, dtype=torch.float32)))
        r_raw0 = torch.log(torch.expm1(torch.tensor(r0, dtype=torch.float32)))

        self.tau_raw = nn.Parameter(torch.full(self.neuron_shape, tau_raw0.item(), dtype=torch.float32))
        self.r_raw = nn.Parameter(torch.full(self.neuron_shape, r_raw0.item(), dtype=torch.float32))

        # membrane state (런타임에 입력 배치에 맞춰 생성)
        self.v = None  # type: ignore[assignment]

        # (선택) 함수 수준 compile은 이미 @jitcompile로 적용됨.
        # 추가로 모듈 전체를 컴파일하고 싶으면, 사용자가 바깥에서 torch.compile(module) 하면 됨.
        self._compile_backend = compile_backend
        self._compile_mode = compile_mode

    def tau(self) -> torch.Tensor:
        return F.softplus(self.tau_raw) + self.tau_min

    def soft_reset_amount(self) -> torch.Tensor:
        return F.softplus(self.r_raw) + self.soft_reset_min

    def alpha(self) -> torch.Tensor:
        # alpha = exp(-dt/tau) in (0,1)
        return torch.exp(-self.dt / self.tau())

    @torch.no_grad()
    def reset_state(self):
        """membrane state 초기화"""
        self.v = None

    def _check_input_shape(self, x: torch.Tensor):
        # x: (B, *neuron_shape) or (T, B, *neuron_shape)
        if x.dim() == 1 + len(self.neuron_shape):
            if tuple(x.shape[1:]) != self.neuron_shape:
                raise ValueError(f"Expected x.shape[1:] == {self.neuron_shape}, got {tuple(x.shape[1:])}.")
        elif x.dim() == 2 + len(self.neuron_shape):
            if tuple(x.shape[2:]) != self.neuron_shape:
                raise ValueError(f"Expected x.shape[2:] == {self.neuron_shape}, got {tuple(x.shape[2:])}.")
        else:
            raise ValueError(
                f"x.dim() must be {1+len(self.neuron_shape)} (single-step) "
                f"or {2+len(self.neuron_shape)} (multi-step), got {x.dim()}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
          - single-step: (B, *neuron_shape) -> spike: (B, *neuron_shape)
          - multi-step : (T, B, *neuron_shape) -> spike: (T, B, *neuron_shape)
        """
        self._check_input_shape(x)

        # parameters (broadcast 준비: (1, *neuron_shape))
        alpha = self.alpha().to(dtype=x.dtype, device=x.device).unsqueeze(0)
        r = self.soft_reset_amount().to(dtype=x.dtype, device=x.device).unsqueeze(0)

        if x.dim() == 1 + len(self.neuron_shape):
            # single-step
            if (self.v is None) or (self.v.shape != x.shape) or (self.v.device != x.device) or (self.v.dtype != x.dtype):
                self.v = torch.zeros_like(x)

            s, v_new = lif_step(
                x=x,
                v=self.v,
                alpha=alpha,
                v_th=self.v_threshold,
                r=r,
                detach_reset=self.detach_reset,
                surrogate_beta=self.surrogate_beta,
            )
            self.v = v_new
            return s

        else:
            # multi-step: (T,B,...) loop
            T = x.shape[0]
            B = x.shape[1]
            target_shape = (B, *self.neuron_shape)

            if (self.v is None) or (self.v.shape != target_shape) or (self.v.device != x.device) or (self.v.dtype != x.dtype):
                self.v = torch.zeros(target_shape, device=x.device, dtype=x.dtype)

            spikes = []
            v = self.v
            for t in range(T):
                s, v = lif_step(
                    x=x[t],
                    v=v,
                    alpha=alpha,
                    v_th=self.v_threshold,
                    r=r,
                    detach_reset=self.detach_reset,
                    surrogate_beta=self.surrogate_beta,
                )
                spikes.append(s)
            self.v = v
            return torch.stack(spikes, dim=0)
