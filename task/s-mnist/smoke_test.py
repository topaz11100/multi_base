import torch

from neurons.CP_LIF_neuron import CP_LIF
from neurons.TS_LIF_neuron import TSLIFNode


def main():
    cp = CP_LIF((4,), v_threshold=1.0)
    x = torch.randn(2, 4)
    cp(x)
    cp.reset_state()

    surrogate_fn = lambda input, beta: (input > 0).float()  # simple deterministic surrogate
    ts = TSLIFNode(neuron_shape=(4,), surrogate_function=surrogate_fn, hard_reset=True)
    ts(torch.randn(2, 4))
    ts.reset()
    ts.reset_state()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
