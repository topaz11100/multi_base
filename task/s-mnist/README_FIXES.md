# S-MNIST Fixes

- **Neuron fixes**
  - `CP_LIF_neuron.py` now uses a valid `tuple[int, ...]` type hint for `neuron_shape` so imports no longer fail.
  - `TS_LIF_neuron.py` initializes per-channel parameters using the provided hidden dimension (via `neuron_shape`), and hard reset no longer references an undefined `spike_d` variable.
- **State reset reliability**
  - `reset_states` accepts `batch_size` and `device`, retries common `reset_state` signatures, and raises warnings instead of silently skipping failures. Direct `module.names` mutation was removed to avoid side effects.
- **CLI/Model safety**
  - SRNN builds now validate that at least two hidden dimensions are provided (e.g., `--hidden 256 256`).
  - CUDA seeding is guarded by availability.
