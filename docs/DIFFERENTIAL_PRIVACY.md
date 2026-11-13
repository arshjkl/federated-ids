# Differential Privacy (DP-SGD / DP-FedAvg)

## Entry points

- `src/privacy/dp_sgd.py` — DPSGDTrainer (Opacus 1.6.0 + manual fallback), privacy accounting.
- `src/federated/run_dp_federated.py` — DP-FedAvg end-to-end training.

## DP-SGD mechanism

Per-example gradient clipping + Gaussian noise, with moments-accountant privacy accounting:
- Per-example gradient clipping to `max_grad_norm`
- Gaussian noise scaled by `noise_multiplier × max_grad_norm`
- Privacy spent tracked as (epsilon, delta) per client and per run

Opacus is preferred; a manual NumPy/PyTorch fallback is used when Opacus is unavailable.

## BatchNorm incompatibility

Per-example DP-SGD (Opacus) rejects BatchNorm layers. Therefore:
- Client models use `use_batchnorm=False` when `noise_multiplier > 0`
- The global model matches: `use_batchnorm=(noise_multiplier == 0.0)`
- Linear weights are transferred from the centralized (BN) checkpoint into the no-BN global model via `_transfer_linear_weights`, which extracts Linear weight/bias tensors in order and skips BatchNorm params/buffers.

## DP-FedAvg configuration

| Parameter | Value |
|---|---|
| Rounds | 30 |
| Clients | 5 |
| Partitioning | non-IID Dirichlet, α=0.5 |
| Local epochs | 3 |
| Batch size | 128 |
| Learning rate | 0.001 |
| max_grad_norm | 1.0 |
| noise_multiplier | 1.0 |
| delta | 1e-5 |
| Warm start | centralized checkpoint (Linear weights only) |

## Privacy reporting

`privacy_report()` reports epsilon for target epsilons [1.0, 2.0, 4.0, 8.0], plus per-round max/mean epsilon, naive composition upper bound, delta, noise_multiplier, max_grad_norm, and dp_enabled flag.

Outputs:
- `models/federated_mlp_dp.pth` — DP model checkpoint + privacy report + config
- `models/dp_federated_results.json` — final metrics, privacy report, history, config

## Tests

10 tests in `tests/test_dp.py` (9 passing, 1 skipped when Opacus unavailable):
`test_clip_gradients_scales_down_large_norm`, `test_clip_gradients_no_op_when_small_norm`,
`test_add_gaussian_noise_changes_gradients`, `test_compute_epsilon_positive_finite`,
`test_compute_epsilon_infinite_when_no_noise`, `test_compute_epsilon_decreases_with_larger_noise`,
`test_privacy_report_structure`, `test_train_epoch_manual_fallback`,
`test_get_privacy_spent_returns_epsilon_delta`, `test_dp_enabled_flag`,
`test_train_epoch_opacus_path` (skipped).
