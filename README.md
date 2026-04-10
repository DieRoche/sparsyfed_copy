# SparsyFed Implementation Audit

## 1. Scope

This document audits the **actual executable implementation path** for `METHOD_NAME = SparsyFed` in this repository.

Primary path audited:
- Entry point: `project/main.py` (Hydra default config: `config_name="cifar_resnet18"`).
- Task config selected by default: `project/conf/task/cifar_resnet18.yaml`.
- Strategy selected by default: `project/conf/strategy/fedavgNZ.yaml`.

Interpretation rule used throughout this audit:
- **Used in active pipeline** = executed when running default `project.main` with the current default configs.
- **Exists in repository** = implemented somewhere but not necessarily reached by the active path.

---

## 2. High-Level Verdict

`SparsyFed` is implemented here primarily as a **model-layer behavior modification during local training** (custom linear/conv modules with sparse activation/input handling in autograd), **not as an explicit client payload compression protocol**.

Key findings:
- Activation of SparsyFed depends mainly on `task.model_and_data` selecting SparsyFed model generators (for CIFAR default: `CIFAR_SPARSYFED_RN18`).
- Client uploads are still Flower `Parameters` (dense ndarray list serialized by Flower), not an explicit sparse/quantized payload object.
- Upload/download traffic logging is estimated from serialized parameter byte size, and upload uses `len(fit_results) + len(failures)` as multiplier.
- FLOPs logging exists (`round_flops`, `total_flops`) but covers only local-train estimate; compression/decompression FLOPs are logged as zero unless some client explicitly overrides.

---

## 3. Actual Execution Pipeline

1. `project/main.py` loads Hydra config `cifar_resnet18` by default.
2. `dispatch_data(cfg)` resolves to CIFAR dispatch and selects model/data generator from `task.model_and_data`.
3. With default task config (`CIFAR_SPARSYFED_RN18`), model generator is `get_network_generator_resnet_sparsyfed(...)`.
4. This replaces eligible `nn.Conv2d`/`nn.Linear` layers with SparsyFed modules before training.
5. Flower simulation runs with `WandbServer` and strategy `FedAvgNZ`.
6. Each round:
   - server sends current global parameters to sampled clients,
   - clients train locally,
   - clients return updated full parameters,
   - server aggregates via `FedAvgNZ.aggregate_fit` (weighted average variant),
   - server evaluates centrally and optionally distributed.
7. WandB logging is performed through `WandbHistory` and server-side metric augmentation in `WandbServer`.

---

## 4. Method Activation and Required Flags

### Activation conditions (active path)

| Item | Where | Required value for active SparsyFed path | Notes |
|---|---|---|---|
| Config entry point | `project/main.py` | Hydra default `cifar_resnet18` | This is the default executable path. |
| Model selection | `project/conf/task/cifar_resnet18.yaml` | `task.model_and_data: CIFAR_SPARSYFED_RN18` | This is the key SparsyFed switch for default path. |
| Train function selection | `project/conf/task/cifar_resnet18.yaml` | `task.train_structure: CIFAR_RN18_PRUNE` | Uses prune-style training function wrapper. |
| SparsyFed params | `project/conf/task/cifar_resnet18.yaml` | `alpha`, `sparsity` | Passed to SparsyFed layer generators. |

### Method-specific vs generic vs unused

- **Method-specific and used**:
  - `project/task/utils/sparsyfed_modules.py` (`SparsyFedLinear`, `SparsyFedConv2D`, `sparsyfed_linear`, `sparsyfed_conv2d`).
  - CIFAR model replacement functions in `project/task/cifar_resnet18/models.py`.
- **Generic FL and used**:
  - `project/client/client.py` (NumPyClient fit/evaluate lifecycle).
  - `project/fed/server/wandb_server.py` (round loop, traffic/FLOPs accumulation).
  - `project/fed/server/strategy/fedavgNZ.py` (server aggregation).
- **Exists but not used in active default path**:
  - Speech and ViT SparsyFed dispatch paths.
  - no-act SparsyFed variants (`*_SPARSYFED_NA_*`) unless selected via config.
  - dataset preparation routines are not automatically executed in `main.py` (`download_and_preprocess` call is commented).

---

## 5. Client-Side Processing After Local Training

### Object immediately after local training

In `Client.fit`, after calling task-specific train function, the client computes:
- `updated_parameters = generic_get_parameters(self.net)`

This is a **full list of NumPy arrays from `state_dict()`**, sorted by key.

### Transformations before server uses it

| Processing type | Status in active path | Evidence/behavior |
|---|---|---|
| Delta computation (`new - old`) | **Missing** | Client directly returns full updated parameters. |
| Gradient/update clipping for payload | **Missing** | No client-side post-training clipping stage before return. |
| Payload normalization | **Missing** | No normalization pass on outgoing payload. |
| Explicit sparsification of payload tensors | **Missing (for transmission)** | SparsyFed sparsity is inside local forward/backward behavior, not payload rewrite. |
| Mask packaging | **Missing (for transmission)** | No `(values, indices, mask)` payload emitted. |
| Quantization | **Missing** | No quant/dequant in client upload path. |
| Low-rank decomposition | **Missing** | No SVD/low-rank object construction in upload path. |
| Serialization/compression before handoff | **Missing for FL transport** | Optional `np.savez_compressed` is only for local artifact dump when `client_updates_dir` is provided in config.extra; not used for Flower upload. |

---

## 6. Client-to-Server Payload and Transmission Logic

- **Actual uploaded payload object**: `updated_parameters` (Python list of NumPy ndarrays) from `generic_get_parameters(self.net)`.
- **Handoff point**: `Client.fit` return tuple `(updated_parameters, num_samples, metrics)`.
- **Framework wrapping**: Flower converts ndarray list to `Parameters` for transport.
- **Representation type**: full model weights, not sparse tuple/delta/custom compressed object.
- **Transport nature in this repository**: in Flower simulation, logical client/server communication is framework-managed; user code constructs payload in-memory and returns via API.
- **Explicit user-level serialization before handoff**: none in active path.

---

## 7. Upload Traffic Validation

`WandbServer.fit` computes upload traffic as:

- `on_wire_upload = parameters_size_bytes(fit_results[0][1].parameters)` (or fallback from previous round/global params).
- `upload_traffic = active_clients * on_wire_upload`.
- `active_clients = len(fit_results) + len(failures)`.

Validation:

| Check | Verdict | Detail |
|---|---|---|
| `upload_traffic_per_client` corresponds to payload size | **PARTIAL** | Uses size from first successful result (or fallback), assumes homogeneous size across clients. |
| `upload_traffic = upload_traffic_per_client * number_of_active_users` | **PASS (formula)** | Implemented exactly with `active_clients * on_wire_upload`. |
| Multiplier uses active users vs total users | **PARTIAL** | Uses `fit_results + failures`, not configured total clients; however failures are counted as upload participants even if no payload arrived. |
| Derived from actual transmitted payload | **PARTIAL** | Size derived from Flower `Parameters` tensor bytes; still estimated at aggregate level, not summed per-client payload objects. |

---

## 8. Server-Side Reconstruction / Decoding

In active path there is **no explicit decode/reconstruct stage** such as:
- sparse reconstruction,
- dequantization,
- low-rank reconstruction,
- custom deserialization.

Server strategy consumes Flower-decoded ndarrays via `parameters_to_ndarrays(fit_res.parameters)` in `FedAvgNZ.aggregate_fit` and aggregates directly.

So reconstruction is only the standard Flower parameter conversion, not method-specific compression decoding.

---

## 9. Global Aggregation / Global Update Logic

Server update behavior (active default strategy `FedAvgNZ`):
- Collect successful client `FitRes`.
- Convert each client `Parameters` -> ndarrays.
- Aggregate with custom `aggregate(results)` (weighted by client sample count, nonzero-mask-influenced expression).
- Convert aggregated ndarrays back to `Parameters` and set as new global model.

Important clarifications:
- Server does **not** perform optimizer-based gradient descent step.
- Global update is aggregation-driven parameter replacement.
- The aggregation uses full client parameter tensors, not client deltas.

---

## 10. Server-to-Client Payload and Download Logic

Server-to-client payload is `self.parameters` from Flower server state (global model parameters).

Before sending to clients in the next round, there is no explicit:
- compression,
- quantization,
- sparsification transform,
- custom serialization in user code.

Download traffic accounting uses `parameters_size_bytes(self.parameters)` multiplied by `active_clients` (same `active_clients` definition as upload).

---

## 11. Download Traffic and Overall Traffic Validation

| Check | Verdict | Detail |
|---|---|---|
| `download_traffic` from actual server-to-client payload | **PARTIAL** | Computed from current global `Parameters` byte size, multiplied by active count; aggregate estimate, not per-client measured transmission. |
| `overall_traffic = upload_traffic + download_traffic` | **FAIL (per-round interpretation)** | Logged `overall_traffic` is cumulative total (`total_upload_traffic + total_download_traffic`) up to current round, not per-round sum. |
| Includes both upload and download components | **PASS (cumulative)** | Both are accumulated and combined. |

---

## 12. FLOPs Logging Validation

### `round_flops`
- Produced on each client in `Client.fit` via `_estimate_round_flops(...)`.
- Includes estimated local training cost derived from:
  - profiled dense forward FLOPs per sample,
  - scaled by inferred parameter density,
  - multiplied for backward and optimizer cost constants.

### `total_flops`
- On server, `_update_flop_metrics` sums all client `round_flops` for the round, then cumulatively accumulates into `_flop_totals["total_flops"]`.

### What is included vs missing

| Component | Included in `round_flops` / `total_flops`? |
|---|---|
| Local training FLOPs | **Yes (estimated)** |
| Server-side aggregation FLOPs | **No** |
| Communication/compression FLOPs | **No in `round_flops`** |
| Evaluation FLOPs | **No** |

Conclusion: variable names exist and are consistent with local-train FLOP accounting only; they are not full end-to-end system FLOPs.

---

## 13. Compression / Decompression FLOPs Validation

Metrics present:
- `round_flops_compression`
- `round_flops_decompression`
- `total_flops_compression`
- `total_flops_decompression`
- `total_flops_including_compression`

Active behavior:
- Client sets defaults: compression/decompression round FLOPs = `0.0` unless overridden by task-specific code.
- No active client/server compression pipeline contributes non-zero values in audited path.
- Server simply aggregates those provided values.

Therefore, `total_flops_compression` currently reports cumulative zeros in the default SparsyFed path and does **not** capture real compression/decompression workload (because no such workload is implemented in transmission path).

---

## 14. Accuracy Logging Validation

`acc_servers_highest` is logged in `WandbHistory.add_metrics_centralized` by renaming centralized metric key `test_accuracy`.

Validation:
- Updated every time centralized evaluation metric includes `test_accuracy`.
- Uses server-side federated eval loader/test function (from `get_fed_eval_fn` path).
- Represents centralized global-model test accuracy for that round.
- Name `acc_servers_highest` is misleading: no best-so-far/max tracking logic is implemented; it logs round value directly.

---

## 15. Experiment Configuration Validation

Validation target requested:
- Dirichlet alpha = 0.5
- train/validation split = 80/20
- optimizer = SGD
- learning rate = 0.01
- batch size = 128

| Item | Verdict | Evidence |
|---|---|---|
| Dirichlet alpha = 0.5 | **PARTIAL** | Config default `dataset.lda_alpha: 0.5` in CIFAR dataset config. Enforced only if dataset was partitioned with this config; main pipeline does not regenerate partitions by default. |
| train/validation split = 80/20 | **PARTIAL** | `dataset.val_ratio: 0.2` exists in config. Active training reads pre-saved `train.pt`/`test.pt`; split enforcement depends on prior dataset preparation, not runtime training loop. |
| optimizer = SGD | **PASS (default CIFAR path)** | CIFAR `train`/`fixed_train` use `torch.optim.SGD`. |
| learning rate = 0.01 | **PASS (default CIFAR task config)** | `task.fit_config.run_config.learning_rate: 0.01` and passed through on_fit config. |
| batch size = 128 | **PASS (default CIFAR task config)** | `task.fit_config.dataloader_config.batch_size: 128` used by client train dataloader. |

---

## 16. WandB Metrics Audit

### Logged in active SparsyFed path (when `use_wandb=true`)

| Metric key in WandB | Source |
|---|---|
| `training_loss_highest` | centralized loss logging |
| `acc_servers_highest` | centralized `test_accuracy` key rename |
| `distributed_test_accuracy` | distributed eval rename of `test_accuracy` |
| `upload_traffic`, `download_traffic`, `upload_traffic_per_client`, `overall_traffic` | server per-round augmentation |
| `round_flops`, `round_flops_compression`, `round_flops_decompression` | aggregated from client metrics |
| `total_flops`, `total_flops_compression`, `total_flops_decompression`, `total_flops_including_compression` | server cumulative FLOP totals |
| `server_to_client_nonzero`, `client_to_server_nonzero`, `*_density`, `nonzero_communication_total`, `learning_rate` | propagated through fit metric aggregation/logging |

### Exists but potentially misleading
- `acc_servers_highest`: not best-so-far.
- `overall_traffic`: cumulative total, not isolated per-round total.
- compression FLOP totals: present but zero in active path.

---

## 17. Faithfulness to the Intended Method Structure

Expected SparsyFed-like structure implied by method name/repo organization vs implementation:

| Stage | Expected conceptually | Actual in code | Classification |
|---|---|---|---|
| Sparse-aware local computation | Sparse behavior in local train path | Implemented via SparsyFed custom modules/autograd in model forward/backward | **Correctly implemented** |
| Client-side communication compression | Sparse/encoded client upload representation | Not implemented; uploads are full parameters | **Missing** |
| Server-side decode/reconstruct | Rebuild sparse/quantized payloads | No method-specific decode stage | **Missing** |
| Communication-aware traffic from real compressed payload | Byte counts from actual compressed object | Uses parameter-size estimates and first-result proxy size | **Partially implemented** |
| Compression FLOP accounting | Non-zero compression/decompression operation costs | Metrics exist but default to zero | **Partially implemented / effectively missing ops** |
| End-to-end method-specific FL strategy | Dedicated SparsyFed aggregation/optimization path | Uses generic `FedAvgNZ` strategy; SparsyFed mostly in client model layers | **Implemented differently** |

---

## 18. Mismatches, Risks, and Ambiguities

1. **Method identity mismatch risk**: The repository labels SparsyFed mainly through model-layer substitutions, while communication path remains dense full-parameter exchange.
2. **Traffic metric ambiguity**: upload/download are estimated from parameter byte sizes and active client count including failures.
3. **`overall_traffic` naming risk**: variable sounds per-round but is cumulative.
4. **`acc_servers_highest` naming risk**: suggests running maximum; actual behavior logs per-round centralized accuracy.
5. **Dataset split/Dirichlet reproducibility ambiguity**: config specifies split and alpha, but runtime may use pre-existing partition files; generation function is not automatically invoked from main.
6. **Compression FLOPs completeness gap**: metric names imply broad accounting but active path contributes zero compression/decompression work.

---

## 19. Final Checklist

| Validation Item | Status |
|---|---|
| 1. Method activation and actual pipeline | **PASS** |
| 2. Post-local-training preprocessing | **PASS (none found, explicitly verified)** |
| 3. FLOPs reporting in wandb (`round_flops`, `total_flops`) | **PARTIAL** |
| 4. Client-side compression logic | **FAIL (missing in active payload path)** |
| 5. Flags/config and wandb linkage | **PASS** |
| 6. Client-to-server transmission process | **PASS** |
| 7. Upload traffic validation | **PARTIAL** |
| 8. Server-side decoding / reconstruction | **PASS (none in active path, explicitly verified)** |
| 9. Global training / server update | **PASS** |
| 10. Server-to-client model processing | **PASS (none in active path, explicitly verified)** |
| 11. Download traffic and overall traffic | **PARTIAL** |
| 12. Compression/decompression FLOPs | **PARTIAL** |
| 13. Server-side accuracy validation (`acc_servers_highest`) | **PARTIAL** |
| 14. Standard experiment configuration validation | **PARTIAL** |
| 15. Comparison with intended method behavior | **PASS** |

