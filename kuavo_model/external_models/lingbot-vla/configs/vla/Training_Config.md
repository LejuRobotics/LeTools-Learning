# Training Configuration Guide

This document explains all configuration parameters used in LingBot-VLA post-training (both **Real-World** and RoboTwin 2.0 simulation scenarios).

## Real-World Example
```yaml
model:
  model_path: robbyant/lingbot-vla-4b        # Path to pre-trained LingBot-VLA model (w/o or w/ depth)
  tokenizer_path: Qwen/Qwen2.5-VL-3B-Instruct

data:
  datasets_type: vla
  data_name: robot_config_filename # must be the same when computing normalization statistics
  train_path: path_to_dataset
  joints:
    - arm.position: 14
    - effector.position: 2
  cameras:
    - camera_top
    - camera_wrist_left
    - camera_wrist_right
  num_workers: 8
  norm_type: meanstd
  norm_stats_file: norm_path # must be the same when computing normalization statistics

train:
  output_dir: "output/"
  data_parallel_mode: fsdp2 # Use FSDP2 for model
  enable_full_shard: false
  module_fsdp_enable: true
  use_compile: true # Apply torch.compile() to model
  rmpad: false
  rmpad_with_pos_ids: false
  ulysses_parallel_size: 1
  freeze_vision_encoder: false
  tokenizer_max_length: 72
  max_action_dim: 75
  max_state_dim: 75
  lr: 5.0e-5
  lr_decay_style: constant
  micro_batch_size: 32
  gradient_accumulation_steps: 1 # global_batch_size = micro_batch_size * gradient_accumulation_steps * 8 = 256 when we train with 8 GPUs
  max_steps: 40000
  ckpt_manager: dcp
  save_steps: 10000 # save ckpt per 10k steps
  save_epochs: 0 # Disable epoch-based checkpointing  
  enable_fp32: true # Use float32 precision for the action expert
  enable_resume: true
  # ---- Depth Injection (only for LingBot-VLA w/ Depth) ----
  align_params:
    mode: 'query'
    num_task_tokens: 8
    use_image_tokens: True
    use_task_tokens: False
    use_text_tokens: False
    use_contrastive: True
    contrastive_loss_weight: 0.3
    depth_loss_weight: 0.004
    llm:
      dim_out: 2048
      image_token_size: 8
      image_input_size: 224
    depth:
      model_type: MoRGBD
      moge_path: "path/to/moGe-2-vitb-normal"
      morgbd_path: "path/to/LingBot-Depth"
      num_layers: 1
      num_heads: 4
      dim_head: 32
      ff_mult: 1
      num_backbone_tokens: 256
      token_size: 16
      dim_out: 1024
      input_size: 224
```

## RoboTwin 2.0 Example (5 tasks)

```yaml
model:
  model_path: robbyant/lingbot-vla-4b        # Path to pre-trained LingBot-VLA model (w/o or w/ depth)
  tokenizer_path: Qwen/Qwen2.5-VL-3B-Instruct

data:
  datasets_type: vla
  data_name: robotwin
  train_path: path_to_robotwin_dataset       # merged data from 5 robotwin2.0 tasks
  joints:
    - arm.position: 14
    - effector.position: 2
  cameras:
    - camera_top
    - camera_wrist_left
    - camera_wrist_right
  num_workers: 8
  norm_type: bounds_99
  norm_stats_file: assets/norm_stats/robotwin_50.json

train:
  output_dir: "output/"
  loss_type: L1_fm
  data_parallel_mode: fsdp2 # Use FSDP2 for model
  enable_full_shard: false
  module_fsdp_enable: true
  use_compile: true # Apply torch.compile() to model
  rmpad: false
  rmpad_with_pos_ids: false
  ulysses_parallel_size: 1
  freeze_vision_encoder: false
  tokenizer_max_length: 72
  max_action_dim: 75
  max_state_dim: 75
  lr: 1.0e-4
  lr_decay_style: constant
  micro_batch_size: 32
  gradient_accumulation_steps: 1 # global_batch_size = micro_batch_size * gradient_accumulation_steps * 8 = 256 when we train with 8 GPUs
  max_steps: 20000
  ckpt_manager: dcp
  save_steps: 20000 # save ckpt per 20k steps
  save_epochs: 0 # Disable epoch-based checkpointing  
  enable_fp32: true # Use float32 precision for the action expert
  enable_resume: true
  # ---- Depth Injection (only for LingBot-VLA w/ Depth) ----
  align_params:
    mode: 'query'
    num_task_tokens: 8
    use_image_tokens: True
    use_task_tokens: False
    use_text_tokens: False
    use_contrastive: True
    contrastive_loss_weight: 0.3
    depth_loss_weight: 0.004
    llm:
      dim_out: 2048
      image_token_size: 8
      image_input_size: 224
    depth:
      model_type: MoRGBD
      moge_path: "path/to/moGe-2-vitb-normal"
      morgbd_path: "path/to/LingBot-Depth"
      num_layers: 1
      num_heads: 4
      dim_head: 32
      ff_mult: 1
      num_backbone_tokens: 256
      token_size: 16
      dim_out: 1024
      input_size: 224
```

---

## Parameter Reference

### Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | — | Path to pre-trained VLA model weights. |
| `tokenizer_path` | str | - | Path to VLM. |

### Data

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_name` | str | — | Dataset name (e.g., `"robotwin"`). Must be the same when computing normalization statistics! |
| `train_path` | str | — | Path to training data directory (LeRobot v3.0 format). |
| `joints` | List[Dict] | — | Max dim of each named joints in data. |
| `cameras` | List[str] | — | Camera names in data. |
| `norm_type` | str | `"bounds_99"` | Normalization type. Options: `"meanstd"`, `"bounds_99"`, `"minmax"`, `"identity"`. |
| `norm_stats_file` | str | — | Path to pre-computed normalization statistics JSON file. Must be the same when computing normalization statistics! |

### Train — Batch Size & Gradient Accumulation

If you have limited GPU memory, we support enabling **gradient accumulation** through setting `gradient_accumulation_steps` > 1 to achieve a larger global batch size.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `micro_batch_size` | int | - | Number of samples per forward pass per GPU. |
| `global_batch_size` | int | `None` | Total batch size across all GPUs and accumulation steps. If `None`, auto-computed as `micro_batch_size × data_parallel_size(num_gpus) × gradient_accumulation_steps`. If set, must equal that value or an error is raised. |
| `gradient_accumulation_steps` | int | `1` | Number of gradient accumulation steps. `global_batch_size` is always derived from this value. |

**How gradient accumulation works:**

`global_batch_size` is always computed as `micro_batch_size × data_parallel_size(num_gpus) × gradient_accumulation_steps`. You only need to set `gradient_accumulation_steps`:

```yaml
micro_batch_size: 32
gradient_accumulation_steps: 2
# global_batch_size is auto-computed: 32 × num_gpus × 2
```

If you also set `global_batch_size` explicitly, it must be consistent with the computed value, otherwise an error is raised.

### Train — Training Duration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_train_epochs` | int | `None` | Number of training epochs. If `None`, trains indefinitely until `max_steps`. |
| `max_steps` | int | `None` | Global maximum number of update steps. If `None`, trains until all epochs complete. |

**How training duration is controlled:**

`num_train_epochs` and `max_steps` jointly control when training stops. At least one must be specified.

- **Only `max_steps`**: set `num_train_epochs` to `None`. Training runs across epochs indefinitely and stops at `max_steps`.
  ```yaml
  max_steps: 20000
  # num_train_epochs: not set → runs until 20000 steps
  ```

- **Only `num_train_epochs`**: set `max_steps` to `None`. Training runs for the specified number of epochs.
  ```yaml
  num_train_epochs: 69
  # max_steps: not set → runs all 69 epochs
  ```

- **Both specified**: training stops at whichever limit is reached first.
  ```yaml
  num_train_epochs: 69
  max_steps: 20000
  # stops at 20000 steps even if 69 epochs are not finished
  ```
> **Note:** When training stops at `max_steps`, a checkpoint is always saved automatically.

### Train — Other Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_type` | str | `"fm"` | Loss function. `"fm"` for MSE flow-matching, `"L1_fm"` for L1 flow-matching. |
| `data_parallel_mode` | str | `"ddp"` | Distributed data parallel strategy. Options: `"ddp"`, `"fsdp1"`, `"fsdp2"`. |
| `use_compile` | bool | `false` | Enable `torch.compile` for training acceleration. |
| `ckpt_manager` | str | `"dcp"` | Checkpoint backend. Options: `"dcp"` (PyTorch Distributed Checkpoint), `"bytecheckpoint"`. |
| `enable_fp32` | bool | `false` | Use float32 precision for the action expert. |
| `enable_resume` | bool | `false` | Automatically resume training from the latest checkpoint in `output_dir`. |


---

> **⚠️ Important:** Due to differences between real-world and simulation environments, their training configurations differ in two key aspects:
>
> | | Real-World | RoboTwin 2.0 |
> |---|---|---|
> | `norm_type` | `meanstd` | `bounds_99` |
> | `loss_type` | default (MSE flow-matching) | `L1_fm` (L1 flow-matching) |


---

## Example: Training Setting on A6000

You can fine-tune `LingBot-VLA-4B` on **4 × A6000 GPUs** platforms:

```bash
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin_load20000h.yaml \
    --data.train_path /path/to/mixed_robotwin_5tasks \
    --data.data_name robotwin \
    --data.norm_stats_file assets/norm_stats/robotwin_50.json \
    --train.output_dir output/ \
    --train.micro_batch_size 4 \
    --train.gradient_accumulation_steps 16

# train.global_batch_size will be auto-computed as:
# micro_batch_size × data_parallel_size(num_gpus) × gradient_accumulation_steps = 4 × 4 × 16 = 256
```
This will consume approximately 47424 / 49140 MB of VRAM per GPU.
