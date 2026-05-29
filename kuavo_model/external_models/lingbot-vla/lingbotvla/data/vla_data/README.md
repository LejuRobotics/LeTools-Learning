
---

# Customized Downstream Task Dataset Construction

## Custom Data
This guide explains how to construct a custom downstream task dataset for post-training and how to deploy it on corresponding downstream tasks. 

## 1. Merge Datasets (Optional)

If you have multiple tasks with **the same robot configuration** and want to train on them jointly, you can merge them before conversion. It is more efficient to merge datasets while they are still in **LeRobot v2.1 format**, and then convert the merged dataset to v3.0 in one go.

For example, to merge the 5 RoboTwin 2.0 tasks ("open_microwave", "click_bell", "stack_blocks_three", "place_shoe", "put_object_cabinet"), **all belonging to** the aloha-agilex robot, you can run:

```bash
python scripts/merge_lerobot_v21.py --sources path_to_open_microwave-aloha-agilex_clean_50-50,path_to_click_bell-aloha-agilex_clean_50-50,path_to_stack_blocks_three-aloha-agilex_clean_50-50,path_to_place_shoe-aloha-agilex_clean_50-50,path_to_put_object_cabinet-aloha-agilex_clean_50-50 --output path_to_merged_data
```

> **Important:** Merged datasets must share the **same robot configuration**, i.e., they must all use the same `configs/robot_configs/<data_name>.yaml` in Step 3.


## 2. Prepare LeRobot Dataset (v3.0)

LingbotVLA loads data through the [LeRobot](https://github.com/huggingface/lerobot) library (`LeRobotDataset`). Your dataset must be in **LeRobot v3.0 format** — either as a HuggingFace Hub repo or a local directory.


Convert Lerobot v2.1 to v3.0:
see [LeRobot](https://github.com/huggingface/lerobot/blob/v0.4.2/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py) for details.


## 3. Prepare Robot Config

The robot config defines how your robot's raw data features (in LeRobot format) map to the unified feature space used by LingbotVLA. Create a YAML file at `configs/robot_configs/<data_name>.yaml` — the filename must match `data.data_name` in your VLA training config.

The config has three top-level sections: **states**, **actions**, and **images**.

#### States

Maps raw observation keys to unified state features (`observation.state.<joint_type>`). Each entry specifies `origin_keys` with `start`/`end` slice indices. Multiple slices are **concatenated** in order. The following example is from `configs/robot_configs/robotwin.yaml` (a bimanual robot with 12-dim arm state and 2-dim gripper state):

```yaml
states:
  - observation.state.arm.position:
      origin_keys:
        - observation.state:       # left arm joints [0:6)
            start: 0
            end: 6
        - observation.state:       # right arm joints [7:13)
            start: 7
            end: 13

  - observation.state.effector.position:
      origin_keys:
        - observation.state:       # left gripper [6:7)
            start: 6
            end: 7
        - observation.state:       # right gripper [13:14)
            start: 13
            end: 14
```

#### Actions

Same structure as states, mapping raw action keys to unified action features (`action.<joint_type>`). The optional `subtract_state` flag controls whether to convert actions to deltas by subtracting the corresponding state. 

Continuing with the RoboTwin example:

```yaml
actions:
  - action.arm.position:
      origin_keys:
        - action:
            start: 0
            end: 6
        - action:
            start: 7
            end: 13
      subtract_state: False

  - action.effector.position:
      origin_keys:
        - action:
            start: 6
            end: 7
        - action:
            start: 13
            end: 14
      subtract_state: False
```
> <p><span style="color:red; font-size:1.em; font-weight:bold;">Note</span>: We recommend setting <code>subtract_state</code> of <code>action.arm.position</code> to <code>True</code> and for<code>action.effector.position</code> to <code>False</code> when training the model with real-world data.</p>


#### Images

Maps raw camera keys to unified camera names. Use `origin_keys` when the original key differs from the target. In the RoboTwin example, the original dataset uses `cam_high`, `cam_left_wrist`, `cam_right_wrist`, which are remapped to the unified names:

```yaml
images:
  - observation.images.camera_top:
      origin_keys: observation.images.cam_high
  - observation.images.camera_wrist_left:
      origin_keys: observation.images.cam_left_wrist
  - observation.images.camera_wrist_right:
      origin_keys: observation.images.cam_right_wrist
```

If the original key already matches the target, use the short form (just the string):

```yaml
images:
  - observation.images.camera_top
```

#### Consistency with VLA Training Config

The joint types and camera names in the robot config must match those declared in your VLA training config (`configs/vla/<config>.yaml`). The `joints` dimensions must equal the total length of all concatenated slices for that joint type. For example, `robotwin.yaml` uses:

| Joint Type | Slices | Total Dim |
|---|---|---|
| `arm.position` | [0:6) + [7:13) | 12 |
| `effector.position` | [6:7) + [13:14) | 2 |

Corresponding VLA training config:

```yaml
# configs/vla/robotwin_load20000h.yaml
data:
  data_name: robotwin           # must match robot config filename
  joints:
    - arm.position: 14          # must >= total dim of concatenated arm position slices (6+6=12); 14 recommended for padding headroom
    - effector.position: 2      # must >= total dim of concatenated effector position slices (1+1=2); 2 recommended for padding headroom
  cameras:
    - camera_top
    - camera_wrist_left
    - camera_wrist_right
```

> See `configs/robot_configs/robotwin.yaml` for a complete example.

> **Important:** 
 - The `<joint_type>` used in states (`observation.state.<joint_type>`) and actions (`action.<joint_type>`) must be defined in `data.joints` of your VLA training config. 
 - Camera names in the images section (`observation.images.<camera_name>`) must be listed in `data.cameras`. 

For example, if `configs/vla/robotwin_load20000h.yaml` declares `joints: [{arm.position: 14}, {effector.position: 2}]` and `cameras: [camera_top, camera_wrist_left, camera_wrist_right]`, then only these joint types and camera names are valid in the robot config. Using an undefined joint type or camera name will raise a `ValueError` at runtime.

> **Note:** You can define additional joint types beyond `arm.position` and `effector.position` by adding new entries to `data.joints`. If you add end-effector (EEF) dimensions, we recommend learning **absolute action**(subtract_state: False), as relative rotation computation is not currently supported.


## 4. Compute Normalization Statistics

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm.py ./configs/vla/real_load20000h.yaml \
    --data.data_name robot_config_filename(e.g. robotwin) \
    --data.train_path path_to_lerobotv3_path(e.g. path_to_merged_data) \
    --data.norm_stats_file norm_path.json
```
**The normalization file will be saved at data.norm_stats_file**

## 5. Training

After computing normalization statistics and obtaining `norm_path.json`, you can start post-training:

```bash
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/real_load20000h.yaml \
    --data.data_name robot_config_filename(e.g. robotwin) \
    --data.train_path path_to_lerobotv3_path(e.g. path_to_merged_data) \
    --data.norm_stats_file norm_path.json \
    --train.output_dir output/
```