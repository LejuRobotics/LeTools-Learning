# Generate Lerobot Dataset from RoboTwin Data

This guide explains how to process raw data from **RoboTwin** and convert it into the **LerobotDataset** format following the official RoboTwin instructions.

## 1. Clone the Official RoboTwin Repository
```bash
git clone git@github.com:RoboTwin-Platform/RoboTwin.git
```

## 2. Create Required Directories
Navigate to the `policy/pi0` directory inside the cloned RoboTwin repository and create the folders:

```bash
cd ./policy/pi0
mkdir processed_data training_data
```

## 3. Convert RoboTwin Raw Data to HDF5

Download [official dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset) and unzip the dataset to '/path/to/RoboTwin/data'

**Example:**
```bash
data
└── adjust_bottle
    └── aloha-agilex_clean_50
```

Use the provided script [process_data_pi0.sh](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/policy/pi0/process_data_pi0.sh):

```bash
cd policy/pi0
bash process_data_pi0.sh ${task_name} ${task_config} ${expert_data_num}
```

**Example (clean demo):**
```bash
bash process_data_pi0.sh adjust_bottle aloha-agilex_clean_50 50
```

**Example (randomized demo):**
```bash
bash process_data_pi0.sh adjust_bottle aloha-agilex_randomized_500 50
```

If successful, the output folder:
```
processed_data/${task_name}-${task_config}-${expert_data_num}/
```

## 4. Prepare Training Data

Copy the required processed datasets into `training_data/${model_name}`:

```bash
cp -r processed_data/${task_name}-${task_config}-${expert_data_num} \
      training_data/${model_name}/
```

## 5. Ensure Sufficient Disk Space

The generated **LerobotDataset** will be stored under:

```
$XDG_CACHE_HOME/huggingface/lerobot/${repo_id}
```

By default, `XDG_CACHE_HOME` points to `~/.cache`, which must have sufficient free space.  
If space is low, change the cache location:

```bash
export XDG_CACHE_HOME=/path/to/your/cache
```

## 6. Generate LerobotDataset v2.1 Format

Run [generate.sh ](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/policy/pi0/generate.sh) to convert the HDF5 datasets to Lerobot.

Parameters:
- **hdf5_path**: Path to the HDF5 training data (e.g., `./training_data/${model_name}/`)
- **repo_id**: Name for the dataset (e.g., `my_repo`)

```bash
bash generate.sh ${hdf5_path} ${repo_id}
```

**Example:**
```bash
bash generate.sh ./training_data/demo_clean/ demo_clean_repo
```

Output:
```
${XDG_CACHE_HOME}/huggingface/lerobot/${repo_id}
```