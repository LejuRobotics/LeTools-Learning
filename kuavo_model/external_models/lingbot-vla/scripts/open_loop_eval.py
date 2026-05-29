import logging
from pathlib import Path
import argparse
import os

import numpy as np
from matplotlib import pyplot as plt

import torch

from deploy.lingbot_vla_policy import LingbotVLAServer
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id}",
        fontsize=16,
        color="blue",
    )
 
    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]
        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro", label="inference point")
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory



def evaluate_single_trajectory(
    policy,
    dataset,
    traj_id: int,
    modality_keys: list[str] | None = None,
    steps=300,
    action_horizon=16,
    save_plot_path=None,
    max_infer_time = 10
):
    # Ensure steps doesn't exceed trajectory length
    start_id, end_id = dataset.meta.episodes[traj_id]["dataset_from_index"], dataset.meta.episodes[traj_id]["dataset_to_index"]
    
    gt_action_across_time = []
    state_joints_across_time = []
    pred_action_across_time  = []

    count = 0
    for data_id in range(start_id, end_id, action_horizon):
        traj = dataset[data_id]
        count += 1

        for image_key in policy.vla.feature_transform.org_features['images']:
            image = (traj[image_key]* 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            traj[image_key] = image
        preds = policy.infer(traj)

        gt_action_across_time += [np.concatenate([traj[action_feature][:action_horizon] for action_feature in policy.vla.feature_transform.org_features['actions']], axis=-1)]
        state_joints_across_time += [np.concatenate([traj[state_feature] for state_feature in policy.vla.feature_transform.org_features['states']], axis=-1)]
        pred_action_across_time += [np.concatenate([preds[action_feature] for action_feature in policy.vla.feature_transform.org_features['actions']], axis=-1)]
        
        if count >=max_infer_time: break
    
    gt_action_across_time = np.concatenate(gt_action_across_time, axis=0)
    state_joints_across_time = np.concatenate(state_joints_across_time, axis=0)
    pred_action_across_time = np.concatenate(pred_action_across_time, axis=0)
    
    pred_action_across_time = np.array(pred_action_across_time)
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # calc MSE and MAE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")

    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        action_keys=policy.vla.feature_transform.org_features['actions'],
        action_horizon=action_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
    )

    return mse, mae




def main(policy, robo_name, data_root, traj_ids, chunk_size, save_plot_path):

    policy.data_config.num_episode = None
    policy.data_config.chunk_size = policy.config.chunk_size

    policy.data_config.train_path = data_root
    policy.data_config.data_name = robo_name

    data_path = Path(data_root)
    if data_path.is_absolute() and data_path.exists():
        repo_id = data_path.name
        root = data_path
    else:
        repo_id = data_root
        root = None
    dataset_meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = {}
    for action_feature in policy.vla.feature_transform.org_features['actions']:
        delta_timestamps[action_feature] = [t / dataset_meta.fps for t in range(policy.config.chunk_size)]
    dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)
    print(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {traj_ids}")

    all_mse = []
    all_mae = []

    for traj_id in traj_ids:
        if traj_id not in dataset.meta.episodes['episode_index']:
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        print(f"Running trajectory: {traj_id}")
        mse, mae = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            save_plot_path=os.path.join(save_plot_path,f'{traj_id}.png'),
            action_horizon=chunk_size,
        )
        print(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
        all_mse.append(mse)
        all_mae.append(mae)

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        print(f"Average MSE across all trajs: {avg_mse}")
        print(f"Average MAE across all trajs: {avg_mae}")
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="open loop test")

    parser.add_argument('--model_path',  type=str, required=True)

    parser.add_argument('--robo_name',   type=str, default=None, help='robot type')
    parser.add_argument('--norm_path',   type=str, default=None, help='norm file path of training data')
    parser.add_argument('--data_path',   type=str, default=None, help='path of validation data')
    
    parser.add_argument('--traj_ids',    type=int, nargs='+', default=[0])

    parser.add_argument('--use_length',  type=int, default=50, help='use length of action chunk')
    parser.add_argument("--num_denoising_step", type=int, default=10, help="num of denoising step")
    parser.add_argument("--use_compile", action='store_true', help="use torch compile or not")

    parser.add_argument('--save_plot_path', type=str, default='./open_loop_test/')
    args = parser.parse_args()

    os.makedirs(args.save_plot_path, exist_ok=True)
    traj_ids = args.traj_ids

    model = LingbotVLAServer(
                path_to_pi_model=args.model_path,
                robot_norm_path=args.norm_path,
                use_length=args.use_length,
                use_bf16=True,
                num_denoising_step=args.num_denoising_step,
                use_compile=args.use_compile
            )
    robo_name = args.robo_name if args.robo_name is not None else model.data_config.data_name
    data_path = args.data_path if args.data_path is not None else model.data_config.train_path
    
    model.reset(robo_name)
    main(model, robo_name, data_path, traj_ids, args.use_length, args.save_plot_path)