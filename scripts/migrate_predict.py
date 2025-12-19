""" Load NN model and predict a closed-loop trajectory."""

import torch
import os
import torch.nn as nn
import numpy as np
from torch_traj_utils.trajectory import TrajectoryScenario, Trajectory
from torch_traj_utils.plot_trajectory import plot_trajectory
from torch_traj_utils.trajectory_metrics import TrajectoryMetrics, TrajMetric
from model_utils import load_checkpoint


def predict(trained_model_dir: str, trained_model_epoch: list, output_filename: str):

    tm = TrajectoryMetrics(goal_limits=np.array([0.25,0.1,0.1,0.1]))
    model = load_checkpoint(trained_model_dir, trained_model_epoch)
    # goal state: pole upright
    s_goal = np.array([0.00, np.pi, 0, 0])
    # sample a distribution of s0's
    s0_vec = np.array([0.0, np.pi/8, 0.0, 0.0])
    batch_size = 256

    tm.reset_random_batches(state_vec=s0_vec, num_batches=1, batch_size=batch_size, seed=42)
    tm.reset_accumulators()

    all_s, all_u = tm.collect_random_batch(model, 0, return_traj=True)

    # collect metrics
    metrics, metrics_ind = tm.collect_metrics()

    metrics.pretty_print()

    # off to the plotter
    #package up a trajectory
    ind = metrics_ind.max_goal_error_ind[1]
    #ind = metrics_ind.median_goal_error_ind[1]
    # compute energies
    energy1 = tm.solver.energy.compute_energy(all_s)
    energy2 = tm.solver.energy.compute_energy_pole(all_s)

    traj = Trajectory(s=all_s[ind,:,:],
                      u=all_u[ind,:,:],
                      N=tm.N,
                      dt=tm.solver_params.dt,
                      J=np.array([]),
                      energy1=energy1[ind,:],
                      energy2=energy2[ind,:],
                      sc=tm.sc,
                      conv=True,
                      status="NN_prediction",
                      time=-1,
                      iters=-1)

    plot_trajectory(solver_params=tm.solver_params,
                    env_params=tm.env_params,
                    traj=traj,
                    filename_base=output_filename,
                    animate=True)


# if __name__ == "__main__":
#     predict(trained_model_dir="trajectories_big_1_HD32_F32_B128_BM_BML2_L4",
#             trained_model_epoch=[145,],
#             output_filename="nn_test_bm_worst")

if __name__ == "__main__":
    predict(trained_model_dir="trajectories_big_1_HD32_F32_B128_BM_BML2_L4_best",
            trained_model_epoch=[145,],
            output_filename="nn_test_bm_worst")