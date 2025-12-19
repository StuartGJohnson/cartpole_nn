""" filter a trajectory dataset (.npz). Figures out trajectory length by inspecting
the lengths of s0 and the states."""

from torch_traj_utils.trajectory import TrajectoryScenario, Trajectory, TrajectoryExpert
from torch_traj_utils.cartpole_solver_velocity import CartpoleSolverVelocity, CartpoleEnvironmentParams, SolverParams

from torch_traj_utils.plot_trajectory import plot_trajectory
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from typing import List

def main(infile: str, outfile: str, update_npz:bool = True):
    good_data = 0
    num_data = 0

    # use limits from the trajectory solver for normalization
    state_normalization = [0.22, np.pi, 0.8, 5*np.pi]
    action_normalization = 0.8
    goal_state = np.array([0.0, np.pi, 0.0, 0.0]) / state_normalization

    all_states = []
    all_actions = []
    all_s0 = []
    fname = infile
    fstuff = np.load(fname)
    all_states.append(fstuff['first_array'])
    all_actions.append(fstuff['second_array'])
    all_s0.append(fstuff['third_array'])

    all_states_out = np.concatenate(all_states,0)
    all_actions_out = np.concatenate(all_actions, 0)
    all_s0_out = np.concatenate(all_s0, 0)

    # determine the trajectory length (this assumes
    # they have the same length throughout
    len_t = int(all_states_out.shape[0]/all_s0_out.shape[0])
    n_t = int(all_states_out.shape[0] / len_t)
    n_s = all_s0_out.shape[1]
    n_u = all_actions_out.shape[1]

    # reorder arrays into trajectories
    states = np.reshape(all_states_out, (n_t, len_t, n_s))
    actions = np.reshape(all_actions_out, (n_t, len_t, n_u))
    s0 = all_s0_out

    # look at (nearly the last state)
    goal_error = np.abs(goal_state - states[:,-1,:])
    print(np.sum(goal_error <0.2, axis=0)/n_t)


    # fname = outfile + thing
    # np.savez(fname,first_array=all_states_out,
    #              second_array=all_actions_out,
    #              third_array=all_s0_out)
    # return all_states_out


    # # probably more intereresting (bias in the training data)
    # plt.figure()
    # plt.hist(all_states_training[:,1]*np.pi, bins=20)
    # #plt.plot(all_states_training[:,1],'+')
    # plt.xlabel("θ")
    # plt.ylabel("count")
    # plt.title("Training data sample histogram")
    # fname = outfile + "_theta_hist.png"
    # plt.savefig(fname)
    # plt.show()

if __name__ == "__main__":
    main("trajectories_big_1_training.npz", "thing")