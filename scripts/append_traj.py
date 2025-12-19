
from torch_traj_utils.trajectory import TrajectoryScenario, Trajectory, TrajectoryExpert
from torch_traj_utils.cartpole_solver_velocity import CartpoleSolverVelocity, CartpoleEnvironmentParams, SolverParams

from torch_traj_utils.plot_trajectory import plot_trajectory
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from typing import List

def doit(tdir_list: List[str], outfile, thing: str):
    all_states = []
    all_actions = []
    all_s0 = []
    for tdir in tdir_list:
        fname = tdir + thing
        fstuff = np.load(fname)
        all_states.append(fstuff['first_array'])
        all_actions.append(fstuff['second_array'])
        all_s0.append(fstuff['third_array'])

    all_states_out = np.concatenate(all_states,0)
    all_actions_out = np.concatenate(all_actions, 0)
    all_s0_out = np.concatenate(all_s0, 0)
    fname = outfile + thing
    np.savez(fname,first_array=all_states_out,
                 second_array=all_actions_out,
                 third_array=all_s0_out)
    return all_states_out

def main(tdir_list: List[str], outfile: str, update_npz:bool = True):
    good_data = 0
    num_data = 0
    training_split = 0.8 # so 20% for validation
    # use limits from the trajectory solver for normalization
    state_normalization = [0.22, np.pi, 0.8, 5*np.pi]
    action_normalization = 0.8

    all_states_training= doit(tdir_list, outfile, "_training.npz")
    doit(tdir_list, outfile, "_validation.npz")


    # probably more intereresting (bias in the training data)
    plt.figure()
    plt.hist(all_states_training[:,1]*np.pi, bins=20)
    #plt.plot(all_states_training[:,1],'+')
    plt.xlabel("θ")
    plt.ylabel("count")
    plt.title("Training data sample histogram")
    fname = outfile + "_theta_hist.png"
    plt.savefig(fname)
    plt.show()

if __name__ == "__main__":
    main(["trajectories_big_1","trajectories_big_2"], "trajectories_big_1p2")