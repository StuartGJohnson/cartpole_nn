
from torch_traj_utils.trajectory import TrajectoryScenario, Trajectory, TrajectoryExpert
from torch_traj_utils.cartpole_solver_velocity import CartpoleSolverVelocity, CartpoleEnvironmentParams, SolverParams

from torch_traj_utils.plot_trajectory import plot_trajectory
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def main(tdir: str, update_npz:bool = True):
    good_data = 0
    num_data = 0
    training_split = 0.8 # so 20% for validation
    # use limits from the trajectory solver for normalization
    state_normalization = [0.22, np.pi, 0.8, 5*np.pi]
    action_normalization = 0.8
    all_states = []
    all_actions = []
    all_s0 = []
    for file in os.listdir(tdir):
        file = os.path.join(tdir, file)
        # unpickle!
        #pickle_fest = [env_params, solver_params, traj]
        with open(file,"rb") as f:
            pickle_fest = pickle.load(f)
        traj: Trajectory = pickle_fest[2]
        num_data += 1
        if traj.conv:
            final_state = traj.s[-1, :]
            goal_state = traj.sc.s_goal
            goal_error = np.abs(goal_state - final_state)/state_normalization
            if np.max(goal_error[1:]) < 0.1:
                good_data += 1
            else:
                continue
            states = traj.s
            actions = traj.u
            norm_states = states / state_normalization
            norm_action = actions / action_normalization
            # convert states to sin,cos format
            state_dim = norm_states.shape[1]
            n_states = norm_states.shape[0]
            #ml_states = np.zeros((n_states-1, state_dim+1))
            ml_states = np.zeros((n_states - 1, state_dim))
            # ml_states[:, 0] = norm_states[:-1,0]
            # ml_states[:, 1] = np.cos(norm_states[:-1,1])
            # ml_states[:, 2] = np.sin(norm_states[:-1,1])
            # ml_states[:, 3] = norm_states[:-1, 2]
            # ml_states[:, 4] = norm_states[:-1, 3]
            ml_states[:,:] = norm_states[:-1,:]
            all_states.append(ml_states)
            all_actions.append(norm_action)
            all_s0.append(traj.sc.s0 / state_normalization)
    training_cut = np.floor(good_data * training_split).astype(int)
    all_states_training = np.concatenate(all_states[0:training_cut],0)
    all_actions_training = np.concatenate(all_actions[0:training_cut], 0)
    all_s0_training = np.vstack(all_s0[0:training_cut])
    fname = tdir + "_training.npz"
    if update_npz:
        np.savez(fname,first_array=all_states_training,
                 second_array=all_actions_training,
                 third_array=all_s0_training)
    all_states_val = np.concatenate(all_states[training_cut:-1],0)
    all_actions_val = np.concatenate(all_actions[training_cut:-1], 0)
    all_s0_val = np.vstack(all_s0[training_cut:-1])
    fname = tdir + "_validation.npz"
    if update_npz:
        np.savez(fname,first_array=all_states_val,
                 second_array=all_actions_val,
                 third_array=all_s0_val)

    print(good_data/num_data)

    # these could be interesting
    # plt.figure()
    # plt.plot(all_states_training,'+')
    # plt.show()
    # plt.figure()
    # plt.plot(all_actions_training,'+')
    # plt.show()
    # plt.figure()
    # plt.plot(all_s0_training[:,1],'+')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(all_states_val,'+')
    # plt.show()
    # plt.figure()
    # plt.plot(all_actions_val,'+')
    # plt.show()
    # plt.figure()
    # plt.plot(all_s0_val[:,1],'+')
    # plt.show()

    # probably more intereresting (bias in the training data)
    plt.figure()
    plt.hist(all_states_training[:,1]*np.pi, bins=20)
    #plt.plot(all_states_training[:,1],'+')
    plt.xlabel("θ")
    plt.ylabel("count")
    plt.title("Training data sample histogram")
    fname = tdir + "_theta_hist.png"
    plt.savefig(fname)
    plt.show()

if __name__ == "__main__":
    main("trajectories_4096_cas", update_npz=True)