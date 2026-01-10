
from torch_traj_utils.trajectory import TrajectoryScenario, Trajectory, TrajectoryExpert
from torch_traj_utils.cartpole_solver_velocity import CartpoleEnvironmentParams, SolverParams
from torch_traj_utils.cartpole_solver_velocity_cas import CasadiSolverParams

from torch_traj_utils.plot_trajectory import plot_trajectory
import numpy as np
import pickle
import os

def main(tdir: str):
    good_data = 0
    good_data_goal = 0
    num_data = 0
    state_normalization = [0.22, np.pi, 0.8, 5 * np.pi]
    action_normalization = [0.8, ]
    for file in os.listdir(tdir):
        file = os.path.join(tdir, file)
        # unpickle!
        #pickle_fest = [env_params, solver_params, traj]
        with open(file,"rb") as f:
            pickle_fest = pickle.load(f)
        traj: Trajectory = pickle_fest[2]
        sp: SolverParams | CasadiSolverParams = pickle_fest[1]
        ep: CartpoleEnvironmentParams = pickle_fest[0]
        num_data += 1
        if traj.conv:
            good_data += 1
            final_state = traj.s[-1, :]
            goal_state = traj.sc.s_goal
            goal_error = np.abs(goal_state - final_state)/state_normalization
            # my solver settings are tuned to be a bit sloppy on final cart
            # position
            if np.max(goal_error[2]) < 0.1:
                good_data_goal += 1
            # plot!
            if (num_data % 10000) == 0:
                 plot_trajectory(solver_params=sp, env_params=ep,
                                  traj=traj, filename_base="",
                                  animate=False)
    print("converged: ", good_data, num_data, good_data/num_data)
    print("converged to goal: ", good_data_goal, num_data, good_data_goal / num_data)

if __name__ == "__main__":
    main("trajectories_4096_cas")