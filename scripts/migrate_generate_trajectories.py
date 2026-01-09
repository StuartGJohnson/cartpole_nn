"""
First cut at generating some training data.
This code writes trajectories to disk.
"""

from torch_traj_utils.trajectory import TrajectoryScenario, Trajectory, TrajectoryExpert
from torch_traj_utils.cartpole_solver_velocity import CartpoleSolverVelocity, CartpoleEnvironmentParams, SolverParams
from torch_traj_utils.cartpole_solver_velocity_cas import CartpoleSolverVelocityCas, CasadiSolverParams
from torch_traj_utils.cartpole_expert import CartpoleSwingupExpert
import numpy as np
import pickle
import os
from concurrent.futures import ProcessPoolExecutor
import math

known_worker_types = ["scp_velocity", "cas_velocity"]

def worker(trial_range: range, worker_num: int, output_dir: str, worker_type: str):
    env_params = CartpoleEnvironmentParams(pole_length=0.395,
                            pole_mass=0.087,
                            cart_mass=0.230,
                            cart_length=0.044,
                            track_length=0.44,
                            max_cart_force=1.77,
                            max_cart_speed=0.8,
                            cart_tau=0.25,
                            n=4,
                            m=1)

    if worker_type not in known_worker_types:
        raise ValueError("Invalid worker type!")
    elif worker_type == "scp_velocity":
        solver_params = SolverParams(dt=0.05,
                                  P=1e3 * np.eye(4),
                                  Q=np.diag([10, 2, 1, 0.25]), #Q = np.diag([1e-2, 1.0, 1e-3, 1e-3]) (quadratic cost)
                                  R=0.001 * np.eye(1),
                                  Rd=0.0 * np.eye(1),
                                  rho=0.05,
                                  rho_u=0.02,
                                  eps=0.005,
                                  cvxpy_eps=1e-4,
                                  max_iters=1000,
                                  u_max=np.array([0.8]),
                                  s_max=np.array([0.44 / 2.0, 1000, 0.8, 5*np.pi])[None, :],
                                  max_solve_secs=-1.0,
                                  solver_type="OSQP")
        solver = CartpoleSolverVelocity(ep=env_params, sp=solver_params)
        expert = CartpoleSwingupExpert(ep=env_params, solver=solver)
    elif worker_type == "cas_velocity":
        solver_params = CasadiSolverParams(
            dt=0.02,
            Q=np.diag([100, 1.0, 0.5, 0.5]),
            R=0.5 * np.diag([50.0]),
            Rd=0.5 * np.diag([50.0]),
            P=np.diag([200.0, 2000.0, 1000.0, 1000.0]),
            # P=np.diag([1.0, 1.0, 1.0, 1.0]),
            s_max=np.array([0.44 / 2.0, 1.5 * np.pi, 0.8, 5 * np.pi])[None, :],
            u_max=np.array([0.8]),
            optimize_time=False,  # set False to keep fixed dt
            dt_min=0.001,
            dt_max=0.02,
            time_weight=0.0,
            max_solve_secs=-1.0,
            ipopt_max_iter=2000,
            ipopt_tol=1e-6,
            ipopt_print_level=0,
            ipopt_sb="yes",
            print_time=False
        )
        solver = CartpoleSolverVelocityCas(ep=env_params, sp=solver_params)
        expert = CartpoleSwingupExpert(ep=env_params, solver=solver)
    else:
        raise ValueError("Invalid worker type!")

    tdir = output_dir
    os.makedirs(tdir, exist_ok=True)

    # goal state: pole upright
    s_goal = np.array([0.00, np.pi, 0, 0])

    # start loop - sample a distribution of s0's
    uvec = np.array([0.05, np.pi/4, 0.2, 0.5])

    np.random.seed(worker_num)

    for i in trial_range:
        s0 = np.squeeze(np.random.uniform(-uvec, uvec, size=(1, 4)))
        scenario= TrajectoryScenario(s_goal=s_goal, s0=s0, t0=0.0, T=4.5)
        traj = expert.trajectory(scenario)
        # plot_trajectory(env_params=env_params,
        #                 traj=traj, filename_base="cartpole_velocity",
        #                 animate=True)

        # pickle!
        pickle_fest = [env_params, solver_params, traj]
        pkl_name = f"cartpole_velocity_{i}.pkl"
        filename = os.path.join(tdir, pkl_name)
        with open(filename,"wb") as f:
            pickle.dump(pickle_fest, f)

def chunk_range(n: int, n_chunks: int):
    """Yield `range` objects that partition range(n) into ~equal chunks."""
    chunk_size = math.ceil(n / n_chunks)
    for start in range(0, n, chunk_size):
        yield range(start, min(start + chunk_size, n))

def run_parallel(N: int, n_workers: int, output_dir:str, worker_type:str) -> float:
    ranges = list(chunk_range(N, n_workers))
    worker_nums = list(range(0, n_workers))

    # do this before spawning workers
    if worker_type not in known_worker_types:
        raise ValueError("Invalid worker type")

    print("\n")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(worker, r, w_num, output_dir, worker_type)
            for r, w_num in zip(ranges, worker_nums)
        ]
        #results = [f.result() for f in futures]
    # combine however you like
    #return sum(results)


if __name__ == "__main__":
    run_parallel(N=4096, n_workers=12, output_dir="trajectories_4096_cas", worker_type="cas_velocity")