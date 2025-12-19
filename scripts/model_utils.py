""" tools for nn models.
"""

import torch
import os
import glob
import torch.nn as nn
import numpy as np
from torch_traj_utils.scp_solver import SolverParams
from torch_traj_utils.cartpole_solver_velocity import CartpoleSolverVelocity, CartpoleEnvironmentParams
from torch_traj_utils.trajectory import TrajectoryScenario, Trajectory
from torch_traj_utils.plot_trajectory import plot_trajectory
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import Tensor
from dataclasses import dataclass
from typing import List, Tuple, Any

from model_factory import BoundedModel, BoundedModel3, HybridModel
from quant_model_factory import QuantBoundedModel, BitSpec
from model_factory import BoundedModelLoss, BoundedModelLoss2, HybridModelLoss1, HybridModelLoss2

def build_model(model_name, input_dim, hidden_dim, output_dim, u_max, bitspec_name=""):
    if bitspec_name == "":
        bs = None
    elif bitspec_name == "BS16":
        bs = BitSpec("BS16", 16, 16, 32, 16)
    elif bitspec_name == "BS8":
        bs = BitSpec("BS8", 8, 8, 16, 8)
    else:
        raise "unknown bitspec type:" + bitspec_name

    if model_name == "BM3":
        model = BoundedModel3(input_dim, hidden_dim, output_dim, u_max)
    elif model_name == "BM":
        model = BoundedModel(input_dim, hidden_dim, output_dim, u_max)
    elif model_name == "HM":
        model = HybridModel(input_dim, hidden_dim, output_dim, u_max)
    elif model_name == "QBM":
        model = QuantBoundedModel(input_dim, hidden_dim, output_dim, u_max, bs)
    else:
        raise "unknown model type:" + model_name
    return model, bs

def build_loss(loss_name: str):
    if loss_name == "BML2":
        loss_fn = BoundedModelLoss2()
    elif loss_name == "BML":
        loss_fn = BoundedModelLoss()
    elif loss_name == "HML1":
        loss_fn = HybridModelLoss1()
    elif loss_name == "HML2":
        loss_fn = HybridModelLoss2()
    else:
        raise "unknown loss type:" + loss_name
    return loss_fn


def load_checkpoint(trained_model_dir: str, trained_model_epoch: List[int]) -> Tuple[nn.Sequential | nn.Module, BitSpec]:
    tdir = trained_model_dir
    epoch_num = trained_model_epoch
    # work around the dubious hack of
    # forcing the user's directory structure
    if "train" not in tdir:
        fdir = os.path.join("train",tdir)
    else:
        fdir = tdir
    epoch_str = ""
    for sub_epoch in trained_model_epoch:
        epoch_str += "_"
        epoch_str += str(sub_epoch)
    fname = "*" + epoch_str + ".pth"
    fpath = os.path.join(fdir, fname)
    matches = glob.glob(fpath)
    if len(matches) == 0:
        raise FileNotFoundError("No files matched the pattern")
    elif len(matches) > 1:
        raise RuntimeError(f"Multiple files matched: {matches}")
    path = matches[0]
    ckpt = torch.load(path, map_location="cpu")

    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt["hidden_dim"]
    output_dim = ckpt["output_dim"]
    dropout_rate = ckpt["dropout_rate"]
    model_type = ckpt["model_type"]
    u_max = 1.05
    if "u_max" in ckpt:
        u_max = ckpt["u_max"]
    if "bitspec_type" in ckpt:
        bitspec_type = ckpt["bitspec_type"]

    model, bs = build_model(model_type, input_dim, hidden_dim, output_dim, u_max, bitspec_type)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, bs
