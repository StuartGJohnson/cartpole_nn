# nn model training

import numpy as np
import sklearn.utils
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torch_traj_utils.trajectory_metrics import TrajectoryMetrics, TrajMetric
from torch_traj_utils.load_training_dataset import load_dataset
from sklearn.utils import shuffle
from typing import Tuple
from model_factory import BoundedModel, BoundedModelLoss, BoundedModelLoss2, BoundedModel3
from model_factory import HybridModel, HybridModelLoss1, HybridModelLoss2, make_hybrid_mode_labels
import os
import logging
import sys
from model_utils import build_model, build_loss

def eval_loss(model: nn.Sequential | nn.Module, loader: DataLoader, loss_fn, u_max=1.05, u_thresh=0.95):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    p = next(model.parameters())
    loss_gt = loss_fn.my_gt_type()
    with torch.no_grad():
        # iterate over batches
        for xb, yb in loader:
            xb = xb.to(dtype=p.dtype, device=p.device)
            yb = yb.to(dtype=p.dtype, device=p.device)
            pred = model(xb)
            if loss_gt == 2:
                yb2 = make_hybrid_mode_labels(yb, u_max, u_thresh)
                yb2 = yb2.to(device=p.device)
                loss = loss_fn(pred, (yb, yb2))
            else:
                loss = loss_fn(pred, yb)
            # Weight by batch size to get dataset-wide mean later
            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
    return total_loss / n_samples

def train(model_name: str, loss_name: str, dname: str, output_dir: str, num_epochs: int, model_type: torch.dtype, hidden_dim:int, batch_size:int, u_max:float, restart_epoch: int = -1):
    """
    dname: directory name for training
    restart_epoch: if > 0, used to reload
        previous model weights and continue training
    """
    # load data - already split
    fname = dname + "_training.npz"
    train_ds, train_s0, input_dim, output_dim = load_dataset(fname)
    fname = dname + "_validation.npz"
    val_ds, val_s0, _, _ = load_dataset(fname)

    #batch_size = 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_s0_loader = DataLoader(train_s0, batch_size=batch_size, shuffle=False)
    val_s0_loader = DataLoader(val_s0, batch_size=batch_size, shuffle=False)

    tm = TrajectoryMetrics()

    #hidden_dim = 32
    #u_max = 1.00
    u_thresh = 0.95

    model, bs = build_model(model_name, input_dim, hidden_dim, output_dim, u_max, bitspec_name="")

    loss_fn = build_loss(loss_name)

    output_dir = output_dir + "_" + model.my_name() + "_" + loss_fn.my_name() + "_L" + str(model.my_linear_ops())


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model = model.to(device=device, dtype=model_type)
    #model = model.to(device=device)

    train_dir = os.path.join("train",output_dir)
    os.makedirs(train_dir, exist_ok=True)

    logging_file =  os.path.join(train_dir,"training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(logging_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if restart_epoch > 0:
        fname = dname + "_" + str(restart_epoch) + ".pth"
        fpath = os.path.join(train_dir, fname)
        ckpt = torch.load(fpath, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        restart_epoch = restart_epoch + 1
    else:
        restart_epoch = 0

    train_loss_vec = []
    val_loss_vec = []
    epoch_vec = []
    train_overall_success_rate = []
    val_overall_success_rate = []

    p = next(model.parameters())
    loss_gt = loss_fn.my_gt_type()

    for epoch in range(restart_epoch, restart_epoch + num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dtype=p.dtype, device=p.device)
            yb = yb.to(dtype=p.dtype, device=p.device)

            # forward
            pred = model(xb)
            if loss_gt == 2:
                yb2 = make_hybrid_mode_labels(yb, u_max, u_thresh)
                yb2 = yb2.to(device=p.device)
                loss = loss_fn(pred, (yb, yb2))
            else:
                loss = loss_fn(pred, yb)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if epoch % 5 == 0:
            train_loss = eval_loss(model, train_loader, loss_fn, u_max, u_thresh)
            val_loss = eval_loss(model, val_loader, loss_fn, u_max, u_thresh)

            logger.info(
                f"Epoch {epoch:3d}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f} "
            )

            tm.reset_accumulators()
            train_metrics = tm.eval_metrics(model, train_s0_loader, device)
            tm.reset_accumulators()
            val_metrics = tm.eval_metrics(model, val_s0_loader, device)
            logger.info(f"Epoch {epoch:3d}: Training metrics:")
            train_metrics.pretty_print(logger)
            logger.info(f"Epoch {epoch:3d}: Validation metrics:")
            val_metrics.pretty_print(logger)

            train_loss_vec.append(train_loss)
            val_loss_vec.append(val_loss)
            epoch_vec.append(epoch)
            train_overall_success_rate.append(train_metrics.all_success)
            val_overall_success_rate.append(val_metrics.all_success)

            checkpoint = {"model_state": model.state_dict(),
                          "optimizer_state": optimizer.state_dict(),
                          "input_dim": input_dim,
                          "output_dim": output_dim,
                          "hidden_dim": hidden_dim,
                          "linear_ops": model.my_linear_ops(),
                          "model_type": model.my_name(),
                          "loss_type": loss_fn.my_name(),
                          "final_tanh": model.my_has_final_tanh(),
                          "tanh_scale": model.my_u_scale(),
                          "u_max": u_max,
                          "dropout_rate": -1}
            cname = dname + "_" + str(epoch) + ".pth"
            cpath = os.path.join(train_dir, cname)
            torch.save(checkpoint, cpath)

    plt.figure()
    plt.plot(epoch_vec, train_loss_vec, 'r', label="training")
    plt.plot(epoch_vec, val_loss_vec, 'g', label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Learning curves: loss")
    fname = "learning_curves_" + str(restart_epoch) + ".png"
    lcname = os.path.join(train_dir, fname)
    plt.savefig(lcname)
    plt.show()

    plt.figure()
    plt.plot(epoch_vec, train_overall_success_rate, 'r', label="training")
    plt.plot(epoch_vec, val_overall_success_rate, 'g', label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("All success rate")
    plt.legend()
    plt.title("Learning curves: metrics")
    fname = "metric_curves_" + str(restart_epoch) + ".png"
    lcname = os.path.join(train_dir, fname)
    plt.savefig(lcname)
    plt.show()

if __name__ == "__main__":
    train(
        model_name="BM",
        loss_name="BML2",
        dname="trajectories_4096_cas",
        output_dir="trajectories_4096_cas_HD32_F32_B128",
        model_type=torch.float32,
        hidden_dim=32,
        batch_size=128,
        u_max=1.0,
        num_epochs=300,
        restart_epoch=-1)
