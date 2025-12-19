""" (Not really a) model factory for pytorch cartpole NN models. Provides models and loss functions."""
from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
from typing import Tuple, Optional, Union

class LearnableScaledTanh(nn.Module):
    def __init__(self, scale: float, init_alpha: float = 1.0):
        super().__init__()
        self.scale = float(scale)
        self.log_alpha = nn.Parameter(torch.tensor(float(init_alpha)).log())
    def forward(self, z):
        alpha = self.log_alpha.exp()
        return self.umax * torch.tanh(alpha * z)

class ScaledTanh(nn.Module):
    """
    Applies the Tanh function and then scales the output by a constant factor.
    Output range will be [-scale, scale].
    """
    def __init__(self, scale: float = 1.0):
        super(ScaledTanh, self).__init__()
        self.scale = scale
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.scale * self.tanh(x)

class BoundedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, umax):
        super().__init__()
        self.umax = float(umax)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            ScaledTanh(umax)
        )

    def forward(self, x):
        return self.model(x)

    def my_name(self) -> str:
        return "BM"
    def my_linear_ops(self) -> int:
        return 4
    def my_has_final_tanh(self):
        return True
    def my_u_scale(self):
        return self.umax

class BoundedModel3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, umax):
        super().__init__()
        self.umax = float(umax)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            ScaledTanh(umax)
        )

    def forward(self, x):
        return self.model(x)

    def my_name(self) -> str:
        return "BM"
    def my_linear_ops(self) -> int:
        return 3
    def my_has_final_tanh(self):
        return True
    def my_u_scale(self):
        return self.umax

class BoundedModelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def my_gt_type(self):
        # number of tensors in ground truth
        return 1

    def my_name(self) -> str:
        return "BML"

    def forward(self, y, y_true):
        return self.loss(y, y_true)

class BoundedModelLoss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def my_gt_type(self):
        # number of tensors in ground truth
        return 1

    def my_name(self) -> str:
        return "BML2"

    def forward(self, y, y_true):
        return F.smooth_l1_loss(y, y_true) # Huber loss

class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, umax):
        super().__init__()
        self.umax = float(umax)

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mode_head = nn.Linear(hidden_dim, 3)   # [-, cont, +]
        self.cont_head = nn.Linear(hidden_dim, output_dim)   # continuous raw

        self.register_buffer("mode_values", torch.tensor([-1.0, 0.0, 1.0]).view(1, 3))

    def my_name(self) -> str:
        return "HM"
    def my_linear_ops(self) -> int:
        return 4
    def my_has_final_tanh(self):
        return True
    def my_u_scale(self):
        return self.umax

    def forward(self, x, temperature=1.0) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        h = self.backbone(x)
        logits: Tensor = self.mode_head(h)                           # [N,3]
        p = F.softmax(logits / temperature, dim=-1)          # [N,3]

        u_cont = self.umax * torch.tanh(self.cont_head(h))   # [N,1]

        # mixture: -umax, u_cont, +umax
        u_mix = (p[:, 0:1] * (-self.umax) +
                 p[:, 1:2] * u_cont +
                 p[:, 2:3] * (+self.umax))

        return u_mix, logits, u_cont, p

def make_hybrid_mode_labels(yt: Tensor, u_max: float, u_thresh: float) -> Tensor:
    """
    y: [N,1] or [N]
    returns: [N] LongTensor with values {0,1,2}
    """
    y = yt.view(-1)  # flatten safely

    y_labels = torch.full_like(y, fill_value=1, dtype=torch.long)  # default: cont
    y_labels[y <= -u_thresh * u_max] = 0
    y_labels[y >=  u_thresh * u_max] = 2
    return y_labels

class HybridModelLoss1(nn.Module):
    def __init__(self):
        super().__init__()

    def my_gt_type(self):
        # number of tensors in ground truth
        return 1

    def my_name(self) -> str:
        return "HML1"

    def forward(self, yt: Tuple[Tensor, ...], y_true):
        y, *_ = yt
        return F.smooth_l1_loss(y, y_true) # Huber loss


class HybridModelLoss2(nn.Module):
    def __init__(self, ce_w=1.0, reg_w=1.0):
        super().__init__()
        self.ce_w = ce_w
        self.reg_w = reg_w

    def my_gt_type(self):
        # number of tensors in ground truth
        return 2

    def my_name(self) -> str:
        return "HML2"

    def forward(self, yt: Tuple[Tensor, ...], y_true_t: Tuple[Tensor, ...]):
        y, logits, *_ = yt
        y_true, y_mode_true = y_true_t
        loss_mode = F.cross_entropy(logits, y_mode_true)
        loss_reg = F.smooth_l1_loss(y, y_true)  # Huber
        return self.ce_w * loss_mode + self.reg_w * loss_reg

