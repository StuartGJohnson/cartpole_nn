import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType

from model_factory import ScaledTanh, BoundedModel
from dataclasses import dataclass

@dataclass
class BitSpec:
    """
    name
    w_bits = 16   weight bit-width
    a_bits = 16   activation bit-width
    acc_bits = 24 accumulator bit-width
    o_bits = 16 Tanh output bits
    """
    name: str
    w_bits: int
    a_bits: int
    acc_bits: int
    o_bits: int


def quant_layer(input_dim, hidden_dim, w_bits, acc_bits, a_bits) -> nn.Module:
    return nn.Sequential(
        qnn.QuantLinear(
            in_features=input_dim,
            out_features=hidden_dim,
            bias=True,
            weight_bit_width=w_bits,
            accumulator_bit_width=acc_bits,
            weight_quant_type=QuantType.INT,  # integer weights
            weight_scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS,  # scale is a learned param
            weight_narrow_range=True,  # typical for signed int
        ),
        qnn.QuantReLU(
            bit_width=a_bits,
            quant_type=QuantType.INT,
            scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS,
            narrow_range=False,
        ),
    )

class QuantBoundedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, u_max, bs: BitSpec):
        super().__init__()
        self.u_max = u_max
        self.bitspec = bs

        self.net = nn.Sequential(
            # Layer 1
            qnn.QuantLinear(
                in_features=input_dim,
                out_features=hidden_dim,
                bias=True,
                weight_bit_width=bs.w_bits,
                accumulator_bit_width=bs.acc_bits,
                weight_quant_type=QuantType.INT,  # integer weights
                weight_scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS,  # scale is a learned param
                weight_narrow_range=True,  # typical for signed int
            ),
            qnn.QuantReLU(
                bit_width=bs.a_bits,
                quant_type=QuantType.INT,
                scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS,
                narrow_range=False,
            ),

            # Layer 2
            qnn.QuantLinear(
                in_features=hidden_dim,
                out_features=hidden_dim,
                bias=True,
                weight_bit_width=bs.w_bits,
                accumulator_bit_width=bs.acc_bits,
                weight_quant_type=QuantType.INT,  # integer weights
                weight_scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS,  # scale is a learned param
                weight_narrow_range=True,  # typical for signed int
            ),
            qnn.QuantReLU(
                bit_width=bs.a_bits,
                quant_type=QuantType.INT,
                scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS,
                narrow_range=False,
            ),

            # Layer 3
            qnn.QuantLinear(
                in_features=hidden_dim,
                out_features=hidden_dim,
                bias=True,
                weight_bit_width=bs.w_bits,
                accumulator_bit_width=bs.acc_bits,
                weight_quant_type=QuantType.INT,  # integer weights
                weight_scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS,  # scale is a learned param
                weight_narrow_range=True,  # typical for signed int
            ),
            qnn.QuantReLU(
                bit_width=bs.a_bits,
                quant_type=QuantType.INT,
                scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS,
                narrow_range=False,
            ),

            # Layer 4
            qnn.QuantLinear(
                in_features=hidden_dim,
                out_features=output_dim,
                bias=True,
                weight_bit_width=bs.w_bits,
                accumulator_bit_width=bs.acc_bits,
                weight_quant_type=QuantType.INT,  # integer weights
                weight_scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS,  # scale is a learned param
                weight_narrow_range=True,  # typical for signed int
            ),
            ScaledTanh(u_max),
            qnn.QuantIdentity(
                bit_width=bs.o_bits,
                quant_type=QuantType.INT,
                scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS
            )
        )

    def forward(self, x):
        return self.net(x)

    def my_name(self) -> str:
        return "QBM"
    def my_linear_ops(self) -> int:
        return 4
    def my_has_final_tanh(self):
        return True
    def my_u_scale(self):
        return self.u_max
    def my_bitspec(self):
        return self.bitspec.name

    def from_bounded_model(self, bm: BoundedModel):
        bm_linears =  [m for m in bm.model if isinstance(m, nn.Linear)]
        qbm_linears = [m for m in self.net if isinstance(m, qnn.QuantLinear)]
        assert len(bm_linears) == len(qbm_linears)
        with torch.no_grad():
            for f, qq in zip(bm_linears, qbm_linears):
                qq.weight.copy_(f.weight)
                if f.bias is not None and qq.bias is not None:
                    qq.bias.copy_(f.bias)
