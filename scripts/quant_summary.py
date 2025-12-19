from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from typing import Optional, Any

import torch
import torch.nn as nn
import brevitas.nn as qnn


# -----------------------------
# Fixed-point proposal objects
# -----------------------------
@dataclass
class QuantScalar:
    name: str
    T: float             # learned threshold/amax
    bits: int
    signed: bool
    qmax: int
    scale: float         # s = T/qmax
    shift: int           # n ~ round(-log2(scale))
    pow2_scale: float    # 2^-shift
    T_pow2: float        # qmax * 2^-shift

@dataclass
class LayerFPProposal:
    layer_name: str
    kind: str                    # e.g. Dense/ReLU/OutputQuant
    in_features: Optional[int]
    out_features: Optional[int]
    fan_in: Optional[int]

    # quantization choices
    w: Optional[QuantScalar]
    a_in: Optional[QuantScalar]
    a_out: Optional[QuantScalar]

    # MAC / accumulator bookkeeping (for Dense)
    product_shift: Optional[int]
    acc_bits: Optional[int]
    acc_bits_min_worstcase: Optional[int]
    acc_note: Optional[str]


# -----------------------------
# Helpers
# -----------------------------
def qmax_signed(bits: int) -> int:
    return (1 << (bits - 1)) - 1

def qmax_unsigned(bits: int) -> int:
    return (1 << bits) - 1

def suggest_shift(scale: float) -> int:
    # nearest power-of-two: scale ~ 2^-n
    return int(round(-math.log2(scale)))

def make_quant_scalar(name: str, T: float, bits: int, signed: bool) -> QuantScalar:
    qmax = qmax_signed(bits) if signed else qmax_unsigned(bits)
    scale = T / qmax
    sh = suggest_shift(scale)
    pow2 = 2.0 ** (-sh)
    T_pow2 = qmax * pow2
    return QuantScalar(
        name=name, T=T, bits=bits, signed=signed, qmax=qmax,
        scale=scale, shift=sh, pow2_scale=pow2, T_pow2=T_pow2
    )

def as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return int(v.item())
    return None

def net_index(name: str) -> int:
    m = re.search(r"net\.(\d+)", name)
    return int(m.group(1)) if m else 10**9


# -----------------------------
# Extract learned thresholds T
# (based on your observed keys)
# -----------------------------
def get_T_weight(sd: dict[str, torch.Tensor], module_name: str) -> Optional[float]:
    k = f"{module_name}.weight_quant.tensor_quant.scaling_impl.value"
    if k in sd:
        return float(sd[k].detach().cpu().item())
    return None

def get_T_act(sd: dict[str, torch.Tensor], module_name: str) -> Optional[float]:
    k = f"{module_name}.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"
    if k in sd:
        return float(sd[k].detach().cpu().item())
    return None


# -----------------------------
# Main: build proposal
# -----------------------------
@torch.no_grad()
def build_fp_proposal(
    model: nn.Module,
    *,
    w_bits: int,
    a_bits: int,
    o_bits: int,
    relu_unsigned: bool = True,
) -> list[LayerFPProposal]:
    """
    Builds a coherent FP plan using:
      - weight thresholds from QuantLinear
      - activation thresholds from QuantReLU + QuantIdentity
    Assumes:
      - weights signed
      - ReLU activations unsigned if relu_unsigned=True else signed
      - output (tanh) quant signed
    """
    sd = model.state_dict()

    # Gather modules in execution-ish order by net.<idx> if present
    mods = [(n, m) for n, m in model.named_modules()
            if isinstance(m, (qnn.QuantLinear, qnn.QuantReLU, qnn.QuantIdentity))]
    mods.sort(key=lambda x: net_index(x[0]))

    proposals: list[LayerFPProposal] = []

    # We'll track activation "out" from previous quant activation node, as the
    # implied "a_in" for the next QuantLinear. If you later add an explicit input QuantIdentity,
    # this becomes explicit.
    prev_act: Optional[QuantScalar] = None

    for name, m in mods:
        if isinstance(m, qnn.QuantReLU):
            T = get_T_act(sd, name)
            if T is None:
                continue
            act = make_quant_scalar(
                name=name + ".act",
                T=T,
                bits=a_bits,
                signed=not relu_unsigned,  # unsigned if relu_unsigned else signed
            )
            # update prev_act for subsequent linear
            prev_act = act

            proposals.append(LayerFPProposal(
                layer_name=name,
                kind="QuantReLU",
                in_features=None, out_features=None, fan_in=None,
                w=None, a_in=None, a_out=act,
                product_shift=None,
                acc_bits=None, acc_bits_min_worstcase=None, acc_note=None
            ))

        elif isinstance(m, qnn.QuantIdentity):
            T = get_T_act(sd, name)
            if T is None:
                continue
            out_act = make_quant_scalar(
                name=name + ".act",
                T=T,
                bits=o_bits,
                signed=True,  # tanh output is signed
            )
            # treat as "out activation"
            prev_act = out_act

            proposals.append(LayerFPProposal(
                layer_name=name,
                kind="QuantIdentity(OutputQuant)",
                in_features=None, out_features=None, fan_in=None,
                w=None, a_in=None, a_out=out_act,
                product_shift=None,
                acc_bits=None, acc_bits_min_worstcase=None, acc_note=None
            ))

        elif isinstance(m, qnn.QuantLinear):
            T_w = get_T_weight(sd, name)
            if T_w is None:
                continue

            wq = make_quant_scalar(
                name=name + ".weight",
                T=T_w,
                bits=w_bits,
                signed=True
            )

            # infer layer dims
            in_f = getattr(m, "in_features", None)
            out_f = getattr(m, "out_features", None)
            fan_in = int(in_f) if isinstance(in_f, int) else None

            # acc width (config)
            acc_bits = as_int(getattr(m, "accumulator_bit_width", None))

            # For the input activation to this linear:
            # - if we have prev_act (from previous QuantReLU), use it
            # - otherwise leave None (means: you should define/assume input quant)
            a_in = prev_act

            # Worst-case accumulator estimate (VERY conservative):
            # product bits ~= w_bits + a_bits (+1 for sign)
            # sum N terms => +ceil(log2(N))
            acc_min = None
            note = None
            prod_shift = None

            if fan_in is not None and a_in is not None:
                prod_shift = wq.shift + a_in.shift

                # Worst-case bits for sum of N max-magnitude products in integer domain:
                # max product magnitude ~ qmax_w * qmax_a, sum N => *N
                # bits ~= log2(qmax_w*qmax_a*N)
                qmax_w = wq.qmax
                qmax_a = a_in.qmax
                worst_mag = qmax_w * qmax_a * fan_in
                acc_min = int(math.ceil(math.log2(worst_mag + 1))) + 1  # +1 margin
                if acc_bits is not None and acc_bits < acc_min:
                    note = f"acc_bits={acc_bits} < conservative worst-case {acc_min} (may still be OK empirically)"
                elif acc_bits is not None:
                    note = "acc_bits meets conservative worst-case bound"
                else:
                    note = "acc_bits not found on module (config metadata)"

            proposals.append(LayerFPProposal(
                layer_name=name,
                kind="QuantLinear(Dense)",
                in_features=int(in_f) if isinstance(in_f, int) else None,
                out_features=int(out_f) if isinstance(out_f, int) else None,
                fan_in=fan_in,
                w=wq,
                a_in=a_in,
                a_out=None,
                product_shift=prod_shift,
                acc_bits=acc_bits,
                acc_bits_min_worstcase=acc_min,
                acc_note=note
            ))

    return proposals


# -----------------------------
# Pretty-print summary
# -----------------------------
def print_fp_proposal(proposals: list[LayerFPProposal]) -> None:
    def fmt_q(q: QuantScalar) -> str:
        sgn = "S" if q.signed else "U"
        return (f"{sgn}{q.bits}  T={q.T:.6g}  "
                f"s=T/qmax={q.scale:.3e}  shift≈{q.shift}  "
                f"(pow2 s={q.pow2_scale:.3e}, T_pow2={q.T_pow2:.6g})")

    print("\n=== Fixed-Point Proposal Summary ===\n")
    for p in proposals:
        print(f"[{p.layer_name}] {p.kind}")
        if p.kind.startswith("QuantLinear"):
            print(f"  dims: {p.in_features} -> {p.out_features} (fan_in={p.fan_in})")
            if p.w is not None:
                print(f"  weight: {fmt_q(p.w)}  (interpret as w_int * 2^-{p.w.shift})")
            if p.a_in is not None:
                print(f"  act_in: {fmt_q(p.a_in)}  (a_int * 2^-{p.a_in.shift})")
            else:
                print("  act_in: <unknown> (consider adding input QuantIdentity for explicit input quant)")
            if p.product_shift is not None:
                print(f"  MAC product shift (approx): {p.product_shift}  (= shift_w + shift_a_in)")
            if p.acc_bits is not None or p.acc_bits_min_worstcase is not None:
                print(f"  accumulator bits: {p.acc_bits}  | conservative min: {p.acc_bits_min_worstcase}")
                if p.acc_note:
                    print(f"  note: {p.acc_note}")
        else:
            if p.a_out is not None:
                print(f"  act_out: {fmt_q(p.a_out)}  (x_int * 2^-{p.a_out.shift})")
        print()

def save_fp_proposal_json(proposals: list[LayerFPProposal], path: str = "fp_proposal.json") -> None:
    with open(path, "w") as f:
        json.dump([asdict(p) for p in proposals], f, indent=2)

