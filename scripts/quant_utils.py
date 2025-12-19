from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
from typing import Dict, List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import re
import brevitas.nn as qnn

@dataclass
class QuantLayerInfo:
    name: str
    cls: str
    w_bits: Optional[int]
    a_bits: Optional[int]
    w_scale: Optional[torch.Tensor]   # float scale (LSB size)
    a_scale: Optional[torch.Tensor]
    int_weight: Optional[torch.Tensor]

def _get_attr(obj: Any, names: list[str]) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None

@torch.no_grad()
def collect_brevitas_quant_info(model: nn.Module) -> list[QuantLayerInfo]:
    infos: list[QuantLayerInfo] = []
    for name, m in model.named_modules():
        # Heuristic: QuantLinear / QuantConv / QuantReLU / QuantIdentity etc.
        is_brevitasish = m.__class__.__module__.startswith("brevitas")
        if not is_brevitasish:
            continue

        w_bits = _get_attr(m, ["weight_bit_width", "bit_width"])
        a_bits = _get_attr(m, ["bit_width"])

        # Weight scale (often property or tensor)
        w_scale = _get_attr(m, ["quant_weight_scale", "weight_scale"])
        if callable(w_scale):
            try:
                w_scale = w_scale()
            except TypeError:
                pass

        # Integer weight (often property)
        int_w = _get_attr(m, ["int_weight", "weight_int"])
        if callable(int_w):
            try:
                int_w = int_w()
            except TypeError:
                pass

        # Activation scale (often a method)
        a_scale = _get_attr(m, ["quant_act_scale", "act_scale", "quant_input_scale"])
        if callable(a_scale):
            try:
                a_scale = a_scale()
            except TypeError:
                pass

        # Keep only modules where we found something meaningful
        if w_scale is None and a_scale is None and int_w is None:
            continue

        infos.append(QuantLayerInfo(
            name=name,
            cls=m.__class__.__name__,
            w_bits=int(w_bits) if isinstance(w_bits, (int, float)) else None,
            a_bits=int(a_bits) if isinstance(a_bits, (int, float)) else None,
            w_scale=w_scale.detach().cpu() if isinstance(w_scale, torch.Tensor) else None,
            a_scale=a_scale.detach().cpu() if isinstance(a_scale, torch.Tensor) else None,
            int_weight=int_w.detach().cpu() if isinstance(int_w, torch.Tensor) else None,
        ))
    return infos

def dump_state_keys_for_module(model: nn.Module, module_name: str) -> List[Tuple[str, torch.Tensor]]:
    """
    Return all state_dict entries whose key starts with module_name + "."
    Example: module_name="net.0" (a QuantLinear inside Sequential).
    """
    sd = model.state_dict()
    prefix = module_name + "."
    out = [(k, v) for k, v in sd.items() if k.startswith(prefix)]
    return out

def find_scale_like_keys(entries: List[Tuple[str, torch.Tensor]]) -> List[Tuple[str, torch.Tensor]]:
    pat = re.compile(r"(scale|scal|alpha|threshold|stats|zero_point|zp)", re.IGNORECASE)
    return [(k, v) for (k, v) in entries if pat.search(k)]

def list_quantlinear_names(model: nn.Module) -> List[str]:
    names = []
    for n, m in model.named_modules():
        if m.__class__.__name__ == "QuantLinear":  # robust enough
            names.append(n)
    return names

def suggest_frac_bits_from_scale(scale: float) -> int:
    # nearest power-of-two: scale ≈ 2^-n  => n ≈ -log2(scale)
    return int(round(-math.log2(scale)))

def qmn_from_bits(total_bits: int, frac_bits: int, signed: bool = True) -> tuple[int, int]:
    # Qm.n: m integer bits *excluding sign* is common in DSP notation,
    # but conventions vary. Here: signed => 1 sign bit + m integer bits + n frac bits.
    n = frac_bits
    if signed:
        m = total_bits - 1 - n
    else:
        m = total_bits - n
    return (m, n)

@torch.no_grad()
def inspect_weight_quant(model, layer_name: str, w_bits: int, narrow_range: bool = True):
    m = dict(model.named_modules())[layer_name]
    #assert isinstance(m, qnn.QuantLinear)

    # float weights
    w = m.weight.detach().cpu().float()
    amax = w.abs().max().item()

    qmax = (2 ** (w_bits - 1) - 1)  # signed symmetric int
    s_eff = amax / qmax if qmax > 0 else float("nan")

    # stored parameter in state_dict
    sd = model.state_dict()
    key = layer_name + ".weight_quant.tensor_quant.scaling_impl.value"
    stored = float(sd[key].detach().cpu().item())

    return {"amax": amax, "qmax": qmax, "s_eff": s_eff, "stored_value": stored}

def dump_scale_keys_for_module(model, module_name: str):
    sd = model.state_dict()
    prefix = module_name + "."
    pat = re.compile(r"(scaling_impl\.value|scale|threshold|amax)", re.IGNORECASE)
    items = [(k, v) for k, v in sd.items() if k.startswith(prefix) and pat.search(k)]
    for k, v in items:
        t = v.detach().cpu().float()
        print(f"  {k:70s} shape={tuple(v.shape)}  val/mean={t.mean().item():.6g}")

def list_module_names(model, cls_name: str):
    return [n for n, m in model.named_modules() if m.__class__.__name__ == cls_name]


import math
import json
import torch
import brevitas.nn as qnn

def _shift_from_T(T: float, qmax: int) -> int:
    s = T / qmax
    return int(round(-math.log2(s)))

def _qmax_signed(bits: int) -> int:
    return 2 ** (bits - 1) - 1

def _qmax_unsigned(bits: int) -> int:
    return 2 ** bits - 1

def _as_int(v):
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return int(v.item())
    return None

@torch.no_grad()
def dump_quant_table_v2(
    model,
    w_bits: int,
    a_bits: int,
    o_bits: int,
    relu_unsigned: bool = True,
) -> list[dict]:
    sd = model.state_dict()

    qmax_w = _qmax_signed(w_bits)
    qmax_relu = _qmax_unsigned(a_bits) if relu_unsigned else _qmax_signed(a_bits)
    qmax_out = _qmax_signed(o_bits)

    rows = []

    # QuantLinear: weight threshold + acc bits
    for name, m in model.named_modules():
        if isinstance(m, qnn.QuantLinear):
            kT = f"{name}.weight_quant.tensor_quant.scaling_impl.value"
            if kT not in sd:
                continue
            T = float(sd[kT].item())
            acc_bits = _as_int(getattr(m, "accumulator_bit_width", None))

            rows.append({
                "name": name,
                "type": "QuantLinear",
                "w_bits": w_bits,
                "acc_bits": acc_bits,
                "T_weight": T,
                "qmax_w": qmax_w,
                "s_weight": T / qmax_w,
                "shift_w": _shift_from_T(T, qmax_w),
            })

    # QuantReLU: activation threshold (post-ReLU)
    for name, m in model.named_modules():
        if isinstance(m, qnn.QuantReLU):
            kT = f"{name}.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"
            if kT not in sd:
                continue
            T = float(sd[kT].item())
            rows.append({
                "name": name,
                "type": "QuantReLU",
                "a_bits": a_bits,
                "signed": not relu_unsigned,
                "T_act": T,
                "qmax_a": qmax_relu,
                "s_act": T / qmax_relu,
                "shift_a": _shift_from_T(T, qmax_relu),
            })

    # QuantIdentity: output threshold (after tanh)
    for name, m in model.named_modules():
        if isinstance(m, qnn.QuantIdentity):
            kT = f"{name}.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"
            if kT not in sd:
                continue
            T = float(sd[kT].item())
            rows.append({
                "name": name,
                "type": "QuantIdentity(out)",
                "o_bits": o_bits,
                "signed": True,
                "T_out": T,
                "qmax_o": qmax_out,
                "s_out": T / qmax_out,
                "shift_o": _shift_from_T(T, qmax_out),
            })

    # Sort by net.<idx> if present
    import re
    def keyfn(r):
        m = re.search(r"net\.(\d+)", r["name"])
        return int(m.group(1)) if m else 10**9
    rows.sort(key=keyfn)
    return rows


