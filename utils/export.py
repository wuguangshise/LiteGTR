"""ONNX export that works across torch versions.

Recent torch makes ``torch.onnx.export`` default to the dynamo-based exporter,
which needs the separate ``onnxscript`` package and fails without it:

    ModuleNotFoundError: No module named 'onnxscript'

We ask for the TorchScript exporter explicitly (``dynamo=False``): it handles
opset 13 and the static-shape export the deployment claim rests on (P1-8), and
needs nothing extra. Older torch has no ``dynamo`` keyword, so fall back.
"""
from __future__ import annotations

import torch


def onnx_export(module: torch.nn.Module, args, path: str, **kwargs) -> None:
    try:
        torch.onnx.export(module, args, path, dynamo=False, **kwargs)
    except TypeError as e:
        if "dynamo" not in str(e):
            raise
        torch.onnx.export(module, args, path, **kwargs)
