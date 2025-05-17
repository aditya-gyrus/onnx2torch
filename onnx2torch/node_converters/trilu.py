__all__ = [
    'OnnxTrilu',
]

from typing import Optional

import torch 
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node


class OnnxTrilu(nn.Module, OnnxToTorchModule):
    def __init__(self,upper: int = 0):
        super().__init__()
        self.upper = upper  # 0: lower, 1: upper

    def forward(self,input_tensor:torch.Tensor, k: Optional[torch.Tensor] = None) -> torch.Tensor:
        diag = int(k.item()) if k is not None else 0
        if self.upper == 0:
            return torch.tril(input_tensor, diagonal=diag)
        else:
            return torch.triu(input_tensor, diagonal=diag)


@add_converter(operation_type="Trilu",version=14)
def _(node:OnnxNode, graph:OnnxGraph) -> OperationConverterResult:
    upper = node.attributes.get('upper',0)

    return OperationConverterResult(
        torch_module=OnnxTrilu(upper=upper),
        onnx_mapping=onnx_mapping_from_node(node = node),
    )