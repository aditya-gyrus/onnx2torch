from typing import List


import numpy as np
import onnx
import pytest

from tests.utils.common import check_onnx_model
from tests.utils.common import make_model_from_nodes


def _test_trilu(  # pylint: disable=missing-function-docstring
    input_data: np.ndarray, upper: int, k: List[int], opset_version: int, **kwargs
)-> None:
    test_inputs = {'input_tensor': input_data, 'k': np.array(k, dtype=np.int64)}

    node = onnx.helper.make_node(
        op_type="Trilu",
        inputs=["input_tensor", "k"],
        outputs=["y"],
        upper=upper,
        **kwargs
    )

    model = make_model_from_nodes(
        nodes=node,
        initializers={},
        inputs_example=test_inputs,
        opset_version=opset_version,
    )

    check_onnx_model(
        model,
        test_inputs,
        atol_onnx_torch=1e-6,
        atol_torch_cpu_cuda=1e-6,
    )


@pytest.mark.parametrize("upper", [0, 1])  # 0 = lower, 1 = upper
@pytest.mark.parametrize("k", [-1, 0, 1])
@pytest.mark.parametrize("shape", [(3, 3), (4, 5), (2, 6), (6, 2)])
@pytest.mark.parametrize("opset_version", [14])
def test_trilu(
    upper, k, shape, opset_version
    ) -> None:
    input_data = np.random.randn(*shape).astype(np.float32)
    _test_trilu(input_data, upper, k, opset_version)
