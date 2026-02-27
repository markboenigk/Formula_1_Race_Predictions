import os
from typing import Any, Callable


def export_xgboost_to_onnx(
    booster: Any,
    feature_name: str,
    output_path: str,
    target_opset: int = 12,
    converter: Callable[..., Any] | None = None,
    num_features: int = 1,
) -> None:
    if converter is None:
        from onnxmltools import convert_xgboost
        from skl2onnx.common.data_types import FloatTensorType

        converter = convert_xgboost
        initial_type = [(feature_name, FloatTensorType([None, num_features]))]
    else:
        initial_type = [(feature_name, None)]

    onnx_model = converter(booster, initial_types=initial_type, target_opset=target_opset)
    with open(os.fspath(output_path), "wb") as f:
        f.write(onnx_model.SerializeToString())
