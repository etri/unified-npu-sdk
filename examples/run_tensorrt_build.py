# import os
# import sys
# 
# # repo 루트: /workspace/unified-sdk
# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# SRC = os.path.join(ROOT, "src")
# 
# if SRC not in sys.path:
#     sys.path.insert(0, SRC)



# examples/test_tensorrt_build.py
from unified_sdk.types import BuildConfig
from unified_sdk.build.api import build_unified

if __name__ == "__main__":
    #cfg = BuildConfig(
    #   backend="tensorrt",
    #   model_or_path="models/hrnet_coco_w32_Nx256x192.onnx",
    #   out_dir="build_output",
    #   model_name="hrnet_coco_w32_Nx256x192",
    #   precision="fp16",
    #   input_name="input.1",
    #   min_input_shape=(1, 3, 256, 192),
    #   opt_input_shape=(4, 3, 256, 192),
    #   max_input_shape=(30, 3, 256, 192),
    #)
    cfg = BuildConfig(
        backend="tensorrt",
        model_or_path="models/yolov7.onnx",
        out_dir="build_output",
        model_name="yolov7",
        precision="fp32",
        input_name="images",
        min_input_shape=(1, 3, 640, 640),
        opt_input_shape=(1, 3, 640, 640),
        max_input_shape=(1, 3, 640, 640),
    )

    result = build_unified(cfg)
    print("Complete!", result.compiled_model_path)
