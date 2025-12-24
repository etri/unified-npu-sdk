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

import torch
from torchvision.models import resnet50, ResNet50_Weights


weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights).eval()


if __name__ == "__main__":

    cfg = BuildConfig(
        backend="rbln",
        model_or_path=model,            # torch.nn.Module 직접 전달
        out_dir="build",
	model_name="resnet50",
	precision="fp32",
    	input_name="input",
    	min_input_shape=(1,3,224,224),
    	opt_input_shape=(1,3,224,224),
    	max_input_shape=(1,3,224,224),
    )

    result = build_unified(cfg)
    print("✅", result.compiled_model_path)  # build/resnet50.rbln

