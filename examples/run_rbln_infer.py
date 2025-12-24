import timeit
import numpy as np
from unified_sdk.types import RuntimeConfig
from unified_sdk.runtime import create_runtime, infer, destroy_runtime


import urllib.request
import torch
from torchvision.io.image import read_image
from torchvision.models import ResNet50_Weights


weights = ResNet50_Weights.DEFAULT
urllib.request.urlretrieve(
    "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg",
    "input.jpg"
)
img = read_image("input.jpg")
batch = weights.transforms()(img).unsqueeze(0).numpy()  # numpy로 맞추면 tensor_type='np'


if __name__ == "__main__":

    cfg = RuntimeConfig(
	backend="rbln",
	engine_path="build/resnet50.rbln",
    	input_name="input",
    	output_name="output",              # RBLN은 보통 output_name이 강제되지 않을 수 있어(필드 유지용)
    	input_shape=(1,3,224,224),
    	extra={"tensor_type": "np"},        # 'np'|'pt' :contentReference[oaicite:10]{index=10}
    )
    rh = create_runtime(cfg)

    x = batch
    
    for _ in range(1): _ = infer(rh, x)

    iters = 50
    times = []
    for _ in range(iters):
        t0 = timeit.default_timer()
        y = infer(rh, x)
        t1 = timeit.default_timer()
        times.append((t1 - t0) * 1000)


    _, idx = torch.topk(torch.from_numpy(y), 1, dim=1) 
    pred_class = weights.meta['categories'][idx]
    print(pred_class)


    print(f"Avg latency: {np.mean(times):.3f} ms, shape={y.shape}")
    
    destroy_runtime(rh)

