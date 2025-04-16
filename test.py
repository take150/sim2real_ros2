import torch
import torchvision.models as models
import time
from torch2trt import torch2trt

# デバイス確認（CUDA推奨）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 学習済みResNet-18モデルを読み込み（evalモード）
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()

# ダミー入力（1枚の画像、3チャネル、224x224）
input_tensor = torch.randn(1, 3, 224, 224).to(device)

# --------------------------
# 通常のPyTorchによる推論時間
# --------------------------
with torch.no_grad():
    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)

    # 計測
    torch_start = time.time()
    for _ in range(1000):
        _ = model(input_tensor)
    torch_end = time.time()

pytorch_time = (torch_end - torch_start) / 100
print(f"[PyTorch] 平均推論時間: {pytorch_time:.6f} 秒")

# --------------------------
# torch2trtで変換したモデルによる推論時間
# --------------------------
# FP32で変換
model_trt = torch2trt(model, [input_tensor], fp16_mode=False)

with torch.no_grad():
    for _ in range(10):
        _ = model_trt(input_tensor)

    trt_start = time.time()
    for _ in range(1000):
        _ = model_trt(input_tensor)
    trt_end = time.time()

trt_time = (trt_end - trt_start) / 100
print(f"[torch2trt] 平均推論時間: {trt_time:.6f} 秒")

# --------------------------
# スピードアップ率
# --------------------------
speedup = pytorch_time / trt_time
print(f"🚀 Speedup: {speedup:.2f}x")
