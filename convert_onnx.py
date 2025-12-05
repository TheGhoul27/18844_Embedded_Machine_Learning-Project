# export_to_onnx.py
import torch
from infer import CNNBiLSTMAttentionRegressor  # same as in train/infer
import numpy as np

CKPT_PATH = "deep_bp_artifacts/ppg_cnn_bilstm_sbp_dbp.pt"
ONNX_PATH = "bp_model.onnx"

device = "cpu"

# 1) Recreate model and load weights
state = torch.load(CKPT_PATH, map_location=device)

model = CNNBiLSTMAttentionRegressor()
model.load_state_dict(state["model"])
model.eval()

# 2) Dummy input: (batch=1, channels=1, time=1000)
dummy = torch.randn(1, 1, 1000, dtype=torch.float32)

# 3) Export to ONNX
torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["x"],
    output_names=["y"],
    dynamic_axes={
        "x": {0: "batch", 2: "time"},   # allow variable batch, time
        "y": {0: "batch"},
    },
    opset_version=17,                   # recent ONNX opset
)
print("Saved ONNX model to", ONNX_PATH)

# 4) (optional) save y_mean/y_std separately as npy for the Pi
np.save("bp_y_mean.npy", state["y_mean"].astype(np.float32))
np.save("bp_y_std.npy",  state["y_std"].astype(np.float32))
print("Saved bp_y_mean.npy and bp_y_std.npy")
