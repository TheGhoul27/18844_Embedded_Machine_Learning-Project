import torch
from infer import CNNBiLSTMAttentionRegressor

state = torch.load("deep_bp_artifacts/ppg_cnn_bilstm_sbp_dbp.pt", map_location="cpu")

model = CNNBiLSTMAttentionRegressor()
model.load_state_dict(state["model"])
model.eval()

# Example dummy input: 1 batch, 1 channel, 1000 samples (8 s at 125 Hz)
dummy = torch.randn(1, 1, 1000)
torch.jit.trace(model, dummy).save("bp_model_traced.pt")