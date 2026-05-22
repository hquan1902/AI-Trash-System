# kiểm tra metadata của model_classification_best.pth
import torch

ckpt_path = "models/model_classification_best.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("Checkpoint path:", ckpt_path)
print("Keys:", ckpt.keys())
print("Epoch:", ckpt.get("epoch"))
print("Val acc:", ckpt.get("val_acc"))
print("Num classes:", ckpt.get("num_classes"))
print("Class map sample:", list(ckpt.get("class_to_idx", {}).items())[:7])