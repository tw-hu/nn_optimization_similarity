import torch

from payload.models.ConvClassifier import build_model

model = build_model()
pt_file = torch.load("experiments/devmode_run_optimizer_adam_atan2_seed10/final.pt")
print(pt_file.keys())
print(pt_file['model_state'].keys())
print(pt_file['metrics'])

b1_out_layer = "_enc.block1.conv1_2"
b2_out_layer = "_enc.block2.conv2_3"
b3_out_layer = "_enc.block3.conv3_3"
b4_out_layer = "_enc.block4.conv4_3"