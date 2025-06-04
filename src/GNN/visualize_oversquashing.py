import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse

from utils import preprocess_data
from model import FAGCN

# -------- Argument Parser --------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cornell')
parser.add_argument('--dataset_file_path', type=str, required=True)
parser.add_argument('--result_file_path', type=str, required=True)
parser.add_argument('--hidden', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--eps', type=float, default=0.3)
parser.add_argument('--layer_num', type=int, default=2)
parser.add_argument('--model_path', type=str, default='llm4heg_model.pth')
parser.add_argument('--train_ratio', type=float, default=0.6)
parser.add_argument('--subset_size', type=int, default=100)
args = parser.parse_args()

# -------- Load Data --------
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

g, nclass, features, labels, train, val, test = preprocess_data(
    args.dataset,
    args.train_ratio,
    False,
    args.result_file_path,
    args.dataset_file_path
)

features = features.to(device)
g = g.to(device)
deg = g.in_degrees().float().clamp(min=1).to(device)
norm = torch.pow(deg, -0.5)
g.ndata['d'] = norm

# -------- Load Model --------
model = FAGCN(g, features.shape[1], args.hidden, nclass, args.dropout, args.eps, args.layer_num).to(device)
model.load_state_dict(torch.load(os.path.join(args.result_file_path, args.model_path)))
model.eval()

# -------- Patch Forward for Intermediates --------
# def forward_with_layers(model, features):
#     h = F.dropout(features, p=model.dropout, training=False)
#     h = torch.relu(model.t1(h))
#     h = F.dropout(h, p=model.dropout, training=False)
#     raw = h
#     all_layers = [h]

#     for i in range(model.layer_num):
#         h = model.layers[i](h, model.yes_weight, model.no_weight)
#         h = model.eps * raw + h
#         all_layers.append(h)

#     return all_layers
def forward_with_layers(model, features):
    h = F.dropout(features, p=model.dropout, training=False)
    h = torch.relu(model.t1(h))
    h = F.dropout(h, p=model.dropout, training=False)
    raw = h
    all_layers = [h]

    for i in range(model.layer_num):
        # Match the original model: pass weights from FAGCN
        h = model.layers[i](h)
        h = model.eps * raw + h
        all_layers.append(h)

    return all_layers

# -------- Jacobian Norm Computation --------
features.requires_grad_(True)
layer_outputs = forward_with_layers(model, features)

subset = torch.randperm(features.shape[0])[:args.subset_size].to(device)
jacobian_norms = []

for layer_idx, h_layer in enumerate(layer_outputs):
    norms = []
    for node_idx in subset:
        scalar_output = h_layer[node_idx].sum()
        grad = torch.autograd.grad(
            scalar_output,
            features,
            retain_graph=True,
            create_graph=False
        )[0]
        if grad is not None:
            node_grad = grad[node_idx]
            norms.append(node_grad.norm().item())

    avg_norm = np.mean(norms)
    jacobian_norms.append(avg_norm)
    print(f"Layer {layer_idx} - Avg Jacobian Norm: {avg_norm:.6f}")

# -------- Plot --------
plt.figure()
plt.plot(range(len(jacobian_norms)), jacobian_norms, marker='o', linestyle='--', color='red')
plt.title("Jacobian Norm vs. GNN Depth (Cornell)")
plt.xlabel("Layer Index (0 = Input to FA Layer)")
plt.ylabel("Average Jacobian Norm")
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(args.result_file_path, f"jacobian_cornell.png")
plt.savefig(save_path)
plt.show()
print(f"Jacobian norm plot saved to: {save_path}")
