import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse

from model import FAGCN
from utils import preprocess_data

def forward_with_layers(model, features):
    h = F.dropout(features, p=model.dropout, training=False)
    h = torch.relu(model.t1(h))
    h = F.dropout(h, p=model.dropout, training=False)
    raw = h
    all_layers = [h]
    for layer in model.layers:
        h = layer(h)
        h = model.eps * raw + h
        all_layers.append(h)
    return all_layers

def compute_jacobian_norms(model, features, subset_size):
    features = features.clone().detach().requires_grad_(True)
    outputs = forward_with_layers(model, features)
    subset = torch.randperm(features.size(0))[:subset_size].to(features.device)
    norms_per_layer = []

    for layer_output in outputs:
        norms = []
        for idx in subset:
            output = layer_output[idx].sum()
            grad = torch.autograd.grad(output, features, retain_graph=True)[0]
            if grad is not None:
                norms.append(grad[idx].norm().item())
        norms_per_layer.append(np.mean(norms))
    norms_per_layer = np.array(norms_per_layer)
    norms_per_layer = norms_per_layer / norms_per_layer.max()
    return norms_per_layer

def visualize_combined(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Cornell data once
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

    depth_results = {}

    for depth in args.depths:
        print(f"\n🔍 Processing depth {depth}...")
        model_path = os.path.join(args.model_dir, f"best_model_depth{depth}.pth")
        if not os.path.exists(model_path):
            print(f"Missing: {model_path}")
            continue

        model = FAGCN(g, features.shape[1], args.hidden, nclass, args.dropout, args.eps, depth).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        jacobian_norms = compute_jacobian_norms(model, features, args.subset_size)
        depth_results[depth] = jacobian_norms
        print(f"✅ Depth {depth} → Norms: {jacobian_norms}")

        # Save per-depth norms (optional)
        if args.save_npy:
            np.save(os.path.join(args.result_file_path, f"jacobian_norms_depth{depth}.npy"), jacobian_norms)

    # Plot
    plt.figure(figsize=(8, 6))
    for depth in sorted(depth_results.keys()):
        norms = depth_results[depth]
        plt.plot(range(len(norms)), norms, marker='o', label=f'Depth={depth}')
    plt.yscale("log")
    plt.xlabel("Layer Index")
    plt.ylabel("Avg Jacobian Norm (log scale)")
    plt.title(f"Jacobian Norm vs. GNN Depth ({args.dataset})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(args.result_file_path, f"jacobian_combined_{args.dataset}.png")
    plt.savefig(save_path)
    print(f"\nCombined plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cornell')
    parser.add_argument('--dataset_file_path', type=str, required=True)
    parser.add_argument('--result_file_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--depths', type=int, nargs='+', required=True,
                        help='List of depths to process, e.g. 2 4 6 8')
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument('--subset_size', type=int, default=100)
    parser.add_argument('--save_npy', action='store_true', help='Optionally save jacobian norms as .npy files')
    args = parser.parse_args()

    visualize_combined(args)
