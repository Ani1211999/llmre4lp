import argparse
import numpy as np
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True,
                        help="Path to results txt log.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    cur_depth = None
    cur_lr = None
    cur_wd = None
    cur_scores = []

    best_scores = defaultdict(list)  # {depth: [(mean_acc, std_acc, lr, wd)]}
    temp_scores = defaultdict(list)

    with open(args.result_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('LAYER_NUM'):
                cur_depth = int(line.split('=')[-1])
            elif line.startswith('LR'):
                cur_lr = float(line.split('=')[-1])
            elif line.startswith('WD'):
                cur_wd = float(line.split('=')[-1])
            elif line.startswith('Split'):
                split_id = int(line.split('=')[-1])
            elif line.startswith('0.'):
                res = float(line.split(' ')[-1])
                cur_scores.append(res)
                if split_id == 9:
                    mean_acc = np.mean(cur_scores)
                    std_acc = np.std(cur_scores)
                    temp_scores[cur_depth].append((mean_acc, std_acc, cur_lr, cur_wd))
                    cur_scores = []

    print("\n📊 Best Results per LAYER_NUM:\n")
    for depth in sorted(temp_scores):
        sorted_runs = sorted(temp_scores[depth], key=lambda x: x[0], reverse=True)
        best = sorted_runs[0]
        print(f"LAYER_NUM={depth} → {100*best[0]:.2f} ± {100*best[1]:.2f} | lr={best[2]} | wd={best[3]}")
