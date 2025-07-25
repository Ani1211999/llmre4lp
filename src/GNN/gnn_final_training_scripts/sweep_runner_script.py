import os
import wandb
import argparse

parser = argparse.ArgumentParser(description="GNN Link Prediction from preprocessed NPZ file.")

parser.add_argument('--sweep_id', type=str, default='a1b2ce4w13',
                        help="sweep id for performing sweeps.")
args = parser.parse_args()
sweep_id = f"graph-diffusion-model-link-prediction/llmre4lp-src_GNN_gnn_final_training_scripts/{args.sweep_id}"  # Replace with your real sweep ID

wandb.agent(sweep_id)  # count=1 for one run per sbatch
