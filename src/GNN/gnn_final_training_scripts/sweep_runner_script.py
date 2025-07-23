import os
import wandb

sweep_id = "graph-diffusion-model-link-prediction/llmre4lp-src_GNN_gnn_final_training_scripts/6su39xdi"  # Replace with your real sweep ID

wandb.agent(sweep_id)  # count=1 for one run per sbatch
