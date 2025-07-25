import json
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
import numpy as np


def evaluate_predictions(pred_file, true_file, output_metrics_file):
    """Evaluate LLM predictions against ground truth and save metrics.
    
    Args:
        pred_file (str): Path to the prediction file (e.g., inference_results_lp_llm/test_all.json).
        true_file (str): Path to the ground truth file (e.g., llm_pred/prompt_json/Arxiv/test_all.json).
        output_metrics_file (str): Path to save the metrics output.
    """
    # Validate file existence
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    if not os.path.exists(true_file):
        raise FileNotFoundError(f"Ground truth file not found: {true_file}")

    # Load predictions and ground truth
    with open(pred_file, 'r') as f:
        data_res = json.load(f)
    with open(true_file, 'r') as g:
        data_true = json.load(g)
    score_dict ={}
    # Create dictionaries for mapping id to response (1 for Yes, 0 for No)
    pred_dict = {item["id"]: 1 if "Yes" in item["res"] else 0 for item in data_res}
    true_dict = {item["id"]: 1 if "Yes" in item["conversations"][1]["value"] else 0 for item in data_true}
    for item in data_res:
        score_dict[item["id"]] = float(item.get("score", 1.0 if "Yes" in item["res"] else 0.0))
    # Extract predictions and ground truth for common IDs
    common_ids = set(pred_dict.keys()) & set(true_dict.keys())
    if not common_ids:
        raise ValueError("No matching IDs found between prediction and ground truth files.")

    truth = [true_dict[id] for id in common_ids]
    pred = [pred_dict[id] for id in common_ids]
    scores = [score_dict[id] for id in common_ids]
    try:
        auc = roc_auc_score(truth, scores)
    except ValueError:
        auc = float("nan")  # Handle case with only one class in truth

    # Compute metrics
    accuracy = accuracy_score(truth, pred)
    f1 = f1_score(truth, pred)
    precision, recall, f1, support = precision_recall_fscore_support(truth, pred, average = "binary")
  
    # Print metrics
    print(f"LLM Link Prediction Performance (evaluated on {len(common_ids)} pairs):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
  

    # Save metrics to file
    os.makedirs(os.path.dirname(output_metrics_file), exist_ok=True)
    with open(output_metrics_file, 'w') as f:
        f.write(f"LLM Link Prediction Performance (evaluated on {len(common_ids)} pairs):\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")


def main():
    """Main function to evaluate LLM link prediction performance."""
    # File paths (adjust if your output file name differs)
    pred_file = "inference_results_lp_llm/Arxiv_1hop/test/preds.json"  # Path to your prediction file
    true_file = "../../llm_prompt_dataset/Arxiv/1hop_test.json"  # Path to ground truth
    metrics_file = "inference_results_lp_llm/Arxiv_1hop/test/eval_metrics.txt"  # Output metrics file

    # Step: Evaluate predictions
    print("Evaluating LLM performance...")
    evaluate_predictions(pred_file, true_file, metrics_file)
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()