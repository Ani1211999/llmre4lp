import numpy as np
import json
import os
import argparse
import random

# --- Prompt Templates --- (Keep this section exactly the same)
prompt_info = {
    'Arxiv': {
        'BT': "Background: Papers in the arXiv 2023 network. Each paper is described by its title and abstract.",
        'F': "Paper 1 description: ",
        'S': "Paper 2 description: ",
        'A': ". Answer template: \"Yes\" or \"No\" for training/validation/test, or a probability (0–1) for inference. Please think step by step.",
        'context': {
            1: "- These papers are directly connected (1 hop apart) in the citation network.",
            3: "- These papers are separated by two intermediate papers (3 hops apart).",
            'inference': "- Your task is to infer a potential citation relationship."
        },
        'task_instruction': {
            1: "Task: Predict whether a direct citation relationship exists between two papers based on their descriptions.",
            3: "Task: Predict whether a potential citation connection might exist through two intermediaries.",
            'inference': "Task: Predict whether a citation relationship might exist between two papers based on their descriptions."
        }
    },
    'Cora': {
        'BT': "Background: Papers in the Cora network. Each paper is described by its title and abstract.",
        'F': "Paper 1 description: ",
        'S': "Paper 2 description: ",
        'A': ". Answer template: \"Yes\" or \"No\" for training/validation/test, or a probability (0–1) for inference. Please think step by step.",
        'context': {
            1: "- These papers are directly connected (1 hop apart) in the citation network.",
            3: "- These papers are separated by two intermediate papers (3 hops apart).",
            'inference': "- Your task is to infer a potential citation relationship."
        },
        'task_instruction': {
            1: "Task: Predict whether a direct citation relationship exists between two papers based on their descriptions.",
            3: "Task: Predict whether a potential citation connection might exist through two intermediaries.",
            'inference': "Task: Predict whether a citation relationship might exist between two papers based on their descriptions."
        }
    }
}

def load_preprocessed_npz(npz_path):
    """Load the preprocessed .npz file"""
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}

def get_context_line(data_name, hop_distance, is_inference_mode):
    """Generates context line based on hop distance and mode"""
    if is_inference_mode:
        return prompt_info[data_name]['context']['inference']
    else:
        return prompt_info[data_name]['context'].get(hop_distance, "")

def get_task_instruction(data_name, hop_distance, is_inference_mode):
    """Generates task instruction line based on hop distance and mode"""
    if is_inference_mode:
        return prompt_info[data_name]['task_instruction']['inference']
    else:
        return prompt_info[data_name]['task_instruction'].get(hop_distance, "")

def generate_prompt_json(data_name, mode, save_dir, npz_path):
    # Load dataset
    data = load_preprocessed_npz(npz_path)
    node_texts = data['node_texts']

    # Define output paths - Updated to include hop3_val
    mode_paths = {
        "llm_train": "llm_train.json",
        "1hop_val": "1hop_val.json",
        "1hop_test": "1hop_test.json",
        "hop3_val": "hop3_val.json",  # New addition
        "hop3_eval": "hop3_eval.json",
    }

    prompt_list = []
    current_mode_save_path = ""

    # --- LLM Training Data (combining 1-hop and 3-hop) ---
    if mode == "llm_train":
        print(f"[INFO] Generating LLM training data for '{data_name}' (1-hop and 3-hop combined)...")
        current_mode_save_path = mode_paths[mode]

        # Initialize a list to hold all training samples with their hop distance
        training_samples = []

        # Add 1-hop positive examples
        hop1_pos_edges = data['train_edges']
        for i in range(hop1_pos_edges.shape[1]):
            u, v = hop1_pos_edges[0, i], hop1_pos_edges[1, i]
            training_samples.append({'edge': (u, v), 'label': 1, 'hop': 1})
        print(f"Added {hop1_pos_edges.shape[1]} 1-hop positive samples.")

        # Add 1-hop negative examples
        hop1_neg_edges = data.get('train_neg_edges', np.empty((2, 0)))
        for i in range(hop1_neg_edges.shape[1]):
            u, v = hop1_neg_edges[0, i], hop1_neg_edges[1, i]
            training_samples.append({'edge': (u, v), 'label': 0, 'hop': 1})
        print(f"Added {hop1_neg_edges.shape[1]} 1-hop negative samples.")

        # Add 3-hop positive examples
        hop3_pos_edges = data.get('hop3_train_edges', np.empty((2, 0)))
        for i in range(hop3_pos_edges.shape[1]):
            u, v = hop3_pos_edges[0, i], hop3_pos_edges[1, i]
            training_samples.append({'edge': (u, v), 'label': 1, 'hop': 3})
        print(f"Added {hop3_pos_edges.shape[1]} 3-hop positive samples.")

        # Add 3-hop negative examples
        hop3_neg_edges = data.get('hop3_train_neg_edges', np.empty((2, 0)))
        for i in range(hop3_neg_edges.shape[1]):
            u, v = hop3_neg_edges[0, i], hop3_neg_edges[1, i]
            training_samples.append({'edge': (u, v), 'label': 0, 'hop': 3})
        print(f"Added {hop3_neg_edges.shape[1]} 3-hop negative samples.")

        # Shuffle the combined list of samples
        random.shuffle(training_samples)
        print(f"Total samples for LLM training: {len(training_samples)}")

        # Generate prompts from the shuffled list
        for sample in training_samples:
            start_node, end_node = sample['edge']
            label = sample['label']
            hop_distance = sample['hop']
            gpt_response = "Yes" if label == 1 else "No"

            context_line = get_context_line(data_name, hop_distance, is_inference_mode=False)
            task_line = get_task_instruction(data_name, hop_distance, is_inference_mode=False)

            human_prompt_value = (
                prompt_info[data_name]['BT'] + "\n" +
                task_line + "\n" +
                context_line + "\n" +
                prompt_info[data_name]['F'] + node_texts[start_node] + "\n" +
                prompt_info[data_name]['S'] + node_texts[end_node] +
                prompt_info[data_name]['A']
            )

            prompt_list.append({
                "id": f"{start_node}_{end_node}",
                "conversations": [
                    {"from": "human", "value": human_prompt_value},
                    {"from": "gpt", "value": gpt_response}
                ],
                "ground_truth": gpt_response
            })

    # --- 1-hop Validation/Test ---
    elif mode in ["1hop_val", "1hop_test"]:
        print(f"[INFO] Generating 1-hop {mode.split('_')[1]} data for '{data_name}'...")
        if mode == "1hop_val":
            pos_edges = data['val_edges']
            neg_edges = data.get('val_neg_edges', np.empty((2, 0)))
        elif mode == "1hop_test":
            pos_edges = data['test_edges']
            neg_edges = data.get('test_neg_edges', np.empty((2, 0)))

        current_mode_save_path = mode_paths[mode]
        hop_distance = 1

        # Positive examples
        for i in range(pos_edges.shape[1]):
            start_node, end_node = pos_edges[0, i], pos_edges[1, i]
            context_line = get_context_line(data_name, hop_distance, is_inference_mode=False)
            task_line = get_task_instruction(data_name, hop_distance, is_inference_mode=False)
            human_prompt_value = (
                prompt_info[data_name]['BT'] + "\n" +
                task_line + "\n" +
                context_line + "\n" +
                prompt_info[data_name]['F'] + node_texts[start_node] + "\n" +
                prompt_info[data_name]['S'] + node_texts[end_node] +
                prompt_info[data_name]['A']
            )
            prompt_list.append({
                "id": f"{start_node}_{end_node}",
                "conversations": [
                    {"from": "human", "value": human_prompt_value},
                    {"from": "gpt", "value": "Yes"}
                ],
                "ground_truth": "Yes"
            })

        # Negative examples
        for i in range(neg_edges.shape[1]):
            start_node, end_node = neg_edges[0, i], neg_edges[1, i]
            context_line = get_context_line(data_name, hop_distance, is_inference_mode=False)
            task_line = get_task_instruction(data_name, hop_distance, is_inference_mode=False)
            human_prompt_value = (
                prompt_info[data_name]['BT'] + "\n" +
                task_line + "\n" +
                context_line + "\n" +
                prompt_info[data_name]['F'] + node_texts[start_node] + "\n" +
                prompt_info[data_name]['S'] + node_texts[end_node] +
                prompt_info[data_name]['A']
            )
            prompt_list.append({
                "id": f"{start_node}_{end_node}",
                "conversations": [
                    {"from": "human", "value": human_prompt_value},
                    {"from": "gpt", "value": "No"}
                ],
                "ground_truth": "No"
            })

    # --- 3-hop Validation ---
    elif mode == "hop3_val":
        hop_distance = 3
        print(f"[INFO] Generating {hop_distance}-hop validation data for '{data_name}'...")
        current_mode_save_path = mode_paths[mode]

        pos_edges = data.get('hop3_val_edges', np.empty((2, 0)))
        neg_edges = data.get('hop3_val_neg_edges', np.empty((2, 0)))

        task_line = get_task_instruction(data_name, hop_distance, is_inference_mode=False)
        context_line_base = get_context_line(data_name, hop_distance, is_inference_mode=False)

        # Positive edges
        for i in range(pos_edges.shape[1]):
            start_node, end_node = pos_edges[0, i], pos_edges[1, i]
            human_prompt_value = (
                prompt_info[data_name]['BT'] + "\n" +
                task_line + "\n" +
                context_line_base + "\n" +
                prompt_info[data_name]['F'] + node_texts[start_node] + "\n" +
                prompt_info[data_name]['S'] + node_texts[end_node] +
                prompt_info[data_name]['A']
            )
            prompt_list.append({
                "id": f"{start_node}_{end_node}",
                "conversations": [
                    {"from": "human", "value": human_prompt_value},
                    {"from": "gpt", "value": "Please provide a probability (0–1)."}
                ],
                "ground_truth": "Yes"
            })

        # Negative edges
        for i in range(neg_edges.shape[1]):
            start_node, end_node = neg_edges[0, i], neg_edges[1, i]
            human_prompt_value = (
                prompt_info[data_name]['BT'] + "\n" +
                task_line + "\n" +
                context_line_base + "\n" +
                prompt_info[data_name]['F'] + node_texts[start_node] + "\n" +
                prompt_info[data_name]['S'] + node_texts[end_node] +
                prompt_info[data_name]['A']
            )
            prompt_list.append({
                "id": f"{start_node}_{end_node}",
                "conversations": [
                    {"from": "human", "value": human_prompt_value},
                    {"from": "gpt", "value": "Please provide a probability (0–1)."}
                ],
                "ground_truth": "No"
            })

    # --- 3-hop Evaluation ---
    elif mode == "hop3_eval":
        hop_distance = 3
        print(f"[INFO] Generating {hop_distance}-hop evaluation data for '{data_name}'...")
        current_mode_save_path = mode_paths[mode]

        pos_edges = data.get('hop3_test_edges', np.empty((2, 0)))
        neg_edges = data.get('hop3_test_neg_edges', np.empty((2, 0)))

        task_line = get_task_instruction(data_name, hop_distance, is_inference_mode=False)
        context_line_base = get_context_line(data_name, hop_distance, is_inference_mode=False)

        # Positive edges
        for i in range(pos_edges.shape[1]):
            start_node, end_node = pos_edges[0, i], pos_edges[1, i]
            human_prompt_value = (
                prompt_info[data_name]['BT'] + "\n" +
                task_line + "\n" +
                context_line_base + "\n" +
                prompt_info[data_name]['F'] + node_texts[start_node] + "\n" +
                prompt_info[data_name]['S'] + node_texts[end_node] +
                prompt_info[data_name]['A']
            )
            prompt_list.append({
                "id": f"{start_node}_{end_node}",
                "conversations": [
                    {"from": "human", "value": human_prompt_value},
                    {"from": "gpt", "value": "Please provide a probability (0–1)."}
                ],
                "ground_truth": "Yes"
            })

        # Negative edges
        for i in range(neg_edges.shape[1]):
            start_node, end_node = neg_edges[0, i], neg_edges[1, i]
            human_prompt_value = (
                prompt_info[data_name]['BT'] + "\n" +
                task_line + "\n" +
                context_line_base + "\n" +
                prompt_info[data_name]['F'] + node_texts[start_node] + "\n" +
                prompt_info[data_name]['S'] + node_texts[end_node] +
                prompt_info[data_name]['A']
            )
            prompt_list.append({
                "id": f"{start_node}_{end_node}",
                "conversations": [
                    {"from": "human", "value": human_prompt_value},
                    {"from": "gpt", "value": "Please provide a probability (0–1)."}
                ],
                "ground_truth": "No"
            })

    else:
        raise ValueError(f"Invalid mode: {mode}. Use llm_train, 1hop_val, 1hop_test, hop3_val, hop3_eval")

    # --- Save JSON File ---
    print("[INFO] Template length:", len(prompt_list))
    if prompt_list:
        print("[INFO] First template:", prompt_list[0])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, current_mode_save_path)
    print("[INFO] Saving to:", save_path)

    with open(save_path, "w") as fout:
        json.dump(prompt_list, fout, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="Arxiv")
    parser.add_argument("--mode", type=str, required=True,
                       help="Mode: llm_train, 1hop_val, 1hop_test, hop3_val, hop3_eval")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--npz_path", type=str, default="dataset/arxiv_2023_v9.npz")
    args = parser.parse_args()

    generate_prompt_json(args.data_name, args.mode, args.save_dir, args.npz_path)