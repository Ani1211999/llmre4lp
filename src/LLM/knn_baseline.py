import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
import torch

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the test_all.json dataset
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Extract paper descriptions and labels from test_all.json
def extract_paper_data(dataset):
    paper_texts = {}  # Map paper ID to its text (title + abstract)
    pairs = []  # List of (paper1_id, paper2_id, label)
    extracted_pairs = []  # List to store all extracted pairs for verification
    skipped_pairs = 0
    for entry in dataset:
        # Extract paper IDs from the ID field
        paper1_id, paper2_id = entry['id'].split('_')
        
        # Extract label
        label = entry['conversations'][1]['value']
        label_binary = 1 if label == "Yes" else 0
        
        # Extract paper descriptions
        prompt = entry['conversations'][0]['value']
        # Use regex to extract Paper 1 and Paper 2 descriptions
        # paper1_match = re.search(r"Paper 1 description: Title: (.*?)\s+Abstract: (.*?)\s+Paper 2", prompt)
        # paper2_match = re.search(r"Paper 2 description: Title: (.*?)\s+Abstract: (.*?)\s+\.", prompt)
        # paper1_match = re.search(r"Paper 1.*Title: (.*?)(?:\s+Abstract: (.*?))?(?:\s+Paper 2|$)", prompt, re.DOTALL)
        # paper2_match = re.search(r"Paper 2.*Title: (.*?)(?:\s+Abstract: (.*?))?(?:\s+\.|$)", prompt, re.DOTALL)
        paper1_match = re.search(
            r"Paper 1 description: Title: (.*?)\s+Abstract: (.*?)(?=\s*Paper 2 description:)",
            prompt,
            re.DOTALL
        )
        # Match Paper 2 description
        paper2_match = re.search(
            r"Paper 2 description: Title: (.*?)\s+Abstract: (.*?)(?=\s*\.\s*Answer template:)",
            prompt,
            re.DOTALL
        )
        if not paper1_match or not paper2_match:
            print(f"Failed to parse prompt for ID {entry['id']}:")
            print(f"Prompt: {prompt}")
            skipped_pairs += 1
            continue
        
        paper1_title = paper1_match.group(1).strip()
        paper1_abstract = paper1_match.group(2).strip() if paper1_match.group(2) else ""
        paper2_title = paper2_match.group(1).strip()
        paper2_abstract = paper2_match.group(2).strip() if paper2_match.group(2) else ""
        # if not paper1_match or not paper2_match:
        #     continue
        
        # paper1_title, paper1_abstract = paper1_match.groups()
        # paper2_title, paper2_abstract = paper2_match.groups()
        
        # Combine title and abstract
        paper1_text = f"{paper1_title} {paper1_abstract}"
        paper2_text = f"{paper2_title} {paper2_abstract}"
        
        # Store paper texts
        paper_texts[paper1_id] = paper1_text
        paper_texts[paper2_id] = paper2_text
        pairs.append((paper1_id, paper2_id, label_binary))
        extracted_pairs.append({
            "id": entry['id'],
            "paper1_id": paper1_id,
            "paper1_text": paper1_text,
            "paper2_id": paper2_id,
            "paper2_text": paper2_text,
            "label": label,
            "label_binary": label_binary
        })
        
        
        # print(f"Paper {paper1_id}: {paper1_text}!!!!")
        # print(f"Paper {paper2_id}: {paper2_text}!!!!")
    with open("extracted_pairs.json", "w") as f:
        json.dump(extracted_pairs, f, indent=2)

    print(f"Total pairs processed: {len(pairs)}")
    print(f"Total pairs skipped: {skipped_pairs}")    
    return paper_texts, pairs

# Compute KNN-based predictions with BERT embeddings
def knn_predictions(paper_texts, pairs, k=5):
    # Load BERT model (runs on GPU if available)
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Convert paper texts to BERT embeddings
    paper_ids = list(paper_texts.keys())
    paper_vectors = list(paper_texts.values())
    embeddings = model.encode(paper_vectors, convert_to_tensor=True, device=device)
    X = embeddings.cpu().numpy()  # Move to CPU for sklearn compatibility
    
    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')  # +1 to exclude the paper itself
    knn.fit(X)
    
    # Find nearest neighbors for each paper
    distances, indices = knn.kneighbors(X)
    neighbor_dict = {paper_ids[i]: set(paper_ids[idx] for idx in neighbors[1:]) for i, neighbors in enumerate(indices)}
    
    # Predict links: "Yes" if papers are mutual neighbors
    knn_predictions = []
    for paper1_id, paper2_id, _ in pairs:
        if paper2_id in neighbor_dict[paper1_id] and paper1_id in neighbor_dict[paper2_id]:
            knn_predictions.append(1)  # Predict "Yes"
        else:
            knn_predictions.append(0)  # Predict "No"
    
    return knn_predictions

# Evaluate KNN predictions
def evaluate_predictions(pairs, knn_predictions):
    ground_truth = [label for _, _, label in pairs]
    
    # Compute metrics for KNN
    accuracy = accuracy_score(ground_truth, knn_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, knn_predictions, average='binary')
    
    # Print results
    print("KNN Baseline with BERT Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Optionally, save predictions for further analysis
    with open("knn_bert_predictions.json", "w") as f:
        json.dump([
            {"id": f"{p1}_{p2}", "prediction": "Yes" if pred == 1 else "No", "ground_truth": "Yes" if gt == 1 else "No"}
            for (p1, p2, gt), pred in zip(pairs, knn_predictions)
        ], f, indent=2)

def main():
    # File path
    test_file = "/mnt/webscistorage/cc7738/ws_aniket/LLM_for_Het/LLM4HeG/llm_pred/prompt_json/Arxiv/new_test_all.json"
    
    # Load dataset
    test_data = load_json(test_file)
    
    # Extract paper texts and pairs
    paper_texts, pairs = extract_paper_data(test_data)
    
    #Generate KNN predictions with BERT
    k = 25  # Number of neighbors (you can tune this)
    knn_preds = knn_predictions(paper_texts, pairs, k=k)
    
    # Evaluate
    evaluate_predictions(pairs, knn_preds)

if __name__ == "__main__":
    main()