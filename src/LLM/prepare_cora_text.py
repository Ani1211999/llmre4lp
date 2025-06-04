import torch
from torch_geometric.datasets import Planetoid
import pandas as pd
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Simulated dictionary for Cora (replace with actual vocab if available)
CORA_VOCAB = [f"word_{i}" for i in range(1433)]

def extract_keywords(features, top_k=5):
    """Extract top-k keywords from a node's feature vector."""
    indices = np.argsort(features)[-top_k:]
    keywords = [CORA_VOCAB[idx] for idx in indices]
    return keywords

def setup_vicuna_7b():
    """Load Vicuna-7B model and tokenizer."""
    model_name = "lmsys/vicuna-7b-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        use_cache=True,
        low_cpu_mem_usage=True
    )
    return model, tokenizer

def generate_description_with_vicuna(model, tokenizer, prompt):
    """Generate title and abstract using Vicuna-7B."""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.2,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try parsing: strict format, fallback to line-based
    title, abstract = "Default Title", "Default Abstract"
    try:
        if "Title:" in response and "Abstract:" in response:
            title = response.split("Title:")[1].split("Abstract:")[0].strip()
            abstract = response.split("Abstract:")[1].strip()
        else:
            lines = response.strip().split("\n")
            if len(lines) >= 2:
                title = lines[0].replace("Title:", "").strip()
                abstract = lines[1].replace("Abstract:", "").strip()
    except Exception as e:
        print(f"Parsing error: {e}")

    print(f"\nTitle: {title}\nAbstract: {abstract}\n")
    return title, abstract

def prepare_cora_text(data_dir='data/Planetoid', num_nodes_subset=100):
    # Load Cora dataset
    dataset = Planetoid(root=data_dir, name='Cora')
    data = dataset[0]

    label_names = [
        'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
        'Probabilistic_Methods', 'Reinforcement_Learning',
        'Rule_Learning', 'Theory'
    ]

    # Load Vicuna-7B
    model, tokenizer = setup_vicuna_7b()

    # Extract keywords and generate text
    text_data = {'node_id': [], 'title': [], 'abstract': []}
    for i in range(min(num_nodes_subset, data.num_nodes)):
        node_features = data.x[i].numpy()
        label = label_names[data.y[i].item()]
        keywords = extract_keywords(node_features, top_k=5)

        prompt = f"""You are a scientific writer. Generate a concise and realistic title and abstract for a research paper in the field of '{label}'.

Use the following 5 keywords in the content: {', '.join(keywords)}.

Respond only in the following format:

Title: <a concise academic-style title>

Abstract: <a 3-4 sentence abstract that integrates the given keywords>

Do not include any explanation or extra text. Just give the title and abstract.
"""

        title, abstract = generate_description_with_vicuna(model, tokenizer, prompt)
        text_data['node_id'].append(i)
        text_data['title'].append(title)
        text_data['abstract'].append(abstract)

    # Placeholder descriptions for remaining nodes
    for i in range(num_nodes_subset, data.num_nodes):
        text_data['node_id'].append(i)
        label = label_names[data.y[i].item()]
        text_data['title'].append(f"Paper {i}")
        text_data['abstract'].append(f"Abstract for paper {i}. Category: {label}.")

    # Save results
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'cora_text.csv')
    pd.DataFrame(text_data).to_csv(csv_path, index=False)
    print(f"\n✅ Saved Cora text data to {csv_path}")

    pt_path = os.path.join(data_dir, 'cora.pt')
    torch.save(data, pt_path)
    print(f"✅ Saved Cora graph data to {pt_path}")

if __name__ == "__main__":
    prepare_cora_text(num_nodes_subset=25)  # You can change to 2708 for full Cora
