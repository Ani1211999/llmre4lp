import numpy as np
import argparse
import os
from sentence_transformers import SentenceTransformer
import torch

def generate_llm_embeddings(node_texts, device, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    Generate LLM embeddings for node texts using SentenceTransformer.
    
    Args:
        node_texts: List or numpy array of text strings (title + abstract).
        device: Torch device (CPU or GPU).
        model_name: Name of the SentenceTransformer model to use.
        batch_size: Batch size for encoding texts.
    
    Returns:
        Numpy array of shape (num_nodes, embedding_dim) containing LLM embeddings.
    """
    print("📝 Generating LLM embeddings...")
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        node_texts,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=True,
        batch_size=batch_size
    )
    embeddings = embeddings.cpu().numpy()  # Convert to numpy for saving
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def main():
    parser = argparse.ArgumentParser(description="Generate LLM embeddings for node texts and update .npz file.")
    parser.add_argument('--input_npz_path', type=str, required=True, 
                        help="Path to the input .npz file containing node_texts and other data.")
    parser.add_argument('--output_npz_path', type=str, required=True, 
                        help="Path to save the output .npz file with LLM embeddings.")
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', 
                        help="SentenceTransformer model name (e.g., 'all-MiniLM-L6-v2').")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Batch size for embedding generation.")
    parser.add_argument('--concatenate_features', action='store_true', 
                        help="If set, concatenate LLM embeddings with existing node_features instead of replacing them.")
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load input .npz file
    print(f"📂 Loading input .npz file from {args.input_npz_path}...")
    data = np.load(args.input_npz_path, allow_pickle=True)
    
    # Ensure node_texts exists
    if 'node_texts' not in data:
        raise ValueError("Input .npz file must contain 'node_texts' array.")
    
    node_texts = data['node_texts']
    print(f"Loaded {len(node_texts)} node texts.")

    # Generate LLM embeddings
    llm_embeddings = generate_llm_embeddings(
        node_texts=node_texts,
        device=device,
        model_name=args.model_name,
        batch_size=args.batch_size
    )

    # Prepare node features
    if args.concatenate_features and 'node_features' in data:
        print("Concatenating LLM embeddings with existing node_features...")
        original_features = data['node_features']
        if original_features.shape[0] != llm_embeddings.shape[0]:
            raise ValueError(f"Mismatch in number of nodes: node_features has {original_features.shape[0]}, "
                           f"but node_texts has {llm_embeddings.shape[0]}.")
        node_features = np.concatenate([original_features, llm_embeddings], axis=1)
        print(f"Combined node_features shape: {node_features.shape}")
    else:
        print("Using LLM embeddings as node_features...")
        node_features = llm_embeddings
        print(f"node_features shape: {node_features.shape}")

    # Save to new .npz file
    print(f"💾 Saving updated data to {args.output_npz_path}...")
    os.makedirs(os.path.dirname(args.output_npz_path), exist_ok=True)
    
    # Create a dictionary with all original data, updating node_features
    output_data = {key: data[key] for key in data.files}
    output_data['node_features'] = node_features
    
    np.savez(args.output_npz_path, **output_data)
    print(f"Successfully saved .npz file with {node_features.shape[0]} nodes and "
          f"embedding dimension {node_features.shape[1]}.")

if __name__ == "__main__":
    main()
