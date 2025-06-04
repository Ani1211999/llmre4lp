import json

def inspect_json(json_path, split_name):
    with open(json_path, 'r') as f:
        data = json.load(f)
    num_edges = len(data)
    node_ids = set()
    for item in data:
        u, v = map(int, item['id'].split('_'))
        node_ids.add(u)
        node_ids.add(v)
    print(f"{split_name} - Number of edges: {num_edges}")
    print(f"{split_name} - Unique node IDs: {len(node_ids)}")
    print(f"{split_name} - Min node ID: {min(node_ids)}")
    print(f"{split_name} - Max node ID: {max(node_ids)}")

# Run inspection
inspect_json("./llm_pred/prompt_json/Arxiv/train_all.json", "Train")
inspect_json("./llm_pred/prompt_json/Arxiv/test_all.json", "Test")
inspect_json("./llm_pred/prompt_json/Arxiv/long_range_infer.json", "Long-Range")