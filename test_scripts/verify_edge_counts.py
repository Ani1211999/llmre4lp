import json

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data
    
train_data = load_json("llm_pred/prompt_json/Arxiv/train_all.json")
test_data = load_json("llm_pred/prompt_json/Arxiv/new_test_all.json")
val_data = load_json("llm_pred/prompt_json/Arxiv/val_all.json")    

print(f"Number of edges in training data:{len(train_data)}\n")
print(f"Number of edges in test data:{len(test_data)}\n")
print(f"Number of edges in validation data:{len(val_data)}")