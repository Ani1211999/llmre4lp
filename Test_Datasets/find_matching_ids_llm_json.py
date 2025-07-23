import json

def get_ids_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {item['id'] for item in data if 'id' in item}

def find_matching_ids(file1, file2):
    ids1 = get_ids_from_json(file1)
    ids2 = get_ids_from_json(file2)

    matches = ids1 & ids2  # Set intersection
    return matches

if __name__ == "__main__":
    file1 = 'llm_pred/prompt_json/Arxiv_updated/hop3_eval.json'
    file2 = 'llm_pred/prompt_json/Arxiv_v3/hop3_eval.json'

    matching_ids = find_matching_ids(file1, file2)

    print(f"Found {len(matching_ids)} matching IDs:")
    for idx, match in enumerate(sorted(matching_ids), 1):
        print(f"{idx}. {match}")