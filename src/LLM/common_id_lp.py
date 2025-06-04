
import json

# Load both JSON files
with open('filtered_long_range_prob_yesno_results.json', 'r') as f1, open('filtered_long_range_prob_score_results.json', 'r') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)

# Extract sets of IDs from each file
ids1 = set(item['id'] for item in data1)
ids2 = set(item['id'] for item in data2)

# Find the intersection
common_ids = ids1.intersection(ids2)

# Print common IDs
print("Common IDs:")
for cid in sorted(common_ids):
    print(cid)

# Optionally: write to a file
with open('common_ids.json', 'w') as fout:
    json.dump(sorted(list(common_ids)), fout, indent=2)
