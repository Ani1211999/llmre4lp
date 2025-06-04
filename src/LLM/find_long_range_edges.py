import json

# Load your JSON data (assuming it's saved in a file called 'results.json')
with open('inference_results_lp_llm/long_range_json/preds.json', 'r') as f:
    data = json.load(f)

# Filter entries with res > 0.7
filtered = [item for item in data if item["res"] == "Yes"]

# Print the filtered results
print("Entries with res > 0.7:")
for item in filtered:
    print(item)

# Optionally: Save filtered results to a new file
with open('filtered_long_range_prob_yesno_results.json', 'w') as f:
    json.dump(filtered, f, indent=2)
