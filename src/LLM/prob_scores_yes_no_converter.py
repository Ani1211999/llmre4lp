import json
import argparse

def convert_scores(input_file, output_file, threshold=0.7):
    # Load input JSON
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Convert float scores to Yes/No
    converted = [
        {
            "id": item["id"],
            "res": "Yes" if float(item["res"]) > threshold else "No"
        }
        for item in data
    ]

    # Save output JSON
    with open(output_file, 'w') as f:
        json.dump(converted, f, indent=4)

    print(f"✅ Processed file saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert probability scores to Yes/No labels.")
    parser.add_argument("--input", "-i", required=True, help="Path to input JSON file")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON file")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Threshold for Yes/No classification")

    args = parser.parse_args()
    convert_scores(args.input, args.output, args.threshold)
