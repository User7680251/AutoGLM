import json
import re
import argparse

# Function to swap {} and [] along with their content in the 'label' field
def swap_brackets(data):
    for item in data:
        label = item['label']
        # Find the content inside {} and []
        action_match = re.search(r"{(.*?)}", label)
        reason_match = re.search(r"\[(.*?)\]", label)

        # If both matches are found, swap them
        if action_match and reason_match:
            action = action_match.group(0)
            reason = reason_match.group(0)
            swapped_label = label.replace(action, "").replace(reason, "").strip() + action + reason
            item['label'] = swapped_label

    return data

# Read JSON data from a file
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Write JSON data to a file
def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_path", type=str, default="visualglm-6b", help='pretrained ckpt')
    args = parser.parse_args()

    # Path to save the modified JSON file
    input_json_path = args.input_json_path
    output_json_path = 'rev.json'

    # Read the JSON data from the input file
    json_data = read_json(input_json_path)

    # Swap the brackets in the JSON data
    swapped_data = swap_brackets(json_data)

    # Write the modified JSON data to the output file
    write_json(output_json_path, swapped_data)

    print("The modified JSON data has been saved to:", output_json_path)

if __name__ == '__main__':
    main()