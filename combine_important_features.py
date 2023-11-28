import os
import json

def combine_important_features(user_token):
    target = 'hda'
    target = 'bts'
    target = 'over25'
    target = 'cs'
    
    # List of folders to process
    folders = [
        "configs/important_features/regular_prediction_last_7_matches_optimized_30/",
    ]

    # Initialize an empty list to store the features
    all_features = []

    # Loop through each folder
    for folder_path in folders:
        # Recursively iterate through all files in the directory
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                # Check if the file is a JSON file
                if filename.startswith(f"{target}_") and filename.endswith(".json"):
                    file_path = os.path.join(root, filename)

                    # Open and load the JSON file
                    with open(file_path, 'r') as file:
                        features_list = json.load(file)

                        # Extend the main list with the features from the current file
                        all_features.extend(features_list)

    # Convert the list to a set to remove duplicates and then back to a list
    unique_features = sorted(list(set(all_features)))

    if len(unique_features) == 0:
        print(f"No unique features to save.")
        return
    
    # Save the unique features as a JSON file
    output_file_path = f"configs/important_features/regular_prediction/{target}_features.json"
    with open(output_file_path, 'w') as output_file:
        json.dump(unique_features, output_file, indent=2)

    print(f"Unique features saved to: {output_file_path}")

