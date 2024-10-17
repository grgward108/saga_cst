# utils/evaluate_penetration_depth.py

import re
import os
import pandas as pd

def extract_and_save_interpenetration_depth(log_file_path, save_csv_path):
    # List to hold extracted data
    data = []

    # Regular expression to extract average_interpenetration_depth_cm
    regex = r"interpenetration_depth_cm:\s*([0-9.]+)"

    # Open the log file and parse lines
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                # Search for the line that contains the average_interpenetration_depth_cm value
                match = re.search(regex, line)
                if match:
                    # Append the value to the data list
                    data.append(float(match.group(1)))

        # Convert the data to a DataFrame
        df = pd.DataFrame(data, columns=['interpenetration_depth_cm'])

        # Save to a CSV file
        df.to_csv(save_csv_path, index=False)
        print(f"CSV file saved at {save_csv_path}")

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

