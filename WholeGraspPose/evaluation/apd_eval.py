# Calculate the Average L2 Pairwise (APD) to evaluate diversity within random samples
import numpy as np

# List of objects to process
objects = ["toothpaste", "camera", "wineglass", "binoculars", "mug"]


datadirectory_template = '../../results/testdump/GraspPose/{object}/markers_results.npy'

# Function to calculate the Average L2 Pairwise Distance (APD)
def calculate_apd(markers):
    n = len(markers)
    if n < 2:
        return 0
    total_distance = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_distance += np.linalg.norm(markers[i] - markers[j])
            count += 1
    return total_distance / count

# Iterate through each object and calculate APD
for obj in objects:
    datadirectory = datadirectory_template.format(object=obj)
    try:
        # Load the data
        data = np.load(datadirectory, allow_pickle=True).item()
        markers_gen = data['markers_gen']

        # Calculate APD
        apd = calculate_apd(markers_gen)
        print(f"Average L2 Pairwise Distance (APD) for {obj}: {apd}")
    except FileNotFoundError:
        print(f"File not found for object: {obj}")
    except KeyError as e:
        print(f"Key error {e} for object: {obj}")
    except Exception as e:
        print(f"An error occurred for object {obj}: {e}")
