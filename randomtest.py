import numpy as np

# Replace 'your_c_weights_path' with the actual file path
c_weights = np.load('WholeGraspPose/configs/rhand_weight.npy')

# Check the type of c_weights to confirm it's a tuple
print(type(c_weights))

# If c_weights is a tuple, you can print the shape of each element if they are arrays
for i, element in enumerate(c_weights):
    print(f"Element {i} shape: {element.shape if hasattr(element, 'shape') else 'Not an array'}")
