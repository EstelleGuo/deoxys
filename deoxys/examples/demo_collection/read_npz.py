import numpy as np

# Load the .npz file
data = np.load('/home/hinton/deoxys_gyh/example_data/run24/testing_demo_action.npz')

# List the names of the arrays in the .npz file
print("Array names in the .npz file:", data.files)

# Access and print each array
for name in data.files:
    print(f"Array name: {name}")
    print(f"Shape of {name}: {data[name].shape}")
    print(f"Data in {name}:\n{data[name][100:200]}\n")
