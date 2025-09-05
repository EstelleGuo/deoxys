import h5py
import matplotlib.pyplot as plt

with h5py.File('/home/hinton/deoxys_gyh/example_data/pusht.hdf5', 'r') as f:
    # Accessing a nested group
    nested_group = f['data/demo_5']  # Nested group path
    print("Dataset in nested group:", nested_group['actions'][:])
    print("Dataset in nested group:", nested_group['obs/camera_0_color'][:])
    print("Dataset in nested group:", nested_group['obs/proprio_ee'][:])
    # print("Dataset in nested group:", nested_group['obs/proprio_gripper_state'][:])
    print("Dataset in nested group:", nested_group['obs/proprio_joints'][:])
    


    # Create a 1x5 grid of subplots (1 row, 5 columns)
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))  # Adjust figsize if necessary
    
    # Loop through the first 5 images (camera_0_color to camera_4_color)
    for i in range(5):
        # Access each image dataset
        dataset_path = f'/data/demo_9/obs/camera_{i}_color'
        image = f[dataset_path][0]  # Load the first image from each dataset
        
        # Display the image in the corresponding subplot
        ax = axes[i]
        ax.imshow(image)
        ax.axis('off')  # Turn off the axis for a cleaner view
        ax.set_title(f'Image {i}')  # Set the title for each image

    # Show the plot
    plt.tight_layout()  # Adjust spacing between images
    plt.show()