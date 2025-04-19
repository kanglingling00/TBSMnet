import os

def generate_file_list(output_txt_path, base_dirs=("scene_flow/driving", "scene_flow/flyingthing", "scene_flow/monkaa"), subfolders=("left", "right")):
    """
    Generate a file with pairs of file paths from 'left' and 'right' subfolders 
    within each subfolder of base directories (driving, flyingthings, mono).
    
    Args:
        output_txt_path (str): Path to save the generated text file.
        base_dirs (tuple): Tuple of base directories (driving, flyingthings, mono).
        subfolders (tuple): Names of the subfolders to process (left, right).
    """
    # Open the output file
    with open(output_txt_path, 'w') as f_out:
        # Process each base directory
        for base_dir in base_dirs:
            base_dir_path = os.path.join("data", base_dir)
            
            # Iterate through all subfolders in the base directory
            for root, dirs, files in os.walk(base_dir_path):
                # Check if 'left' and 'right' subfolders exist in the current directory
                if subfolders[0] in dirs and subfolders[1] in dirs:
                    left_folder = os.path.join(root, subfolders[0])
                    right_folder = os.path.join(root, subfolders[1])
                    
                    # Get list of .png files from each folder
                    files_left = sorted(f for f in os.listdir(left_folder) if f.endswith('.png'))
                    files_right = sorted(f for f in os.listdir(right_folder) if f.endswith('.png'))
                    
                    if len(files_left) != len(files_right):
                        print(f"Warning: The number of files in the 'left' and 'right' folders in {root} does not match.")
                    
                    # Generate pairs and write to output file
                    for file_left, file_right in zip(files_left, files_right):
                        line = f"{os.path.relpath(left_folder, 'data')}/{file_left} {os.path.relpath(right_folder, 'data')}/{file_right}\n"
                        f_out.write(line)

    print(f"File list generated and saved to {output_txt_path}")

# Example usage
generate_file_list(output_txt_path='filenames/scence_flow_vit_train.txt')
