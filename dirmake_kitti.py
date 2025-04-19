import os

import os

def generate_file_list(output_txt_path, base_paths=("kitti_2015/training", "kitti_2015/testing"), subfolders=("image_2", "image_3")):
    """
    Generate a file with pairs of left-right image paths (only '_10.png' images).
    
    Args:
        output_txt_path (str): Path to save the generated text file.
        base_paths (tuple): Tuple of base directories to process.
        subfolders (tuple): Names of the left/right image subfolders.
    """
    with open(output_txt_path, 'w') as f_out:
        for base_path in base_paths:
            base_path_final = os.path.join("data", base_path)
            folder_left = os.path.join(base_path_final, subfolders[0])  # image_2 (left)
            folder_right = os.path.join(base_path_final, subfolders[1])  # image_3 (right)
            
            # Get only _10.png files from each folder
            left_images = sorted(f for f in os.listdir(folder_left) if f.endswith('_10.png'))
            right_images = sorted(f for f in os.listdir(folder_right) if f.endswith('_10.png'))
            
            # Verify matching counts
            if len(left_images) != len(right_images):
                print(f"Warning: Mismatched left/right image counts in {base_path} "
                      f"({len(left_images)} vs {len(right_images)})")
                # Continue with the smaller count
                min_count = min(len(left_images), len(right_images))
                left_images = left_images[:min_count]
                right_images = right_images[:min_count]
            
            # Write pairs to file
            for left_img, right_img in zip(left_images, right_images):
                # Verify the base filenames match (e.g., 000000_10.png)
                if left_img != right_img:
                    print(f"Warning: Mismatched filenames: {left_img} vs {right_img}")
                    continue
                
                line = f"{base_path}/{subfolders[0]}/{left_img} {base_path}/{subfolders[1]}/{right_img}\n"
                f_out.write(line)

    print(f"Successfully generated {output_txt_path} with {sum(1 for line in open(output_txt_path))} image pairs")

# Example usage
generate_file_list(output_txt_path='filenames/kitti_2015_train_test.txt')

