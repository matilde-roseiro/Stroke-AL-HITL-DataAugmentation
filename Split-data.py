import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import imageio
from PIL import Image

# ======================
# Configurations and Paths
# ======================
num_subjects = 82                         # Total number of patients
new_size = (512, 512)                     # Size for resizing images and masks
current_dir = Path(os.getcwd())           # Current working directory
dataset_dir = current_dir / 'Patients_CT' # Directory containing raw dataset

# ======================
# Read labels
# ======================
 # CSV with slice information
hemorrhage_diagnosis_df = pd.read_csv("/workspace/Matilde-IA/hemorrhage_diagnosis.csv")
# Convert DataFrame to NumPy array
hemorrhage_diagnosis_array = hemorrhage_diagnosis_df.values  

# ======================
# Prepare arrays
# ======================
all_ct_scans = np.zeros([hemorrhage_diagnosis_array.shape[0],
                         new_size[0], new_size[1]], dtype=np.uint8)     # Array for storing images
all_segments = np.zeros([hemorrhage_diagnosis_array.shape[0],
                         new_size[0], new_size[1]], dtype=np.uint8)     # Array for storing masks

# ======================
# Train Folder
# ======================
train_path = Path('train')                      # Main training folder
image_path = train_path / 'image'               # Subfolder for images
label_path = train_path / 'label'               # Subfolder for masks

# Create folder if it doesn't exist
image_path.mkdir(parents=True, exist_ok=True)
label_path.mkdir(parents=True, exist_ok=True)

# ======================
# Process Images
# ======================
counter = 0
# Iterate over patients (starting from 49)
for subj_no in tqdm(range(49, num_subjects + 49)):
    dataset_dir_subj = dataset_dir / f"{subj_no:03d}"

    # Filter rows in the CSV for the current patient
    idx = hemorrhage_diagnosis_array[:, 0] == subj_no
    slice_numbers = hemorrhage_diagnosis_array[idx, 1]
    
    for slice_idx in range(slice_numbers.size):
        # --- Process CT image ---
        img_path = dataset_dir_subj / 'brain' / f"{slice_numbers[slice_idx]}.jpg"
        img = imageio.imread(img_path)
        resized_img = np.array(Image.fromarray(img).resize(new_size))
        all_ct_scans[counter] = resized_img
        imageio.imwrite(image_path / f"{counter}.png", resized_img)

        # --- Process mask / segmentation ---
        seg_path = dataset_dir_subj / 'brain' / f"{slice_numbers[slice_idx]}_HGE_Seg.jpg"
        if seg_path.exists():
            seg_img = imageio.imread(seg_path)
            
            # Resize mask with NEAREST neighbor to avoid smoothing labels
            resized_seg = np.array(Image.fromarray(seg_img).resize(new_size))
            
            # Normalize to binary: 0 or 255
            resized_seg = np.where(resized_seg > 128, 255, 0).astype(np.uint8)
        else:
            resized_seg = np.zeros(new_size, dtype=np.uint8)

        all_segments[counter] = resized_seg
        imageio.imwrite(label_path / f"{counter}.png", resized_seg)

        counter += 1

print("Processing completed!")
