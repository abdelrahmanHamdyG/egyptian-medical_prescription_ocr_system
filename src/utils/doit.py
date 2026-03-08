import cv2
import os
import glob
import shutil
import numpy as np
from craft_text_detector import Craft
import craft_text_detector.craft_utils as craft_utils

# ==========================================
# CRAFT LIBRARY BUG FIX (Monkey Patch)
# ==========================================
def patched_adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] = np.array(polys[k]) * np.array([ratio_w * ratio_net, ratio_h * ratio_net])
    return polys

craft_utils.adjustResultCoordinates = patched_adjustResultCoordinates
# ==========================================

def process_image_folder(input_dir, final_output_dir, padding=10):
    # This is the "decoy" folder to catch the library's annoying nested files
    temp_junk_dir = "temp_craft_junk"
    
    # Ensure our true flat output directory exists
    os.makedirs(final_output_dir, exist_ok=True)

    # =========================
    # 1. Initialize CRAFT
    # =========================
    print("Loading CRAFT model...")
    craft = Craft(
        output_dir=temp_junk_dir, # Send library auto-exports here
        crop_type="box",   
        cuda=True,
        refiner=True,      
        text_threshold=0.6,   
        link_threshold=0.335,  
        low_text=0.335         
    )

    image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]")) + \
                  glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]"))

    if not image_paths:
        print(f"No images found in '{input_dir}'")
        return

    print(f"Found {len(image_paths)} images. Generating padded crops...\n")
    total_crops = 0

    # =========================
    # 2. Iterate Over All Images
    # =========================
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        img_h, img_w = image.shape[:2]

        # Run detection (this automatically dumps nested folders into temp_junk_dir)
        prediction_result = craft.detect_text(img_path)
        boxes = prediction_result["boxes"]

        # =========================
        # 3. Apply Padding & Save Flat
        # =========================
        for i, box in enumerate(boxes):
            box_coords = np.array(box).astype(np.int32)
            x, y, w, h = cv2.boundingRect(box_coords)

            # Apply 10-pixel padding safely
            x_min = max(0, x - padding)
            y_min = max(0, y - padding)
            x_max = min(img_w, x + w + padding)
            y_max = min(img_h, y + h + padding)

            crop = image[y_min:y_max, x_min:x_max]

            # Save the crop into the SINGLE flat folder
            crop_filename = f"{base_name}_crop_{i}.png"
            cv2.imwrite(os.path.join(final_output_dir, crop_filename), crop)
            total_crops += 1

    # =========================
    # 4. Cleanup Memory and Junk Folders
    # =========================
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()

    # Delete the decoy folder and all the nested folders inside it
    if os.path.exists(temp_junk_dir):
        shutil.rmtree(temp_junk_dir)

    print(f"\n✅ Batch processing complete!")
    print(f"Saved {total_crops} crops directly into '{final_output_dir}/'")
    print("Deleted all temporary nested folders.")

# =========================
# Usage
# =========================
if __name__ == "__main__":
    INPUT_FOLDER = "data/raw/eval/n_data_filtered" 
    FLAT_OUTPUT_FOLDER = "data/processed/all_crops_flat_2"
    
    process_image_folder(INPUT_FOLDER, FLAT_OUTPUT_FOLDER, padding=7)