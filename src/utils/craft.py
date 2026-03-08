import cv2
import os
import numpy as np
from craft_text_detector import Craft
import craft_text_detector.craft_utils as craft_utils

# ==========================================
# 1. Apply the Polygon Patch
# ==========================================
def patched_adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    """
    Patched version of adjustResultCoordinates to handle varying polygon sizes.
    """
    if len(polys) > 0:
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] = np.array(polys[k]) * np.array([ratio_w * ratio_net, ratio_h * ratio_net])
    return polys

craft_utils.adjustResultCoordinates = patched_adjustResultCoordinates

# ==========================================
# 2. Main Experiment Function
# ==========================================
def run_craft_experiments(input_folder, output_base_folder):
    
    # T = text_threshold (Lower = catches fainter text)
    # L = link_threshold (Lower = aggressively connects words/strokes)
    # LT = low_text      (Absolute minimum confidence to keep a text region)
    
    experiments = [
        # 1. Library Defaults
        # {"id": "05", "name": "Merge_Words",       "T": 0.65, "L": 0.45, "LT": 0.45},
        {"id": "05", "name": "Merge_Words",       "T": 0.6, "L": 0.325, "LT": 0.325},
        {"id": "05", "name": "Merge_Words",       "T": 0.6, "L": 0.34, "LT": 0.34},
        {"id": "05", "name": "Merge_Words",       "T": 0.6, "L": 0.35, "LT": 0.35},
        {"id": "05", "name": "Merge_Words",       "T": 0.55, "L": 0.375, "LT": 0.355},
        {"id": "05", "name": "Merge_Words",       "T": 0.50, "L": 0.35, "LT": 0.35},
        {"id": "05", "name": "Merge_Words",       "T": 0.70, "L": 0.5, "LT": 0.5},
        {"id": "05", "name": "Merge_Words",       "T": 0.50, "L": 0.4, "LT": 0.35},
        {"id": "05", "name": "Merge_Words",       "T": 0.50, "L": 0.3, "LT": 0.3},
        {"id": "05", "name": "Merge_Words",       "T": 0.50, "L": 0.21, "LT": 0.28},
        {"id": "05", "name": "Merge_Words",       "T": 0.50, "L": 0.2, "LT": 0.30},
        {"id": "05", "name": "Merge_Words",       "T": 0.50, "L": 0.25, "LT": 0.30},
        {"id": "05", "name": "Merge_Words",       "T": 0.50, "L": 0.3, "LT": 0.25},
        
        {"id": "01", "name": "Defaults",          "T": 0.6, "L": 0.48, "LT": 0.32},
        {"id": "01", "name": "Defaults",          "T": 0.6, "L": 0.33, "LT": 0.43},
        {"id": "01", "name": "Defaults",          "T": 0.6, "L": 0.3, "LT": 0.4},
        {"id": "01", "name": "Defaults",          "T": 0.6, "L": 0.3, "LT": 0.45},
        {"id": "01", "name": "Defaults",          "T": 0.6, "L": 0.4, "LT": 0.35},
        {"id": "01", "name": "Defaults",          "T": 0.7, "L": 0.45, "LT": 0.35},
        {"id": "01", "name": "Defaults",          "T": 0.65, "L": 0.5, "LT": 0.3},
        {"id": "01", "name": "Defaults",          "T": 0.70, "L": 0.5, "LT": 0.3},
        {"id": "01", "name": "Defaults",          "T": 0.70, "L": 0.5, "LT": 0.5},
        {"id": "16", "name": "Defaults_5",          "T": 0.68, "L": 0.38, "LT": 0.365},
        {"id": "15", "name": "Defaults_4",          "T": 0.70, "L": 0.4, "LT": 0.35},
        {"id": "01", "name": "Defaults",          "T": 0.70, "L": 0.40, "LT": 0.40},
        {"id": "13", "name": "Defaults_2",          "T": 0.70, "L": 0.35, "LT": 0.35},
        {"id": "14", "name": "Defaults_3",          "T": 0.68, "L": 0.37, "LT": 0.37},
        
        # 2. Your Original Parameters
        {"id": "02", "name": "User_Original",     "T": 0.20, "L": 0.60, "LT": 0.05},
        
        # 3. High Sensitivity (Catching Faint Ink)
        {"id": "03", "name": "Faint_Ink_1",       "T": 0.40, "L": 0.40, "LT": 0.20},
        {"id": "04", "name": "Faint_Ink_2",       "T": 0.20, "L": 0.20, "LT": 0.10},
        
        # 4. Aggressive Linking (Crucial for Cursive Arabic)
        # Low link threshold forces characters to group together
        {"id": "05", "name": "Merge_Words",       "T": 0.50, "L": 0.15, "LT": 0.30},
        {"id": "06", "name": "Merge_Everything",  "T": 0.30, "L": 0.05, "LT": 0.10},
        
        # 5. Strict Characters / Loose Linking
        {"id": "07", "name": "StrictChar_LooseLink","T": 0.70, "L": 0.10, "LT": 0.40},
        
        # 6. Loose Characters / Strict Linking (Might chop Arabic into letters)
        {"id": "08", "name": "LooseChar_StrictLink","T": 0.20, "L": 0.80, "LT": 0.10},
        
        # 7. Balanced Approaches
        {"id": "09", "name": "Balanced_1",        "T": 0.50, "L": 0.30, "LT": 0.20},
        {"id": "10", "name": "Balanced_2",        "T": 0.40, "L": 0.20, "LT": 0.20},
        
        # 8. Extreme Sensitivity (Will likely catch background noise)
        {"id": "11", "name": "Extreme_Sensitive", "T": 0.10, "L": 0.10, "LT": 0.05},
        {"id": "12", "name": "Extreme_Link",      "T": 0.30, "L": 0.01, "LT": 0.10},
    ]

    os.makedirs(output_base_folder, exist_ok=True)
    
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder '{input_folder}' does not exist!")
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    print(f"Found {len(image_files)} images. Starting {len(experiments)} CRAFT experiments...\n")

    # ==========================================
    # 3. Loop Through Each Experiment
    # ==========================================
    for exp in experiments:
        folder_name = f"Exp{exp['id']}_{exp['name']}_T{exp['T']}_L{exp['L']}_LT{exp['LT']}"
        exp_dir = os.path.join(output_base_folder, folder_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        print(f"--- Running {folder_name} ---")
        
        # Initialize CRAFT with the specific thresholds for this experiment
        craft = Craft(
            output_dir=exp_dir,
            crop_type="box",   
            cuda=True,
            refiner=True,      
            text_threshold=exp['T'],  
            link_threshold=exp['L'], 
            low_text=exp['LT']
        )

        # Process all images
        for img_filename in image_files:
            img_path = os.path.join(input_folder, img_filename)
            
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Run detection (craft-text-detector handles the file reading internally here)
            # We redirect output_dir to the experiment folder so its internal logs/crops go there if enabled
            prediction_result = craft.detect_text(img_path)
            boxes = prediction_result["boxes"]
            
            image_display = image.copy()

            # Draw all bounding boxes
            for box in boxes:
                # Format the box into a clean numpy array
                box_arr = np.array(box).astype(np.int32)
                
                # Draw polygon
                cv2.polylines(
                    image_display,
                    [box_arr.reshape((-1, 1, 2))],
                    True,
                    (0, 255, 0),
                    2
                )

            # Save the single full visualization image
            vis_path = os.path.join(exp_dir, f"craft_vis_{img_filename}")
            cv2.imwrite(vis_path, image_display)

        # VERY IMPORTANT: Free GPU memory before loading the next experiment
        print("Unloading models to free GPU memory...")
        craft.unload_craftnet_model()
        craft.unload_refinenet_model()

    print("\n" + "="*50)
    print(f"✅ All {len(experiments)} CRAFT experiments complete! Check '{output_base_folder}'")
    print("="*50)

# =========================
# Usage
# =========================
if __name__ == "__main__":
    INPUT_DIR = r"data\raw\eval\n_data"
    OUTPUT_DIR = r"craft_threshold_experiments_2"

    run_craft_experiments(input_folder=INPUT_DIR, output_base_folder=OUTPUT_DIR)