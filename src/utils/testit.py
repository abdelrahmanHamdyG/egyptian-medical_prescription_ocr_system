import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import logging

# Disable verbose PaddleOCR logging to keep the console clean during 12 initializations
logging.getLogger("ppocr").setLevel(logging.ERROR)

def run_threshold_experiments(input_folder, output_base_folder):
    # ==========================================
    # 1. Define the 12 Hyperparameter Combinations
    # ==========================================
    # L = det_limit_side_len (Resolution before detection)
    # T = det_db_thresh (Binarization threshold - lower catches fainter text)
    # B = det_db_box_thresh (Confidence to keep box - lower keeps "unsure" boxes)
    # U = det_db_unclip_ratio (Polygon expansion - higher makes boxes wider)
    
    experiments = [
        # 1. Refining Exp 5 (The Low B strategy) + Wider Boxes
        # Exp 5 was strong. We keep its T and B, but expand the box (U) to grab cut-off letters.
        {"id": "13", "name": "Exp5_Wide",         "L": 1920, "T": 0.30, "B": 0.20, "U": 1.9},
        {"id": "14", "name": "Exp5_ExtraWide",    "L": 1920, "T": 0.30, "B": 0.20, "U": 2.2},

        # 2. Refining Exp 4 (The Low T strategy) + Wider Boxes
        # Exp 4 found faint text well. We keep the low T, slightly relax B, and widen the box.
        {"id": "15", "name": "Exp4_Refined",      "L": 1920, "T": 0.15, "B": 0.45, "U": 1.8},

        # 3. The "Golden Middle" Combinations
        # A mathematical average of the best traits from 4, 5, 11, and 12.
        {"id": "16", "name": "Golden_Average",    "L": 1920, "T": 0.22, "B": 0.35, "U": 1.8},
        {"id": "17", "name": "Golden_Sensitive",  "L": 1920, "T": 0.20, "B": 0.25, "U": 2.0},

        # 4. The High-Res Cursive Catcher
        # Applying the best new balanced parameters to a higher resolution just in case it helps with dense lines.
        {"id": "18", "name": "HighRes_Golden",    "L": 2560, "T": 0.22, "B": 0.25, "U": 2.0},
    ]

    # Ensure output base directory exists
    os.makedirs(output_base_folder, exist_ok=True)
    
    # Get images
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder '{input_folder}' does not exist!")
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    print(f"Found {len(image_files)} images. Starting 12 experiments...\n")

    # ==========================================
    # 2. Loop Through Each Experiment
    # ==========================================
    for exp in experiments:
        # Create a descriptive folder name so you know the settings just by looking at it
        folder_name = f"Exp{exp['id']}_{exp['name']}_L{exp['L']}_T{exp['T']}_B{exp['B']}_U{exp['U']}"
        exp_dir = os.path.join(output_base_folder, folder_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        print(f"--- Running {folder_name} ---")
        
        # Initialize PaddleOCR with this experiment's parameters
        detector = PaddleOCR(
            use_angle_cls=True, 
            lang='ar',           
            det=True,            
            rec=False,           
            show_log=False,
            det_limit_side_len=exp['L'], 
            det_db_thresh=exp['T'],      
            det_db_box_thresh=exp['B'],  
            det_db_unclip_ratio=exp['U'] 
        )

        # ==========================================
        # 3. Process All Images for This Experiment
        # ==========================================
        for img_filename in image_files:
            img_path = os.path.join(input_folder, img_filename)
            
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Run detection
            det_result = detector.ocr(image, cls=False, det=True, rec=False)
            boxes = det_result[0] if det_result[0] is not None else []
            
            image_display = image.copy()

            # Draw all bounding boxes
            for item in boxes:
                # The Bulletproof Check
                if len(item) == 2 and isinstance(item[1], tuple):
                    box_coords = item[0]  
                else:
                    box_coords = item     

                # Convert to numpy array safely and draw
                points = np.array(box_coords).astype(np.int32)
                cv2.polylines(image_display, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            # Save the single full visualization image
            vis_path = os.path.join(exp_dir, img_filename)
            cv2.imwrite(vis_path, image_display)

    print("\n" + "="*50)
    print(f"✅ All 12 experiments complete! Check the folders in '{output_base_folder}'")
    print("="*50)

# =========================
# Usage
# =========================
if __name__ == "__main__":
    INPUT_DIR = r"data\raw\eval\n_data"
    OUTPUT_DIR = r"threshold_experiments"

    run_threshold_experiments(input_folder=INPUT_DIR, output_base_folder=OUTPUT_DIR)