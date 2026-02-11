import cv2
import os
import csv
import glob
import gc  # <--- NEW: Garbage Collector to fix memory leaks/lag

# --- CONFIGURATION ---
INPUT_FOLDER = "dataset_images"          
OUTPUT_FOLDER = "output_eval"   
CSV_FILE = "labels_eval.csv"          
DISPLAY_HEIGHT = 700  
# ---------------------

# Global variables
ref_point = []
cropping = False
display_image = None
display_clone = None
original_image = None 
scale_factor = 1.0    

def shape_selection(event, x, y, flags, param):
    global ref_point, cropping, display_image, display_clone

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        img_copy = display_clone.copy()
        cv2.rectangle(img_copy, ref_point[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("Annotator", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        cv2.rectangle(display_image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Annotator", display_image)

def save_crop(filename, counter):
    global ref_point, original_image, scale_factor
    
    if not ref_point or len(ref_point) != 2:
        return False

    # 1. Coordinate Math
    x1, y1 = ref_point[0]
    x2, y2 = ref_point[1]
    
    x_start = min(x1, x2)
    x_end = max(x1, x2)
    y_start = min(y1, y2)
    y_end = max(y1, y2)

    if x_start == x_end or y_start == y_end:
        return False

    # 2. Scale up to original size
    real_y1 = int(y_start / scale_factor)
    real_y2 = int(y_end / scale_factor)
    real_x1 = int(x_start / scale_factor)
    real_x2 = int(x_end / scale_factor)

    # 3. Crop High-Res Image
    try:
        roi = original_image[real_y1:real_y2, real_x1:real_x2]
    except Exception as e:
        print(f"[!] Crop failed: {e}")
        return False

    cv2.imshow("Crop Preview", roi)
    cv2.waitKey(1) 

    # 4. Get Label
    print(f"\n>>> Enter label for crop (empty to discard): ", end="")
    label = input() 
    cv2.destroyWindow("Crop Preview")

    if not label.strip():
        print("[-] Discarded.")
        return True 

    # 5. Save Image
    base_name = os.path.splitext(filename)[0]
    crop_filename = f"{base_name}_crop_{counter}.jpg"
    crop_path = os.path.join(OUTPUT_FOLDER, crop_filename)
    cv2.imwrite(crop_path, roi)

    # 6. INSTANT SAVE TO CSV
    file_exists = os.path.isfile(CSV_FILE)
    
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['filename', 'label'])
        
        writer.writerow([crop_filename, label])
        
        # FORCE WRITE TO DISK IMMEDIATELY
        f.flush()
        os.fsync(f.fileno()) 

    print(f"[+] Saved & Logged: {crop_filename}")
    return True

def main():
    global display_image, display_clone, original_image, ref_point, scale_factor

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    img_files.sort()

    if not img_files:
        print(f"No images found in '{INPUT_FOLDER}'.")
        return

    print("--- ROBUST OCR ANNOTATOR ---")
    
    # Optional: Resume Feature
    # Change start_index if you want to skip the first N images
    start_index = 0 
    
    for idx, filepath in enumerate(img_files):
        if idx < start_index:
            continue

        filename = os.path.basename(filepath)
        print(f"\nProcessing {idx+1}/{len(img_files)}: {filename}")

        # Load Image
        original_image = cv2.imread(filepath)
        if original_image is None:
            continue

        # Resize for display
        h, w = original_image.shape[:2]
        scale_factor = DISPLAY_HEIGHT / float(h)
        if scale_factor > 1: scale_factor = 1.0
        new_dim = (int(w * scale_factor), int(h * scale_factor))
        
        display_image = cv2.resize(original_image, new_dim)
        display_clone = display_image.copy()

        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", shape_selection)

        crop_counter = 0

        while True:
            cv2.imshow("Annotator", display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"): # Reset
                display_image = display_clone.copy()
                ref_point = []

            elif key == ord("s"): # Save
                if len(ref_point) == 2:
                    success = save_crop(filename, crop_counter)
                    if success:
                        crop_counter += 1
                        display_image = display_clone.copy() 
                        ref_point = []
                else:
                    print("Select region first.")

            elif key == ord("n"): # Next Image
                break
            
            elif key == ord("q"): # Quit
                print("Exiting...")
                cv2.destroyAllWindows()
                return

        # --- MEMORY CLEANUP ---
        # This prevents the "Lag and Stop Working" issue
        del original_image
        del display_image
        del display_clone
        cv2.destroyAllWindows()
        gc.collect() 
        # ----------------------

if __name__ == "__main__":
    main()