import cv2
import os
import numpy as np
from craft_text_detector import Craft


def run_craft_detection(image_path, output_dir="output_crops"):
    # =========================
    # 1. Initialize CRAFT
    # =========================
    print("Loading CRAFT model (with refiner)...")

    craft = Craft(
        output_dir=output_dir,
        crop_type="box",   # allow curved / handwritten text
        cuda=True,
        refiner=True,        # enable refiner
        text_threshold=0.5,  
        link_threshold=0.35, 
        low_text=0.2
    )

    # =========================
    # 2. Load image
    # =========================
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found!")

    # =========================
    # 3. Run detection
    # =========================
    prediction_result = craft.detect_text(image_path)

    # =========================
    # 4. Free GPU memory
    # =========================
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()

    # IMPORTANT:
    # Use ONLY boxes (stable), ignore polys
    boxes = prediction_result["boxes"]
    print(f"Found {len(boxes)} text regions.")

    # =========================
    # 5. Visualization + Cropping
    # =========================
    os.makedirs(output_dir, exist_ok=True)
    image_display = image.copy()
    crops_for_ocr = []

    for i, box in enumerate(boxes):
        box = np.array(box).astype(np.int32)

        # Draw detection for visualization
        cv2.polylines(
            image_display,
            [box.reshape((-1, 1, 2))],
            True,
            (0, 255, 0),
            2
        )

        # Convert polygon -> bounding rectangle
        x, y, w, h = cv2.boundingRect(box)

        # Padding to avoid cutting characters
        pad = 6
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2 * pad)
        h = min(image.shape[0] - y, h + 2 * pad)

        crop = image[y:y + h, x:x + w]

        crop_path = os.path.join(output_dir, f"crop_{i}.png")
        cv2.imwrite(crop_path, crop)
        crops_for_ocr.append(crop)

    # Save visualization image
    cv2.imwrite(
        os.path.join(output_dir, "full_detection_result.png"),
        image_display
    )

    print(f"✅ Saved {len(crops_for_ocr)} crops to '{output_dir}/'")
    return crops_for_ocr


# =========================
# Usage
# =========================
if __name__ == "__main__":
    image_path = "0_43.png"

    # Create dummy image if missing
    if not os.path.exists(image_path):
        dummy = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.putText(
            dummy,
            "Paracetamol 500mg",
            (40, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        cv2.imwrite(image_path, dummy)

    crops = run_craft_detection(image_path)

    # Plug OCR here (TrOCR / CRNN)
    # for crop in crops:
    #     text = trocr_model(crop)
    #     print(text)
