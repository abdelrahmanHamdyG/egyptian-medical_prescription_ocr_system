import os
import csv
import random
import math
import re   # INTEGRATED: Regex for text cleaning
import numpy as np
import cv2
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from src.generators.generator_config import  GENERATOR_CONFIG
# =========================
# CONFIGURATION
# =========================

GLOBAL_CONFIG=GENERATOR_CONFIG["GLOBAL"]
CONFIG=GENERATOR_CONFIG["ENGLISH"]

class PrescriptionGenerator:
    def __init__(self):
        # Recursively find all fonts in the fonts folder
        self.fonts = glob(os.path.join(GLOBAL_CONFIG["ENG_FONTS_DIR"], "**", "*.[to][t]f"), recursive=True)
        
        
        os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
        print(f"Loaded {len(self.fonts)} fonts.")

    def _clean_text(self, text):
        """
        Removes any character that isn't a Letter, Number, Space, dot, comma, or dash.
        """
        # Keep only: a-z, A-Z, 0-9, space, ., -, ,
        return re.sub(r'[^a-zA-Z0-9\s\.\-,]', '', text)

    def _get_text(self, raw_line):
        """
        Selects words based on the 80/15/5 probability rule 
        and optionally drops a character.
        """
        # 1. Clean the raw line first using Regex
        clean_line = self._clean_text(raw_line)
        
        words = clean_line.split()
        if not words: return ""

        # Weighted selection
        choice = random.choices(CONFIG["WORD_COUNTS"], weights=CONFIG["WORD_WEIGHTS"])[0]
        
        if choice == "all":
            selected = words
        else:
            # Safely slice (handles cases where line has fewer words than choice)
            selected = words[:choice]
        
        text = " ".join(selected)

        # === Capitalize first letter with 30% chance

        if  random.random()<0.3:
            text = text.capitalize()    
        else:
            text=text.lower()
        # ===============================================

        # Character Dropout (Simulates missed pen stroke)
        if len(text) > 3 and random.random() < CONFIG["PROBS"]["CHAR_DROPOUT"]:
            chars = list(text)
            # Pick a random index from 1 to len-1
            idx = random.randint(1, len(chars)-1)
            chars.pop(idx)
            text = "".join(chars)
            
        return text

    def _apply_morphology(self, img_np):
        """
        Simulates ink bleeding (Dilation) or faint pens (Erosion).
        """
        kernel = np.ones((2,2), np.uint8) 
        
        # Invert image because OpenCV morphology works on white pixels
        img_inv = cv2.bitwise_not(img_np)
        
        if random.random() < 0.5:
            # Dilate -> Thicker lines (Heavy Ink)
            img_inv = cv2.dilate(img_inv, kernel, iterations=1)
        else:
            # Erode -> Thinner lines (Faint/Fine Pen)
            img_inv = cv2.erode(img_inv, kernel, iterations=1)
            
        return cv2.bitwise_not(img_inv)

    def _apply_ink_texture(self, img_np):
        """
        Simulates 'Alpha' noise: Randomly lightens pixels *inside* the text
        """
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # Create random noise
        noise = np.random.randint(0, 100, (img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        
        # Mask: Where is the ink? (Value < 200 means dark/ink)
        ink_mask = (v < 200) 
        
        # Where there is ink AND noise is high -> Brighten pixel (fade it)
        fade_amount = 60 
        v[ink_mask & (noise > 70)] += fade_amount 
        
        # Merge back
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

    def _add_salt_pepper(self, img_np):
        """
        Adds random black (pepper) and white (salt) dots.
        """
        noise = np.random.rand(img_np.shape[0], img_np.shape[1])
        # Pepper (Black dots) - very low probability
        img_np[noise < 0.005] = 0   
        # Salt (White dots)
        img_np[noise > 0.995] = 255 
        return img_np

    def generate_image(self, raw_entry):
        """
        Returns: (Image_Object, text_string, font_name_string)
        """
        text = self._get_text(raw_entry)
        if not text: return None, None, None

        # 1. SETUP CANVAS
        pad_w, pad_h = 120, 50
        W, H = CONFIG["FINAL_SIZE"][0] + pad_w, CONFIG["FINAL_SIZE"][1] + pad_h
        
        bg_color = random.choice([(255, 255, 255), (252, 252, 250), (250, 250, 245)])
        text_color = random.choice(GLOBAL_CONFIG["TEXT_COLORS"])
        
        # Select font and save the name
        font_path = random.choice(self.fonts)
        font_name = os.path.basename(font_path)
        
        # 2. FIT FONT SIZE
        font_size = CONFIG["BASE_FONT_SIZE"]
        font = ImageFont.truetype(font_path, font_size)
        target_content_width = CONFIG["FINAL_SIZE"][0] - 20 
        
        while font.getlength(text) > target_content_width and font_size > 20:
            font_size -= 2
            font = ImageFont.truetype(font_path, font_size)

        # 3. DRAW TEXT
        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)
        
        text_w = draw.textlength(text, font=font)
        x_start = W // 2
        y_center = H // 2

        draw.text((x_start, y_center), text, font=font, fill=text_color, anchor="mm")

        
        

        # 4. ROTATION (Safe Rotation)
        if random.random() < CONFIG["PROBS"]["ROTATE"]:
            angle = random.uniform(-4, 4) # +/- 4 degrees is safe
            img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=bg_color)

        # 5. CROP CENTER (To exact 600x80)
        left = (W - CONFIG["FINAL_SIZE"][0]) // 2
        top = (H - CONFIG["FINAL_SIZE"][1]) // 2
        right = left + CONFIG["FINAL_SIZE"][0]
        bottom = top + CONFIG["FINAL_SIZE"][1]
        
        img = img.crop((left, top, right, bottom))

        # 6. NOISE PIPELINE
        img_np = np.array(img)

        if random.random() < CONFIG["PROBS"]["MORPHOLOGY"]:
            img_np = self._apply_morphology(img_np)

        if random.random() < CONFIG["PROBS"]["INK_NOISE"]:
            img_np = self._apply_ink_texture(img_np)

        if random.random() < CONFIG["PROBS"]["SALT_PEPPER"]:
            img_np = self._add_salt_pepper(img_np)

        if random.random() < CONFIG["PROBS"]["BLUR"]:
            img_np = cv2.GaussianBlur(img_np, (3, 3), 0)

        # Return the font name as the 3rd element
        return Image.fromarray(img_np), text, font_name

    def run(self):
        # 1. READ ALL DATA
        with open(GLOBAL_CONFIG["MEDICINES"], "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            all_data = [row[0].strip() for row in reader if row]
            
        print(f"Total medicines found: {len(all_data)}")
        
        # 2. SUBSET SAMPLING
        if len(all_data) < CONFIG["SAMPLE_SIZE"]:
            print(f"Warning: CSV has fewer than {CONFIG['SAMPLE_SIZE']} items. Using all of them.")
            subset_data = all_data
        else:
            subset_data = random.sample(all_data, CONFIG["SAMPLE_SIZE"])
            
        print(f"Selected random subset of {len(subset_data)} items to generate.")

        # 3. GENERATION LOOP
        out_csv = os.path.join(CONFIG["OUTPUT_DIR"], "labels.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # === UPDATED HEADER: Added "font" ===
            writer.writerow(["filename", "text", "font"])

            for i, entry in enumerate(subset_data):
                try:
                    # === UPDATED UNPACKING: Receive font name ===
                    img, label, font_used = self.generate_image(entry)
                    if img is None: continue

                    fname = f"presc_{i:06d}.png"
                    save_path = os.path.join(CONFIG["OUTPUT_DIR"], fname)
                    img.save(save_path)
                    
                    # === UPDATED ROW: Write font name ===
                    writer.writerow([fname, label, font_used])

                    if i % 500 == 0:
                        print(f"Generated {i}/{len(subset_data)} images...")
                
                except Exception as e:
                    print(f"Error on index {i}: {e}")

        print("Dataset generation completed successfully!")

if __name__ == "__main__":
    PrescriptionGenerator().run()