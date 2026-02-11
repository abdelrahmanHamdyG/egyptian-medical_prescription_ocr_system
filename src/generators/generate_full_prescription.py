import os
import csv
import random
import math
import re
import numpy as np
import cv2
from glob import glob
from PIL import Image, ImageDraw, ImageFont

# --- ARABIC SUPPORT ---
import arabic_reshaper
from bidi.algorithm import get_display

# =========================
# CONFIGURATION
# =========================
# Assuming your directory structure remains the same
from src.generators.generator_config import GENERATOR_CONFIG

GLOBAL_CONFIG = GENERATOR_CONFIG["GLOBAL"]
CONFIG = GENERATOR_CONFIG["DUAL_LANG"]

class DualLanguageGenerator:
    def __init__(self):
        # 1. Load Fonts
        self.eng_fonts = glob(os.path.join(GLOBAL_CONFIG["ENG_FONTS_DIR"], "**", "*.[to][t]f"), recursive=True)
        self.ara_fonts = glob(os.path.join(GLOBAL_CONFIG["ARA_FONTS_DIR"], "**", "*.[to][t]f"), recursive=True)
        
        if not self.eng_fonts: raise ValueError("No English fonts found!")
        if not self.ara_fonts: raise ValueError("No Arabic fonts found!")
        
        # 2. Load Arabic Dataset (The "New Way")
        self.ara_dataset = []
        if not os.path.exists(GLOBAL_CONFIG["INSTRUCTIONS"]):
            raise ValueError(f"Missing Arabic Instructions CSV at {GLOBAL_CONFIG['INSTRUCTIONS']}!")
            
        with open(GLOBAL_CONFIG["INSTRUCTIONS"], "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if row:
                    # Taking the first column: instruction_ar
                    self.ara_dataset.append(row[0].strip())
        
        if not self.ara_dataset:
            raise ValueError("Arabic dataset is empty or formatted incorrectly.")

        os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
        print(f"Loaded {len(self.eng_fonts)} English and {len(self.ara_fonts)} Arabic fonts.")
        print(f"Loaded {len(self.ara_dataset)} Arabic base instructions.")

    # =========================
    # TEXT GENERATION LOGIC
    # =========================

    def _clean_eng(self, text):
        return re.sub(r'[^a-zA-Z0-9\s\.\-,]', '', text)

    def _to_indian_nums(self, text):
        translation = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
        return text.translate(translation)

    def _get_english_text(self, raw_line):
        clean_line = self._clean_eng(raw_line)
        words = clean_line.split()
        if not words: return ""

        choice = random.choices(CONFIG["WORD_COUNTS"], weights=CONFIG["WORD_WEIGHTS"])[0]
        if choice == "all": 
            selected = words
        else: 
            selected = words[:choice]
        
        text = " ".join(selected)
        text = text.capitalize() if random.random() < 0.3 else text.lower()

        if len(text) > 3 and random.random() < CONFIG["PROBS"]["ENG_CHAR_DROP"]:
            chars = list(text)
            chars.pop(random.randint(1, len(chars)-1))
            text = "".join(chars)
            
        return text

    def _get_arabic_text(self):
        """
        Logic from Script 1:
        - 60% Full text
        - 20% 1 Random word
        - 20% 2 Consecutive words
        """
        base_text = random.choice(self.ara_dataset)
        words = base_text.split()
        rand_val = random.random()
        
        if rand_val < 0.60:
            final_text = base_text
        elif rand_val < 0.80:
            final_text = random.choice(words) if words else base_text
        else:
            if len(words) >= 2:
                start_idx = random.randint(0, len(words) - 2)
                final_text = " ".join(words[start_idx : start_idx + 2])
            else:
                final_text = words[0] if words else base_text

        # Apply Indian Numbers
        raw_label = self._to_indian_nums(final_text)
        
        # Reshape and Bidi for PIL rendering
        reshaped = arabic_reshaper.reshape(raw_label)
        display_text = get_display(reshaped)
        
        return display_text, raw_label

    # =========================
    # LAYOUT & POSITIONING
    # =========================

    def _get_layout_positions(self, W, H, eng_w, eng_h, ara_w, ara_h):
        layout_type = "VERTICAL"
        
        # Determine if Horizontal layout is viable
        if random.random() < CONFIG["PROBS"]["LAYOUT_HORIZONTAL"]:
            spacing = 40
            total_w = eng_w + spacing + ara_w
            if total_w < (W - 100): # Margin check
                layout_type = "HORIZONTAL"

        jitter_x = int(random.gauss(0, 10))
        jitter_y = int(random.gauss(0, 5))

        if layout_type == "VERTICAL":
            # English Top, Arabic Bottom
            eng_cy = (H // 2) - 40 + jitter_y
            ara_cy = (H // 2) + 40 + int(random.gauss(0, 5))
            
            x_eng = (W - eng_w) // 2 + jitter_x
            y_eng = eng_cy - (eng_h // 2)
            
            x_ara = (W - ara_w) // 2 + int(random.gauss(0, 10))
            y_ara = ara_cy - (ara_h // 2)
        else:
            # Horizontal Side-by-Side
            spacing = 40
            total_content_w = eng_w + spacing + ara_w
            start_x = (W - total_content_w) // 2 + jitter_x
            center_y = (H // 2) + jitter_y
            
            x_eng = start_x
            y_eng = center_y - (eng_h // 2)
            x_ara = x_eng + eng_w + spacing
            y_ara = center_y - (ara_h // 2)

        # Safety Clamp
        margin = 15
        x_eng = max(margin, min(W - eng_w - margin, x_eng))
        y_eng = max(margin, min(H - eng_h - margin, y_eng))
        x_ara = max(margin, min(W - ara_w - margin, x_ara))
        y_ara = max(margin, min(H - ara_h - margin, y_ara))

        return (x_eng, y_eng), (x_ara, y_ara), layout_type

    # =========================
    # AUGMENTATIONS
    # =========================

    def _apply_morphology(self, img_np):
        kernel = np.ones((2,2), np.uint8)
        img_inv = cv2.bitwise_not(img_np)
        if random.random() < 0.5:
            img_inv = cv2.dilate(img_inv, kernel, iterations=1)
        else:
            img_inv = cv2.erode(img_inv, kernel, iterations=1)
        return cv2.bitwise_not(img_inv)

    def _apply_ink_noise(self, img_np):
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        noise = np.random.randint(0, 100, (img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        v[(v < 210) & (noise > 75)] += 50
        return cv2.cvtColor(cv2.merge((h,s,v)), cv2.COLOR_HSV2RGB)

    def _add_salt_pepper(self, img_np):
        noise = np.random.rand(*img_np.shape[:2])
        img_np[noise < 0.002] = 0
        img_np[noise > 0.998] = 255
        return img_np

    # =========================
    # CORE GENERATION
    # =========================

    def generate_image(self, eng_raw_input):
        eng_text = self._get_english_text(eng_raw_input)
        if not eng_text: return None
        
        ara_display, ara_label = self._get_arabic_text()
        
        W_final, H_final = CONFIG["FINAL_SIZE"]
        pad_w, pad_h = 300, 200 
        W, H = W_final + pad_w, H_final + pad_h
        
        bg_color = random.choice([(255, 255, 255), (252, 252, 250), (250, 250, 245)])
        text_color = random.choice(GLOBAL_CONFIG["TEXT_COLORS"])
        
        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)
        
        # --- Handle English Font ---
        eng_font_path = random.choice(self.eng_fonts)
        eng_size = CONFIG["ENG_BASE_SIZE"]
        eng_font = ImageFont.truetype(eng_font_path, eng_size)
        
        while eng_font.getlength(eng_text) > (W_final * 0.8) and eng_size > 18:
            eng_size -= 2
            eng_font = ImageFont.truetype(eng_font_path, eng_size)
            
        eng_bbox = draw.textbbox((0,0), eng_text, font=eng_font)
        eng_w, eng_h = eng_bbox[2]-eng_bbox[0], eng_bbox[3]-eng_bbox[1]

        # --- Handle Arabic Font ---
        ara_font_path = random.choice(self.ara_fonts)
        ara_size = CONFIG["ARA_BASE_SIZE"]
        ara_font = ImageFont.truetype(ara_font_path, ara_size)
        
        while ara_font.getlength(ara_display) > (W_final * 0.8) and ara_size > 18:
            ara_size -= 2
            ara_font = ImageFont.truetype(ara_font_path, ara_size)
            
        ara_bbox = draw.textbbox((0,0), ara_display, font=ara_font)
        ara_w, ara_h = ara_bbox[2]-ara_bbox[0], ara_bbox[3]-ara_bbox[1]

        # --- Positioning ---
        pos_eng, pos_ara, layout_used = self._get_layout_positions(W, H, eng_w, eng_h, ara_w, ara_h)

        draw.text(pos_eng, eng_text, font=eng_font, fill=text_color)
        draw.text(pos_ara, ara_display, font=ara_font, fill=text_color)

        # --- Post-Processing ---
        if random.random() < CONFIG["PROBS"]["ROTATE"]:
            angle = random.uniform(-2.5, 2.5)
            img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=bg_color)
            
        left, top = (W - W_final) // 2, (H - H_final) // 2
        img = img.crop((left, top, left + W_final, top + H_final))
        
        img_np = np.array(img)
        if random.random() < CONFIG["PROBS"]["MORPHOLOGY"]: img_np = self._apply_morphology(img_np)
        if random.random() < CONFIG["PROBS"]["INK_NOISE"]: img_np = self._apply_ink_noise(img_np)
        if random.random() < CONFIG["PROBS"]["SALT_PEPPER"]: img_np = self._add_salt_pepper(img_np)
        if random.random() < CONFIG["PROBS"]["BLUR"]: img_np = cv2.GaussianBlur(img_np, (3,3), 0)
            
        return Image.fromarray(img_np), {
            "eng_text": eng_text, "ara_text": ara_label,
            "eng_font": os.path.basename(eng_font_path),
            "ara_font": os.path.basename(ara_font_path),
            "layout": layout_used
        }

    def run(self):
        # Load English Data
        with open(GLOBAL_CONFIG["MEDICINES"], "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            eng_data = [row[0].strip() for row in reader if row]
            
        subset = random.sample(eng_data, min(len(eng_data), CONFIG["SAMPLE_SIZE"]))
        csv_path = os.path.join(CONFIG["OUTPUT_DIR"], "labels.csv")
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "eng_text", "ara_text", "eng_font", "ara_font", "layout"])
            
            for i, item in enumerate(subset):
                try:
                    res = self.generate_image(item)
                    if not res: continue
                    
                    img, info = res
                    fname = f"dual_{i:06d}.png"
                    img.save(os.path.join(CONFIG["OUTPUT_DIR"], fname))
                    writer.writerow([fname, info["eng_text"], info["ara_text"], info["eng_font"], info["ara_font"], info["layout"]])
                    
                    if i % 100 == 0: print(f"Progress: {i}/{len(subset)}")
                except Exception as e:
                    print(f"Error at index {i}: {e}")
                    
        print(f"Generation complete. Files saved to: {CONFIG['OUTPUT_DIR']}")

if __name__ == "__main__":
    DualLanguageGenerator().run()