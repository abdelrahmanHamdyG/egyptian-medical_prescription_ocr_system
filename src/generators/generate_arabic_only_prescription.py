import os
import csv
import random
import math
import numpy as np
import cv2
from glob import glob
from PIL import Image, ImageDraw, ImageFont

# --- IMPORTS FOR ARABIC TEXT ---
import arabic_reshaper
from bidi.algorithm import get_display
from src.generators.generator_config import  GENERATOR_CONFIG


GLOBAL_CONFIG=GENERATOR_CONFIG["GLOBAL"]
CONFIG=GENERATOR_CONFIG["ARABIC"]

class ArabicPrescriptionGenerator:
    def __init__(self):
        # 1. Load Fonts
        self.fonts = glob(os.path.join(GLOBAL_CONFIG["ARA_FONTS_DIR"], "**", "*.[to][t]f"), recursive=True)
        
        
        # 2. Load Dataset
        self.dataset = []
        if not os.path.exists(GLOBAL_CONFIG["INSTRUCTIONS"]):
            raise ValueError(f"Missing {GLOBAL_CONFIG['INSTRUCTIONS']}!")
            
        with open(GLOBAL_CONFIG["INSTRUCTIONS"], "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # Skip header if it exists
            header = next(reader, None) 
            
            for row in reader:
                if row:
                    # Taking the first column: instruction_ar
                    self.dataset.append(row[0].strip())
        
        if not self.dataset:
            raise ValueError("CSV file is empty or formatted incorrectly.")

        os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    def _fix_arabic(self, text):
        """Reshapes letters (Connects them) and Reverses direction (RTL)."""
        reshaped_text = arabic_reshaper.reshape(text) 
        bidi_text = get_display(reshaped_text)        
        return bidi_text

    def _to_indian_numbers(self, text):
        """Converts standard 0-9 to Arabic-Indic ٠-٩"""
        translation = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
        return text.translate(translation)

    def _generate_text_logic(self):
        """
        Logic:
        - Pick random row
        - 60% Full text
        - 20% 1 Random word
        - 20% 2 Consecutive words (in order)
        """
        # 1. Pick a random instruction line
        base_text = random.choice(self.dataset)
        words = base_text.split()
        
        rand_val = random.random()
        
        if rand_val < 0.60:
            # --- 60% -> Full Instruction ---
            final_text = base_text
            
        elif rand_val < 0.80:
            # --- 20% -> 1 Random Word ---
            if words:
                final_text = random.choice(words)
            else:
                final_text = base_text
                
        else:
            # --- 20% -> 2 CONSECUTIVE Words ---
            if len(words) >= 2:
                # Pick a random start index that allows for 2 words
                # Range is 0 to length-2
                start_idx = random.randint(0, len(words) - 2)
                
                # Slice the two consecutive words and join them
                consecutive_words = words[start_idx : start_idx + 2]
                final_text = " ".join(consecutive_words)
            elif words:
                # Fallback if sentence only has 1 word
                final_text = words[0]
            else:
                final_text = base_text

        # Apply Indian Numbers
        final_text = self._to_indian_numbers(final_text)
        return final_text 

    

    def generate_image(self):
        # 1. Get Text based on logic
        raw_label = self._generate_text_logic()
        
        # 2. Fix Arabic (Reshape + Bidi)
        display_text = self._fix_arabic(raw_label)

        # 3. Canvas Setup
        pad_w, pad_h = 100, 60
        W, H = CONFIG["FINAL_SIZE"][0] + pad_w, CONFIG["FINAL_SIZE"][1] + pad_h
        bg_color = random.choice([(255, 255, 255), (252, 252, 250), (250, 250, 245)])
        text_color = random.choice(GLOBAL_CONFIG["TEXT_COLORS"])
        
        # Select Font
        font_path = random.choice(self.fonts)
        font_name = os.path.basename(font_path)
        
        # 4. Fit Font Size
        font_size = CONFIG["BASE_FONT_SIZE"]
        font = ImageFont.truetype(font_path, font_size)
        target_w = CONFIG["FINAL_SIZE"][0] - 20 
        
        # Shrink font if text is too wide
        while font.getlength(display_text) > target_w and font_size > 30:
            font_size -= 2
            font = ImageFont.truetype(font_path, font_size)

        # 5. Draw Text
        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)
        
        text_bbox = draw.textbbox((0, 0), display_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        x = (W - text_w) // 2
        y = (H - text_h) // 2

        draw.text((x, y), display_text, font=font, fill=text_color)

        # 6. Apply Wave/Handwriting Effect
        img_np = np.array(img)
        

        # 7. Rotate & Crop
        img = Image.fromarray(img_np)
        if random.random() < CONFIG["PROBS"]["ROTATE"]:
            angle = random.uniform(-2, 2) 
            img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=bg_color)

        left = (W - CONFIG["FINAL_SIZE"][0]) // 2
        top = (H - CONFIG["FINAL_SIZE"][1]) // 2
        img = img.crop((left, top, left + CONFIG["FINAL_SIZE"][0], top + CONFIG["FINAL_SIZE"][1]))

        # 8. Noise Pipeline
        img_np = np.array(img)
        
        # Morphology (Thicken Ink)
        if random.random() < CONFIG["PROBS"]["MORPHOLOGY"]:
            kernel = np.ones((2,2), np.uint8) 
            img_inv = cv2.bitwise_not(img_np)
            # Dilate = Thicker ink
            morphed = cv2.dilate(img_inv, kernel, iterations=1)
            img_np = cv2.bitwise_not(cv2.addWeighted(img_inv, 0.7, morphed, 0.3, 0))

        # Color/Ink Noise
        if random.random() < CONFIG["PROBS"]["INK_NOISE"]:
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            noise = np.random.randint(0, 100, (img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
            v[(v < 200) & (noise > 70)] += 50 
            img_np = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)

        # Salt & Pepper
        if random.random() < CONFIG["PROBS"]["SALT_PEPPER"]:
            noise = np.random.rand(img_np.shape[0], img_np.shape[1])
            img_np[noise < 0.001] = 0; img_np[noise > 0.999] = 255

        # Blur (Softens edges)
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0)

        return Image.fromarray(img_np), raw_label, font_name

    def run(self):
        out_csv = os.path.join(CONFIG["OUTPUT_DIR"], "labels.csv")
        
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "text", "font"])

            print(f"Generating {CONFIG['SAMPLE_SIZE']} Arabic samples from {len(self.dataset)} unique instructions...")

            for i in range(CONFIG["SAMPLE_SIZE"]):
                try:
                    img, label, font_used = self.generate_image()
                    
                    fname = f"ara_{i:06d}.png"
                    img.save(os.path.join(CONFIG["OUTPUT_DIR"], fname))
                    
                    writer.writerow([fname, label, font_used])
                    
                    if i % 500 == 0 and i > 0: 
                        print(f"Progress: {i}/{CONFIG['SAMPLE_SIZE']}")
                except Exception as e:
                    print(f"Error skipping sample {i}: {e}")
            
            print("Done! Check folder:", CONFIG["OUTPUT_DIR"])

if __name__ == "__main__":
    ArabicPrescriptionGenerator().run()