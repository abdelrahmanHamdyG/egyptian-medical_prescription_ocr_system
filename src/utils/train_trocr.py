import os
import pandas as pd
from glob import glob

# ==========================================
# 1. KHATT MAPPING DICTIONARY
# ==========================================
# This maps the codes (e.g., 'ba', 'aa') to Arabic characters (e.g., 'ب', 'ا')
KHATT_CODE_MAP = {
    'aa': 'ا', 'ba': 'ب', 'ta': 'ت', 'th': 'ث', 'ja': 'ج',
    'ha': 'ح', 'kh': 'خ', 'da': 'د', 'dh': 'ذ', 'ra': 'ر',
    'za': 'ز', 'se': 'س', 'sh': 'ش', 'sa': 'ص', 'de': 'ض',
    'to': 'ط', 'zha': 'ظ', 'ay': 'ع', 'gh': 'غ', 'fa': 'ف',
    'ka': 'ق', 'ke': 'ك', 'la': 'ل', 'ma': 'م', 'na': 'ن',
    'he': 'ه', 'wa': 'و', 'ya': 'ي',
    'teE': 'ة', 'al': 'ال', 'laa': 'لا',
    'ae': 'ئ', 'ah': 'ء', 'ao': 'ؤ', 'aaE': 'أ', 'aE': 'إ', 'aaa': 'آ',
    'sp': ' ', 'dot': '.', 'col': ':', 'com': '،', 'ques': '؟', 
    'quo': '"', 'sem': '؛', 'dash': '-', 'scr': '' 
}

# ==========================================
# 2. CONFIGURATION & PATHS
# ==========================================
PATHS = {
    "SYNTH_DRUGS": "synth_prescriptions",
    "SYNTH_ARABIC": "synth_arabic_final",
    "SYNTH_MIXED": "synth_dual_lang_safe",
    
    # Check these paths on your machine
    "KHATT_IMGS": r"khatt\archive\Train_deskewed\Train_deskewed",
    "KHATT_LABELS": r"khatt\archive\Train.csv",
    
    "IAM_IMGS_ROOT": r"archive\iam_words\words", 
    "IAM_LABELS": r"archive\iam_words\words.txt"
}
# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def load_synthetic_folder(folder_path):
    """Loads your synthetic folders (csv + images)"""
    csv_path = os.path.join(folder_path, "labels.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {folder_path}: CSV not found.")
        return []
        
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        full_path = os.path.join(folder_path, row['filename'])
        
        if os.path.exists(full_path):
            # Check if it is the Mixed dataset (needs concatenation)
            if "dual" in str(row['filename']) or "dual" in folder_path:
                label = str(row['eng_text']) + " " + str(row['ara_text'])
            else:
                label = str(row['text'])
                
            data.append({"file_name": full_path, "text": label})
    
    print(f"Loaded {len(data)} from {folder_path}")
    return data

def decode_khatt_codes(row_codes):
    """Converts a list of codes ['dh', 'he', 'ba'] into an Arabic string 'ذهب'"""
    arabic_chars = []
    for code in row_codes:
        code = str(code).strip()
        if code in KHATT_CODE_MAP:
            arabic_chars.append(KHATT_CODE_MAP[code])
    return "".join(arabic_chars)

def load_khatt_subset(image_folder, csv_path, limit=3500):
    """Loads KHATT data, converts labels, and limits to 3.5k samples"""
    if not os.path.exists(csv_path):
        print(f"Skipping KHATT: CSV not found at {csv_path}")
        return []

    print("Loading KHATT dataset...")
    # Read CSV (header=0 because your file has 0,1,2... header)
    df = pd.read_csv(csv_path, header=0)
    
    khatt_data = []
    
    for _, row in df.iterrows():
        
        # Stop if we reached the limit
        if len(khatt_data) >= limit:
            break
            
        # 1. Get Filename (Column 0)
        filename = row.iloc[0]
        filename=filename.replace('.tif','.jpg')
        full_path = os.path.join(image_folder, filename)
        
        
        # 2. Check if image exists
        if not os.path.exists(full_path):
            print(f"we are continuing as the full path is {full_path}")
            continue
            
        # 3. Get Codes (Columns 1 onwards)
        codes = row.iloc[1:].values
        
        # 4. Convert Codes to Arabic Text
        arabic_label = decode_khatt_codes(codes)
        
        # 5. Add if label is not empty
        if arabic_label.strip():
            khatt_data.append({"file_name": full_path, "text": arabic_label})
            
    print(f"Loaded {len(khatt_data)} samples from KHATT")
    return khatt_data


def load_iam_subset(root_dir, labels_path, limit=3500):
    if not os.path.exists(labels_path): return []
    print("Loading IAM dataset...")
    
    iam_data = []
    with open(labels_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if len(iam_data) >= limit: break
        if line.startswith("#"): continue
        
        parts = line.strip().split()
        if len(parts) < 9 or parts[1] != 'ok': continue # Only 'ok' quality

        word_id = parts[0]
        text_label = parts[-1]
        
        # Build Path: a01-000u-00-00 -> a01 / a01-000u / ...png
        id_parts = word_id.split('-')
        folder1 = id_parts[0]
        folder2 = f"{id_parts[0]}-{id_parts[1]}"
        full_path = os.path.join(root_dir, folder1, folder2, word_id + ".png")
        
        if os.path.exists(full_path):
            iam_data.append({"file_name": full_path, "text": text_label})
        
    print(f"Loaded {len(iam_data)} samples from IAM")
    return iam_data
# ==========================================
# 4. MAIN EXECUTION
# ==========================================

all_data = []

# 1. Load Synthetic Data
all_data.extend(load_synthetic_folder(PATHS["SYNTH_DRUGS"]))
all_data.extend(load_synthetic_folder(PATHS["SYNTH_ARABIC"]))
all_data.extend(load_synthetic_folder(PATHS["SYNTH_MIXED"]))

# 2. Load KHATT Data (Limited to 3500)
all_data.extend(load_khatt_subset(PATHS["KHATT_IMGS"], PATHS["KHATT_LABELS"], limit=3500))
all_data.extend(load_iam_subset(PATHS["IAM_IMGS_ROOT"], PATHS["IAM_LABELS"], limit=4500))
# Verification
print("-" * 30)
print(f"TOTAL DATASET SIZE: {len(all_data)}")
if len(all_data) > 0:
    print("Last added sample:", all_data[-1])
print("-" * 30)